from datetime import datetime
from operator import itemgetter

from context import bq_meta_string
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from prompts import (
    _DATABASE_EXPERT,
    _SQL_ERROR_PROMPT,
    _SQL_EXPERT_PROMPT,
    _SQL_EXTRACTION_PROMPT,
    CHAIN_OF_THOUGHT_PROMPT,
    DATA_ANALYST_PROMPT,
)
from tools import SaveDatabaseSchema


class Metric:
    def __init__(
        self,
        description,
        query_client,
        vectorized_context=None,
        model_name="gpt-4-turbo-preview",
        temperature=0.1,
        parent_decision=None,
        tools=SaveDatabaseSchema,
    ):
        self.parent_decision = parent_decision
        self.description = description
        self.model_name = model_name
        self.temperature = temperature
        self.openai_model = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        self.anthropic_model = ChatAnthropic(temperature=self.temperature, model_name="claude-3-opus-20240229")
        self.tools = tools
        self.memory = ConversationBufferMemory(return_messages=True)
        self.vectorized_context = vectorized_context
        self.client = query_client
        self.long_context = None
        self.temp_context = None
        self.schema = None
        self.schema_json = None
        self.chain_of_thought = None
        self.query = None
        self.query_results = None
        self.analysis = None
        self.context_retriever_shotgun = self.vectorized_context.as_retriever(search_kwargs={"k": 10})
        self.context_retriever_sniper = self.vectorized_context.as_retriever(search_kwargs={"k": 2})

    def get_relevant_long_context(self):
        self.long_context = self.context_retriever_shotgun.get_relevant_documents(self.description)
        return self.long_context

    def get_relevant_short_context(self, input):
        self.temp_context = self.context_retriever_sniper.get_relevant_documents(input)
        return self.temp_context

    def get_database_schemas(self):
        if not self.long_context:
            self.get_relevant_long_context()
        database_expert_prompt = PromptTemplate.from_template(_DATABASE_EXPERT).format(
            metric=self.description,
            context="{context}",
            database_metadata=bq_meta_string,
            format_instructions="{format_instructions}",
        )
        parser = JsonOutputParser(pydantic_object=self.tools)
        prompt = PromptTemplate(
            template=database_expert_prompt,
            input_variables=["context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        model = self.openai_model
        chain = prompt | model | parser
        result = chain.invoke({"context": self.long_context})
        self.schema_json = result
        queries = self.write_sample_queries(self.schema_json)
        for table_name, query in queries.items():
            try:
                sample_results = self.get_sample_data(query)
                for table_schema in self.schema_json["tables"]:
                    if table_schema["table_name"] == table_name:
                        table_schema["sample_results"] = sample_results
            except Exception as e:
                print(f"Error fetching sample data for {table_name}: {e}")
                raise e
        self.schema = str(self.schema_json).replace("{", "[").replace("}", "]")
        return self.schema

    def write_sample_queries(self, schema_json):
        tables = schema_json["tables"]
        queries = {}
        for table in tables:
            dataset_name = table["dataset_name"]
            table_name = table["table_name"]
            column_names = ", ".join([column["column_name"] for column in table["columns"]])
            query = f"SELECT {column_names} FROM `{dataset_name}.{table_name}` LIMIT 10"
            queries[table_name] = query
        return queries

    def get_sample_data(self, query):
        try:
            job = self.client.query(query)
            samples = job.to_dataframe()
            return samples.to_dict("records")
        except Exception as e:
            print(f"Error in processing query: {e}")
            raise e

    def write_query(self):
        now = datetime.now()
        formatted_now = now.strftime("%b %d, %Y")
        if not self.schema:
            self.get_database_schemas()
        sql_expert_prompt = PromptTemplate.from_template(_SQL_EXPERT_PROMPT).format(
            metric=self.description, context=self.schema, current_date=formatted_now
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", CHAIN_OF_THOUGHT_PROMPT), ("human", sql_expert_prompt)]
        ).format()
        self.chain_of_thought = self.openai_model.invoke(prompt).content
        return self.chain_of_thought

    def correct_query(self, query, context, error):
        if not self.schema:
            self.get_database_schemas()
        error_context = str(context).replace("{", "[").replace("}", "]")
        error = str(error).replace("{", "[").replace("}", "]")
        sql_error_prompt = PromptTemplate.from_template(_SQL_ERROR_PROMPT).format(
            current_query=query, error=error, context=error_context
        )
        self.memory.load_memory_variables({})
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CHAIN_OF_THOUGHT_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", sql_error_prompt),
            ]
        )
        chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history")
            )
            | prompt
            | self.openai_model
        )
        inputs = {"input": "Begin!"}
        self.chain_of_thought = chain.invoke(inputs).content
        self.memory.save_context(inputs, {"output": self.chain_of_thought})
        return self.chain_of_thought

    def extract_query(self):
        if not self.chain_of_thought:
            self.write_query()
        self.chain_of_thought = str(self.chain_of_thought).replace("{", "[").replace("}", "]")
        sql_extraction_prompt = ChatPromptTemplate.from_messages(
            [("system", _SQL_EXTRACTION_PROMPT), ("human", self.chain_of_thought)]
        ).format()
        extracted_query_response = self.openai_model.invoke(sql_extraction_prompt)
        _raw_query = extracted_query_response.content
        _raw_query = _raw_query.replace(";", " ").replace("LIMIT 1000", "")
        self.query = f"{_raw_query} LIMIT 1000"
        return self.query

    def execute_query(self, max_attempts=3):
        if not self.query:
            self.extract_query()
        attempt = 0
        while attempt < max_attempts:
            try:
                job = self.client.query(self.query)
                self.query_results = job.to_dataframe()
                print(f"Query executed successfully on attempt {attempt + 1}.")
                return self.query_results
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt}: Error executing query - {e}\nTrying to correct...\nQuery: {self.query}")
                self.get_relevant_short_context(input=(self.query + str(e)))
                self.temp_context = str(self.temp_context).replace("{", "[").replace("}", "]")
                error = str(e).replace("{", "[").replace("}", "]")
                self.correct_query(query=self.query, error=error, context=self.temp_context)
                corrected_query = self.extract_query()
                if corrected_query:
                    print(f"Query corrected on attempt {attempt}. New query: {corrected_query}")
                    self.query = corrected_query
                else:
                    print(f"Failed to correct query after {attempt} attempts. Last error: {e}")
                    raise e
        raise Exception(f"All {max_attempts} attempts to execute and correct the query have failed.")

    def data_analyst(self):
        if self.query_results is not None and not self.query_results.empty:
            self.get_relevant_short_context(input=self.query)
            results = self.query_results.to_dict("records")
            prompt = ChatPromptTemplate.from_messages([("system", DATA_ANALYST_PROMPT)])
            chain = create_stuff_documents_chain(self.openai_model, prompt)
            self.analysis = chain.invoke(
                {
                    "decision": self.parent_decision,
                    "metric_description": self.description,
                    "query_results": results,
                    "query": self.query,
                    "context": self.temp_context,
                }
            )
        else:
            self.query_results = self.execute_query()
            results = self.query_results.to_dict("records")
            self.get_relevant_short_context(input=self.query)
            prompt = ChatPromptTemplate.from_messages([("system", DATA_ANALYST_PROMPT)])
            chain = create_stuff_documents_chain(self.openai_model, prompt)
            self.analysis = chain.invoke(
                {
                    "decision": self.parent_decision,
                    "metric_description": self.description,
                    "query_results": results,
                    "query": self.query,
                    "context": self.temp_context,
                }
            )
        return self.analysis
