import ast

from context import db_vectorized_context, guru_vectorized_content
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from metric import Metric
from prompts import (
    _ANALYSIS_SUMMARY_PROMPT,
    _METRIC_AVAILABILITY_PROMPT,
    _METRIC_BRAINSTORM_PROMPT,
    CHAIN_OF_THOUGHT_PROMPT,
    METRIC_EXTRACTION_PROMPT,
)
from tools import SaveCheckedMetrics


class Decision:
    def __init__(
        self,
        description,
        company_vectorized_context=guru_vectorized_content,
        database_vectorized_context=db_vectorized_context,
        model_name="gpt-4-turbo-preview",
        temperature=0.1,
        tools=SaveCheckedMetrics,
        query_client=None,
    ):
        self.description = description
        self.company_vectorized_context = company_vectorized_context
        self.database_vectorized_context = database_vectorized_context
        self.client = query_client
        self.model_name = model_name
        self.temperature = temperature
        self.openai_model = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        self.anthropic_model = ChatAnthropic(temperature=self.temperature, model_name="claude-3-opus-20240229")
        self.tools = tools
        self.memory = ConversationBufferMemory(return_messages=True)
        self.long_company_context = None
        self.short_company_context = None
        self.long_db_context = None
        self.short_db_context = None
        self.chain_of_thought = None
        self.metrics = None
        self.checked_metrics = []
        self.metrics_analysis = []
        self.summary_analysis = None
        self.company_context_retriever_shotgun = self.company_vectorized_context.as_retriever(search_kwargs={"k": 5})
        self.company_context_retriever_sniper = self.company_vectorized_context.as_retriever(search_kwargs={"k": 2})
        self.db_context_retriever_shotgun = self.database_vectorized_context.as_retriever(search_kwargs={"k": 5})
        self.db_context_retriever_sniper = self.database_vectorized_context.as_retriever(search_kwargs={"k": 5})

    def get_relevant_long_company_context(self):
        self.long_company_context = self.company_context_retriever_shotgun.get_relevant_documents(self.description)
        return self.long_company_context

    def get_relevant_short_company_context(self, input):
        self.short_company_context = self.company_context_retriever_sniper.get_relevant_documents(input)
        return self.short_company_context

    def get_relevant_long_database_context(self):
        self.long_db_context = self.db_context_retriever_shotgun.get_relevant_documents(str(self.metrics))
        return self.long_db_context

    def get_relevant_short_database_context(self, input):
        self.short_db_context = self.db_context_retriever_sniper.get_relevant_documents(input)
        return self.short_db_context

    def brainstorm_metrics(self):
        if not self.long_company_context:
            self.get_relevant_long_company_context()
        metric_brainstorm_prompt = PromptTemplate.from_template(_METRIC_BRAINSTORM_PROMPT).format(
            decision_description=self.description, context="{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", CHAIN_OF_THOUGHT_PROMPT), ("human", metric_brainstorm_prompt)]
        )
        chain = create_stuff_documents_chain(self.openai_model, prompt)
        self.chain_of_thought = chain.invoke({"context": self.long_company_context})
        return self.chain_of_thought

    def extract_metrics(self):
        if not self.chain_of_thought:
            self.brainstorm_metrics()
        prompt = ChatPromptTemplate.from_messages([("human", METRIC_EXTRACTION_PROMPT)]).format(
            chain_of_thought=self.chain_of_thought
        )
        _metrics = self.openai_model.invoke(prompt).content
        try:
            self.metrics = ast.literal_eval(_metrics)
        except (ValueError, SyntaxError, Exception) as e:
            print(f"Error: {str(e)}")
        return self.metrics

    def check_metric_supportability(self):
        if not self.metrics:
            self.extract_metrics()
        metrics = self.metrics
        for metric in metrics:
            self.get_relevant_short_database_context(metric)
            metric_availability_prompt = PromptTemplate.from_template(_METRIC_AVAILABILITY_PROMPT)
            metric_availability_prompt = metric_availability_prompt.format(
                metrics=metric,
                context="{context}",
                format_instructions="{format_instructions}",
            )
            parser = JsonOutputParser(pydantic_object=self.tools)
            prompt = PromptTemplate(
                template=metric_availability_prompt,
                input_variables=["context"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            model = self.openai_model
            chain = prompt | model | parser
            result = chain.invoke({"context": self.short_db_context})
            self.checked_metrics.append(result)
        return self.checked_metrics

    def analyze_metrics(self):
        if not self.checked_metrics:
            self.check_metric_supportability()
        self.metrics_analysis = []
        for metric_data in self.checked_metrics:
            metric_description = metric_data["metric"]
            metric_supported = "supported"
            if metric_supported == "supported":
                try:
                    temp_metric = Metric(
                        description=metric_description,
                        query_client=self.client,
                        vectorized_context=db_vectorized_context,
                        parent_decision=self.description,
                    )
                    analyst_response = temp_metric.data_analyst()
                    query = temp_metric.query
                    results = temp_metric.query_results
                except Exception as e:
                    analyst_response = f"Error processing data: {e}"
                    query = temp_metric.query
                    results = None
                self.metrics_analysis.append([metric_description, analyst_response, query, results])
            else:
                self.metrics_analysis.append([metric_description, metric_supported, metric_data["tables"], []])
        return self.metrics_analysis

    def summarize_findings(self):
        if not self.metrics_analysis:
            self.analyze_metrics()
        prompt = ChatPromptTemplate.from_messages([("system", _ANALYSIS_SUMMARY_PROMPT)])
        chain = create_stuff_documents_chain(self.openai_model, prompt)
        self.summary_analysis = chain.invoke(
            {
                "context": self.long_company_context,
                "decision_description": self.description,
                "metrics_analysis": self.metrics_analysis,
            }
        )
        return self.summary_analysis
