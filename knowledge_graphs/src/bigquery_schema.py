from tqdm import tqdm
import re
import pickle
from datetime import datetime
from typing import List
from pathlib import Path
from pydantic import BaseModel
from google.cloud.bigquery import Client, SchemaField
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, ScopeType, traverse_scope
import pandas as pd

from .ontology.ontology import BaseNode, BaseEdge
from .config.experiment_settings import settings
from .describe_data import llm_describe_data
from .utils.embeddings import aembed, embed_batch


class DataWarehouseNode(BaseNode):
    embedding: List[float]
    columns: List[str]


class QueryNode(BaseNode):
    embedding: List[float]
    user_email: str
    sql: str


class ColumnSchema(BaseModel):
    column_name: str
    column_type: str
    is_nullable: bool
    description: str | None

    def to_nl_description(self):
        return f"`{self.column_name}`, {self.full_description()}"

    def full_description(self) -> str:
        return f'{self.column_type} (is {"nullable" if self.is_nullable else "not nullable"}), Description: {self.description}'


class TableSchema(BaseModel):
    table_id: str
    locator: str
    description: str
    columns: list[ColumnSchema]

    def to_nl_description(self):
        return f"Table: `{self.locator}`, Description: {self.description}, Columns: {"\n".join([column.to_nl_description() for column in self.columns]) if self.columns else "No columns found"}"

    async def get_description(self):
        return await llm_describe_data(self.locator, self.to_nl_description(), sample_size=10)

    async def to_graph_node(self) -> DataWarehouseNode:
        llm_description = await self.get_description()
        if llm_description:
            description = llm_description.table_description
            embedding = await aembed(
                f"table: {llm_description.table_description} columns: {", ".join([column.description for column in llm_description.columns])}"
            )
            columns = {f"{column.column_name}: {column.description}" for column in llm_description.columns}
        else:
            description = self.description
            embedding = await aembed(f"table: {self.description}")
            columns = {f"{column.column_name}: {column.full_description()}" for column in self.columns}

        return DataWarehouseNode(
            name=self.locator,
            category="StructuredData",
            description=description,
            embedding=embedding,
            columns=columns,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            other_fields={column.column_name: column.full_description() for column in self.columns},
        )


class PickledBase:
    def save_pickle(self, data, pickle_path: Path):
        pickle_path = pickle_path

        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)

    def load_pickle(self, pickle_path: Path):
        pickle_path = pickle_path

        with open(pickle_path, "rb") as f:
            return pickle.load(f)


class BigQuerySchemaDetector(PickledBase):
    # Note: this requres you to have setup application-default auth for gcloud, use `make auth` to do this!
    client: Client = Client(settings.gcp_project)

    def run(self) -> list[TableSchema]:
        pickle_path: Path = settings.output_path / "bigquery_schema.pickle"
        # Pickling is only used in development to speed up each run
        if settings.is_env("development"):
            if pickle_path.exists():
                return self.load_pickle(pickle_path)
            else:
                schema = self.detect_schema()
                self.save_pickle(schema, pickle_path)
                return schema

        return self.detect_schema()

    def column_to_schema(self, column: SchemaField):
        return ColumnSchema(
            column_name=column.name,
            column_type=column.field_type,
            is_nullable=column.is_nullable,
            description=column.description,
        )

    def detect_schema(self):
        full_schema = []
        datasets = list(self.client.list_datasets())
        # Excluding the fivetran_differently_lessee_staging dataset because it contains ephemeral tables
        # that are not useful for the graph.
        excluded_datasets = ["fivetran_differently_lessee_staging"]

        for dataset in tqdm(datasets):
            if dataset.dataset_id in excluded_datasets:
                continue

            tables = list(self.client.list_tables(dataset))
            for table in tqdm(tables):
                table = self.client.get_table(table)
                try:
                    description = table.description or ""
                except AttributeError:
                    description = ""

                table_schema = TableSchema(
                    table_id=table.table_id,
                    locator=f"{table.project}.{table.dataset_id}.{table.table_id}",
                    description=description,
                    columns=[self.column_to_schema(column) for column in table.schema],
                )

                full_schema.append(table_schema)

        return full_schema


class Query(BaseModel):
    query_hash: str
    query: str
    user_email: str
    statement_type: str

    @property
    def source_tables(self):
        return self.extract_source_tables_from_query()

    def to_graph_node(self, embedding) -> QueryNode:
        description = f"A query executed by {self.user_email} in BigQuery."
        return QueryNode(
            name=self.query_hash,
            category="Query",
            description=description,
            sql=self.query,
            embedding=embedding,
            user_email=self.user_email,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            other_fields={"statement_type": self.statement_type, "user_email": self.user_email},
        )

    def to_graph_edges(self) -> List[BaseEdge]:
        edges = []
        if self.statement_type == "CREATE_TABLE_AS_SELECT":
            target_table = self.extract_target_table()
            if target_table:
                edges.append(
                    BaseEdge(
                        source=self.query_hash,
                        target=target_table,
                        name=self.statement_type,
                    )
                )

        for table in self.source_tables:
            if self.statement_type == "CREATE_TABLE_AS_SELECT":
                source = table
                target = self.query_hash
                name = "SELECTS_FROM_TABLE"
            else:
                source = self.query_hash
                target = table
                name = self.statement_type

            edges.append(
                BaseEdge(
                    source=source,
                    target=target,
                    name=name,
                )
            )

        return edges

    def extract_target_table(self):
        try:
            cleaned = re.sub(r"--.*", "", self.query)
            parsed = sqlglot.parse_one(cleaned, read="bigquery")

            if isinstance(parsed, sqlglot.expressions.Command) or isinstance(parsed, sqlglot.expressions.Create):
                if isinstance(parsed.this, sqlglot.expressions.UserDefinedFunction):
                    # We don't care about UDFs too much.
                    return None
                else:
                    target_table = ".".join([part.name for part in parsed.this.parts])

            return target_table

        except sqlglot.ParseError as e:
            print(f"Error parsing SQL query: {e}")
            return None

    def extract_source_tables_from_query(self):
        try:
            # Remove line comments starting with -- from the query
            cleaned = re.sub(r"--.*", "", self.query)
            parsed = sqlglot.parse_one(cleaned, read="bigquery")

            identifiers = [
                ".".join([part.name for part in source.parts])
                for scope in traverse_scope(parsed)
                for source in scope.sources.values()
                if isinstance(source, exp.Table) and not self.is_cte(source, scope)
            ]
        except sqlglot.ParseError as e:
            print(f"Error parsing query: {e}")
            identifiers = []

        return identifiers

    def is_cte(self, source: exp.Table, scope: Scope) -> bool:
        """
        Is the source a CTE?

        CTEs in the parent scope look like tables (and are represented by
        exp.Table objects), but should not be considered as such;
        otherwise a user with access to table `foo` could access any table
        with a query like this:

            WITH foo AS (SELECT * FROM target_table) SELECT * FROM foo

        """
        parent_sources = scope.parent.sources if scope.parent else {}
        ctes_in_scope = {
            name
            for name, parent_scope in parent_sources.items()
            if isinstance(parent_scope, Scope) and parent_scope.scope_type == ScopeType.CTE
        }

        return source.name in ctes_in_scope


class BigQueryHistory(PickledBase):
    client: Client = Client(settings.gcp_project)

    def run(self) -> list[Query]:
        pickle_path: Path = settings.output_path / "bigquery_history.pickle"
        if settings.is_env("development"):
            if pickle_path.exists():
                print("Loading Pickled BigQuery History...")
                query_history = self.load_pickle(pickle_path)
                return query_history
            else:
                print("Fetching BigQuery History...")
                jobs = self.extract_jobs(days_back=7)
                print("Transforming History to List of Query...")
                query_history = self.transform_jobs(jobs)
                print("Saving Pickled BigQuery History...")
                self.save_pickle(query_history, pickle_path)

        jobs = self.extract_jobs(days_back=7)
        return self.transform_jobs(jobs)

    def transform_jobs(self, jobs: pd.DataFrame, user_emails: List[str] = []) -> List[Query]:
        query_history: list[Query] = []

        for _, job in jobs.iterrows():
            if user_emails and job.user_email not in user_emails:
                continue

            query = Query(
                query_hash=job.query_hash,
                query=job.query,
                user_email=job.user_email,
                statement_type=job.statement_type or "UNKNOWN",
            )
            query_history.append(query)

        return query_history

    def extract_jobs(self, days_back: int = 7) -> pd.DataFrame:
        # Service account of the ETL tool whose writes we ignore when attributing queries.
        excluded_emails = ["etl-service-account@your-etl-vendor.iam.gserviceaccount.com"]
        jobs_query = f"""
        with query_history as (
            select
                job_type
                , query
                , to_hex(md5(cast(query as bytes))) AS query_hash
                , user_email
                , creation_time
                , statement_type
            from `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
            where state = 'DONE'
                and date(creation_time) between date_sub(current_date('UTC'), INTERVAL {days_back} DAY) and current_date('UTC')
                and job_type = 'QUERY' -- note that other job types do not have a query attribute so we have to filter them out here
                and user_email not in ('{"', '".join(excluded_emails)}')
        ), deduped as (
            select
                *
                , row_number() over (partition by query_hash order by creation_time desc) as rn
            from query_history
        )

        select
            job_type
            , query
            , query_hash
            , user_email
            , creation_time
            , statement_type
        from deduped
        where rn = 1
        """

        jobs = self.client.query(jobs_query)
        return jobs.to_dataframe()

    async def get_nodes(self, query_history: List[Query]) -> List[QueryNode]:
        pickle_path = settings.output_path / "bigquery_history_nodes.pickle"

        nodes = []
        if settings.is_env("development") and pickle_path.exists():
            nodes = self.load_pickle(pickle_path)
            return nodes

        embeddings = embed_batch([f"A query executed by {query.user_email} in BigQuery." for query in query_history])
        for query, embedding in tqdm(zip(query_history, embeddings), total=len(query_history)):
            query_node = query.to_graph_node(embedding)
            nodes.append(query_node)

        if settings.is_env("development"):
            self.save_pickle(nodes, pickle_path)

        return nodes

    def get_edges(self, query_history: List[Query]) -> List[BaseEdge]:
        pickle_path = settings.output_path / "bigquery_history_edges.pickle"

        if settings.is_env("development"):
            if pickle_path.exists():
                return self.load_pickle(pickle_path)
            else:
                edges = [edge for query in query_history for edge in query.to_graph_edges()]
                self.save_pickle(edges, pickle_path)
                return edges

        return [edge for query in query_history for edge in query.to_graph_edges()]
