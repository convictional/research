import pandas as pd
from google.cloud.bigquery import Client, QueryJobConfig

from ..settings import settings


def query_bq(query: str) -> pd.DataFrame:
    # Note: this requres you to have setup application-default auth for gcloud, use `make auth` to do this!
    client = Client(settings.gcp_project)
    job_config = QueryJobConfig(use_query_cache=False)
    query_job = client.query(query, job_config=job_config)
    return query_job.to_dataframe()
