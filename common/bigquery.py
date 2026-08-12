import pandas as pd
from google.cloud.bigquery import Client, QueryJobConfig


def query_bq(query: str, gcp_project: str) -> pd.DataFrame:
    # Note: this requres you to have setup application-default auth for gcloud, use `make auth` to do this!
    client = Client(gcp_project)
    job_config = QueryJobConfig(use_query_cache=False)
    query_job = client.query(query, job_config=job_config)
    # Return a dataframe
    # If you need a list of dicts, just use df.to_dict(orient="records") wherever you need it,
    # after calling this function.
    # Variations in return types is a recipe for bugs.
    return query_job.to_dataframe()
