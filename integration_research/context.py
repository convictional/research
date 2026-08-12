import os

import pandas as pd
import yaml
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

PROJECT_ID = os.environ.get("GCP_PROJECT", "")  # set to your BigQuery project


def setup_bigquery_client():
    project_id = PROJECT_ID
    try:
        client = bigquery.Client(project=project_id)
        return client
    except DefaultCredentialsError:
        print("No default credentials found. Please run `gcloud auth application-default login` to authenticate.")
        return None


def get_bq_meta_string(client):
    bq_dataset_query = """
        SELECT *
        FROM `{project}.region-us.INFORMATION_SCHEMA.TABLES`
    """.format(project=PROJECT_ID)
    job = client.query(bq_dataset_query)
    bq_meta_data = job.to_dataframe()
    bq_meta_data = bq_meta_data[["table_schema", "table_name"]].copy()
    bq_meta_data = bq_meta_data[bq_meta_data["table_schema"].str.startswith("prod_")]
    grouped = bq_meta_data.groupby("table_schema")["table_name"].apply(list).reset_index()
    bq_meta_string = "; ".join(
        [f"{row['table_schema']}: [{', '.join(row['table_name'])}]" for _, row in grouped.iterrows()]
    )
    return bq_meta_string


def load_db_context_yml():
    # NOTE: This YAML described the real BigQuery warehouse schema and was removed before
    # open-sourcing. Supply your own schema description at this path to run the experiment.
    db_context_file_path = "context/db_context.yml"
    with open(db_context_file_path, "r") as file:
        db_yml_content = file.read()
    yml_dict = yaml.safe_load(db_yml_content)
    return yml_dict


def load_guru_context(client):
    guru_card_query = """
        SELECT
            preferred_phrase AS card_title,
            last_verified_by_user AS card_last_verified_by_user,
            CAST(date_created AS DATE) AS card_created_date,
            last_modified AS card_last_modified,
            content AS card_content,
            share_status AS card_share_status,
            verification_interval AS card_verification_interval,
            verification_state AS card_verification_state
        FROM
            guru.card
        WHERE
            NOT _fivetran_deleted
            AND (
                NOT archived
                OR archived IS NULL
            )
    """
    job = client.query(guru_card_query)
    raw_guru_df = job.to_dataframe()
    return raw_guru_df


def process_guru_context(raw_guru_df):
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
    ]
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunk_size = 5000
    chunk_overlap = 300
    character_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    processed_rows = []
    for _, row in raw_guru_df.iterrows():
        initial_splits = html_splitter.split_text(row["card_content"])
        for document in initial_splits:
            chunks = character_splitter.split_text(document.page_content)
            for chunk in chunks:
                new_row = row.copy()
                new_row["card_content"] = chunk
                new_row["card_header"] = ", ".join([f"{k}: {v}" for k, v in document.metadata.items()])
                processed_rows.append(new_row)
    guru_df = pd.DataFrame(processed_rows).reset_index(drop=True)
    return guru_df


def vectorize_guru_context(guru_df):
    guru_loader = DataFrameLoader(guru_df, page_content_column="card_content")
    guru_docs = guru_loader.load()
    guru_vectorized_content = FAISS.from_documents(guru_docs, OpenAIEmbeddings())
    return guru_vectorized_content


def vectorize_db_context(yml_dict):
    table_data = []
    for dataset in yml_dict["datasets"]:
        dataset_name = dataset["name"]
        for model in dataset.get("models", []):
            table_name = model["name"]
            table_description = model.get("description", "")
            table_path = f"{dataset_name}.{table_name}"
            columns = []
            for column in model.get("columns", []):
                column_name = column["name"]
                column_type = column.get("data_type", "")
                column_description = column.get("description", "")
                columns.append(f"{column_name} - {column_type} - {column_description}")
            table_info = {
                "table_path": table_path,
                "table_name": table_name,
                "table_description": table_description,
                "columns": ", ".join(columns),
            }
            table_data.append(table_info)
    db_context_df = pd.DataFrame(table_data)
    db_loader = DataFrameLoader(db_context_df, page_content_column="columns")
    db_docs = db_loader.load()
    db_vectorized_context = FAISS.from_documents(db_docs, OpenAIEmbeddings())
    return db_vectorized_context


def setup_context():
    client = setup_bigquery_client()
    bq_meta_string = get_bq_meta_string(client)
    yml_dict = load_db_context_yml()
    raw_guru_df = load_guru_context(client)
    guru_df = process_guru_context(raw_guru_df)
    guru_vectorized_content = vectorize_guru_context(guru_df)
    db_vectorized_context = vectorize_db_context(yml_dict)
    return bq_meta_string, guru_vectorized_content, db_vectorized_context


bq_meta_string, guru_vectorized_content, db_vectorized_context = setup_context()
