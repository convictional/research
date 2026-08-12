from pydantic import BaseModel, Field
from google.cloud.bigquery import Client
import pandas as pd

from .utils.instruct_llm import get_async_instructor_client
from .config.prompts import DESCRIBE_DATA_SYSTEM_PROMPT, DESCRIBE_DATA_USER_PROMPT
from .config.experiment_settings import settings


class ColumnDescription(BaseModel):
    column_name: str = Field(..., description="The name of the column")
    description: str = Field(..., description="A brief description of the column. DO NOT include the column name.")


class DataDescription(BaseModel):
    table_description: str = Field(..., description="A description of the data in the table")
    columns: list[ColumnDescription]


bq_client = Client(settings.gcp_project)


def get_sample(locator: str, sample_size: int) -> pd.DataFrame | None:
    try:
        df = bq_client.query("SELECT * FROM `{}` LIMIT {}".format(locator, sample_size)).to_dataframe()
    except Exception as e:
        # This is a bit of a catch-all but we don't want to crash the whole process.
        # In testing, it looks like an error occurs when a view references a table that no longer exists
        print(f"Error getting sample from {locator}: {e}")
        return None
    return df


instructor_client = get_async_instructor_client()


async def llm_describe_data(
    table_locator: str,
    cur_description: str,
    sample_size: int = 10,
) -> DataDescription | None:
    print(f"Getting description for table {table_locator}")
    pickle_path = settings.output_path / "data_descriptions" / f"{table_locator}.pkl"
    pickle_path.parent.mkdir(parents=True, exist_ok=True)

    df = get_sample(table_locator, sample_size)

    if df is None or df.empty:
        return None

    try:
        new_description = await instructor_client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.1,
            max_tokens=4096,
            messages=[
                {
                    "role": "system",
                    "content": DESCRIBE_DATA_SYSTEM_PROMPT.format(
                        sample_size=sample_size,
                    ),
                },
                {
                    "role": "user",
                    "content": DESCRIBE_DATA_USER_PROMPT.format(
                        df=df.head(sample_size),
                        description=cur_description if cur_description else "No description provided",
                    ),
                },
            ],
            response_model=DataDescription,
        )
    except Exception as e:
        print(f"Error generating description for {table_locator}: {e}")
        new_description = DataDescription(table_description=cur_description, columns=[])

    return new_description
