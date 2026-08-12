from langchain_core.pydantic_v1 import BaseModel, Field


class SaveCheckedMetrics(BaseModel):
    """Save metrics checked for supportability in our data"""

    metric: str = Field(description="The metric that was just checked")
    supportability: str = Field(description="supported or not_supported as determined in the chain of thought")
    tables: list[str] = Field(
        description="""IF supported, then the dataset.tables identified as needed for
                       the metric ELSE the needed data for the metric"""
    )


class SaveDatabaseSchema(BaseModel):
    """Save database schemas including datasets, tables and columns"""

    tables: list[dict] = Field(
        description="Each table in the returned database schema",
        example=[
            {
                "table_name": "String: the table's name",
                "dataset_name": "String: the name of the dataset where the table lives",
                "short_description": "String: one line description of the purpose of the table",
                "primary_key": "String: the primary key of the table",
                "columns": [
                    {
                        "column_name": "String: the name of the column",
                        "column_type": "String: the data type of the column",
                    }
                ],
            }
        ],
    )
