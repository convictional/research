from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field


class User(BaseModel):
    user_name: str = Field(..., title="The name of the user")
    org_profile: str = Field(..., title="The organization profile of the user")
    user_profile: str = Field(..., title="The user profile of the user")


class UsersData:
    """
    This class represents all of the users data that will be used in the experiment.
    """

    def __init__(self, input_file_path: Path):
        self.users: list[User] = self._load_user_data(input_file_path)
        print(f"Loaded data for {len(self.users)} users.")

    def _load_user_data(self, input_file_path: Path) -> list[User]:
        print(f"Loading user data from {input_file_path}...")

        data_records = pd.read_csv(input_file_path).to_dict(orient="records")
        return [
            User(
                user_name=record["user_name"],
                org_profile=record["org_system_prompt"],
                user_profile=record["user_generated_profile"],
            )
            for record in data_records
        ]
