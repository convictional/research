import json

from ..settings import settings
from pydantic import BaseModel, Field


real_world_test_cases_file = (
    settings.input_path / "calculator_tools_real_world_tests" / "calculator_tools_real_world_test_cases.json"
)


class UserQueryTestCase(BaseModel):
    query: str = Field(..., title="The user query")
    goal: str = Field(..., title="The goal of the user query")


def get_user_query_test_cases() -> list[UserQueryTestCase]:
    print(f"Loading user query test cases from {real_world_test_cases_file}...")

    with open(real_world_test_cases_file, "r") as file:
        data = json.load(file)

    test_cases = [UserQueryTestCase(query=case["user_query"], goal=case["user_goal"]) for case in data["test_cases"]]

    print(f"Loaded {len(test_cases)} user query test cases.")

    return test_cases
