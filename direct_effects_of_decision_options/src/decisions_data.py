from pydantic import BaseModel, Field
import json


from .settings import settings


class DecisionOption(BaseModel):
    title: str = Field(..., title="The title of the decision option")
    description: str = Field(None, title="The description of the decision option")


class DecisionCriteria(BaseModel):
    title: str = Field(..., title="The title of the decision criteria")
    description: str = Field(None, title="The description of the decision criteria")


class Decision(BaseModel):
    title: str = Field(..., title="The title of the decision")
    goals: str = Field(None, title="The goals of the decision")
    options: list[DecisionOption] = Field(..., title="The decision options")
    criteria: list[DecisionCriteria] = Field(..., title="The decision criteria")


def get_decision_data(decision_id: int) -> Decision:
    """
    Get decision data from the "database".
    This is mock data, so there is no actual database.
    The data is hardcoded in json format and fetched based on the decision_id.
    """
    print("Getting decision data...")
    decision_data_file_name = get_decision_data_file_name(decision_id)

    print(f"Loading decision data from {decision_data_file_name}...")
    input_file_path = settings.input_path / decision_data_file_name
    decision_raw_data = json.loads(input_file_path.read_text())

    decision = Decision(
        title=decision_raw_data["title"],
        goals=decision_raw_data["goals"],
        options=[
            DecisionOption(title=option["title"], description=option["description"])
            for option in decision_raw_data["options"]
        ],
        criteria=[
            DecisionCriteria(title=criterion["title"], description=criterion["description"])
            for criterion in decision_raw_data["criteria"]
        ],
    )

    return decision


def get_decision_data_file_name(decision_id: int) -> str:
    """
    Get the decision data file path based on the decision_id.

    Decision ids:
    - 1: Decision to hire a new sales role
    """
    decision_id_to_file_name_map = {
        1: "decision_hire_sales_role.json",
        2: "decision_acquire_company.json",
        3: "decision_data_analytics_tech_stack.json",
        4: "decision_sell_dropship.json",
        5: "decision_new_meeting_bot_product.json",
        6: "decision_new_international_tariffs.json",
        7: "decision_switch_llm_provider.json",
        8: "decision_product_pivot.json",
    }

    return decision_id_to_file_name_map[decision_id]
