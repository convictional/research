from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd

from .settings import settings


class TestCaseFilter(BaseModel):
    key: Literal["id", "name"] = Field(..., description="The key to filter the test cases")
    value: Union[str, int] = Field(..., description="The value to filter the test cases")

    @model_validator(mode='after')
    def validate_value_type(self):
        if self.key == 'id' and not isinstance(self.value, int):
            raise ValueError(f"Value must be an integer when key is 'id'. Got {type(self.value)} instead.")

        if self.key == 'name' and not isinstance(self.value, str):
            raise ValueError(f"Value must be a string when key is 'name'. Got {type(self.value)} instead.")

        return self


class TestCaseCattachment(BaseModel):
    file_name: str = Field(..., description="The name of the file attachment")
    description: str = Field(..., description="The description of the file attachment")
    data: Optional[str] = Field(None, description="The data of the file attachment")

    @field_validator("file_name")
    @classmethod
    def validate_file_name_file_exists(cls, value):
        # Check that each attachment file exists in the input directory
        attachment_path = settings.input_path / "attachment_files" / value
        if not attachment_path.exists():
            raise ValueError(f"Attachment '{value}' does not exist in the attachment files input directory.")
        return value

    @model_validator(mode='after')
    def load_attachment_data(self):
        """Load the data for the attachment file"""
        # For CSV files
        if self.file_name.endswith(".csv"):
            self.data = pd.read_csv(settings.input_path / "attachment_files" / self.file_name).to_csv(index=False)
        else:
            print(f"No data loaded for file name {self.file_name}. Attachment file type is not supported.")

        return self


class TestCase(BaseModel):
    id: int = Field(..., description="The unique identifier for the test case")
    name: str = Field(..., description="The name of the test case")
    results_csv_output_file_name: str = Field(
        ..., description="The name of the csv file that will contain the results of the test case"
    )
    user_query: str = Field(..., description="The user query for the test case")
    attachments: list[TestCaseCattachment] = Field(..., description="The list of file attachments for the test case")
    run_duration_s: Optional[int] = Field(None, description="The duration in seconds that the test case took to run")
    num_calculation_steps: Optional[int] = Field(None, description="The number of calculation steps for the test case")
    num_messages: Optional[int] = Field(None, description="The number of final messages for the test case")
    num_llm_responses: Optional[int] = Field(None, description="The number of LLM responses for the test case")
    cumulative_num_input_tokens: Optional[int] = Field(
        None, description="The cumulative number of input tokens for the test case"
    )
    cumulative_num_output_tokens: Optional[int] = Field(
        None, description="The cumulative number of output tokens for the test case"
    )
    last_llm_response_num_input_tokens: Optional[int] = Field(
        None, description="The number of input tokens for the last LLM response"
    )

    @field_validator("results_csv_output_file_name")
    @classmethod
    def validate_results_csv_output_file_name(cls, value):
        # Check that results_csv_output_file_name does not end with any extension at all
        if "." in value:
            raise ValueError(
                f"Test case 'results_csv_output_file_name' must not contain any file extension. Got '{value}' instead."
            )
        return value


class LLMCalculationRequest(BaseModel):
    tool_use_id: Optional[str] = Field(None, description="The ID of the tool use")
    id: str = Field(..., description="The identifier for the specific calculation given the tool use ID")
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="The arithmetic operation to perform"
    )
    operands: List[float] = Field(
        description="Array/list of the numbers to operate on. Each element must be a single number. For add: sum all numbers. For subtract: start with first number and subtract the rest. For multiply: multiply all numbers. For divide: start with first number and divide by the rest. CANNOT divide by zero.",
        min_items=2,
        max_items=100,
    )
    context: str = Field(description="Explanation of what this calculation represents")
    result: Optional[str] = Field(None, description="The result of the calculation")

    def calculate(self) -> None:
        """
        Calculate the result of the arithmetic operation on the operands
        and assign it to the result field.
        """
        operands = self.operands

        if len(operands) < 2:
            raise ValueError("At least 2 operands are required")

        try:
            if self.operation == "add":
                calculated_result = sum(operands)
            elif self.operation == "subtract":
                # Start with the first value and subtract all others
                calculated_result = operands[0]
                for operand in operands[1:]:
                    calculated_result -= operand
            elif self.operation == "multiply":
                # Multiply all values together
                calculated_result = 1
                for operand in operands:
                    calculated_result *= operand
            elif self.operation == "divide":
                # Start with the first value and divide by all others
                calculated_result = operands[0]
                for operand in operands[1:]:
                    if operand == 0:
                        raise ValueError("Cannot divide by zero")
                    calculated_result /= operand

            # Assign the calculated result to the result field
            self.result = str(calculated_result)

        except Exception as e:
            # Optionally, handle the error and still set a result
            self.result = f"Error: {str(e)}"
            raise  # Re-raise the exception if you want calling code to handle it


class AuditorResponse(BaseModel):
    analysis: str = Field(..., description="The auditor's analysis of the messages")
    request_duration_s: Optional[int] = Field(
        None, description="The duration in seconds that the audit request took to process"
    )
