from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class CalculationRequest(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="The arithmetic operation to perform"
    )
    operands: List[float] = Field(description="The numbers to operate on", min_items=2, max_items=2)
    context: str = Field(description="Explanation of what this calculation represents")
    result: Optional[str] = Field(None, description="The result of the calculation")

    def calculate(self) -> None:
        """
        Calculate the result of the given calculation and update the result field
        """
        if self.operation == "add":
            result = self.operands[0] + self.operands[1]
        elif self.operation == "subtract":
            result = self.operands[0] - self.operands[1]
        elif self.operation == "multiply":
            result = self.operands[0] * self.operands[1]
        elif self.operation == "divide":
            if self.operands[1] == 0:
                raise ValueError("Cannot divide by zero")
            result = self.operands[0] / self.operands[1]

        # Update the result field with string representation
        self.result = str(result)


class FinalAnswer(BaseModel):
    answer: str = Field(description="The final answer to the user query")
    explanation: str = Field(description="Explanation and description of the final answer")
