from models.base_model import BaseModel


class SimpleModel(BaseModel):
    def process_prompt(self, prompt: str) -> dict:
        # Placeholder implementation
        response = f"Simple Model Response: {prompt}"
        return {"response": response}
