from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def process_prompt(self, prompt: str) -> dict:
        pass
