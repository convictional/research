from datetime import datetime
from json import JSONEncoder
from typing import Any

from pydantic import BaseModel


class Encoder(JSONEncoder):
    """
    Class for json encoding
    """

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        return JSONEncoder.default(self, obj)


def to_json(data: Any) -> str:
    """
    This function mimics logic in the app, and is needed when setting up prompt templates.
    """
    return Encoder().encode(data)
