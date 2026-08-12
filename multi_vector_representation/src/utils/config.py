import os
from dataclasses import dataclass


@dataclass
class Config:
    db_connection_string: str
    colbert_model_name: str
    max_sequence_length: int
    embedding_batch_size: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            db_connection_string=os.getenv(
                "DATABASE_URL", "postgresql://localhost/decide_development"
            ),
            colbert_model_name=os.getenv("COLBERT_MODEL", "jinaai/jina-colbert-v2"),
            max_sequence_length=int(os.getenv("MAX_SEQ_LENGTH", "8192")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
        )
