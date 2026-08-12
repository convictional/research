"""
Pydantic models for database schema.

These models provide type safety and validation for data moving in/out of PostgreSQL.
Based on migrations/001_create_training_schema.sql
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# =============================================================================
# EPISODES
# =============================================================================


class EpisodeCreate(BaseModel):
    """Data required to create a new training episode."""

    episode_num: int
    start_date: datetime
    end_date: datetime
    corpus_snapshot_date: datetime
    num_new_chunks: int = 0
    num_updated_chunks: int = 0
    status: str = "pending"


class Episode(EpisodeCreate):
    """Complete episode model with database fields."""

    id: int
    created_at: datetime
    completed_at: datetime | None = None


# =============================================================================
# TRAINING PAIRS
# =============================================================================


class TrainingPairCreate(BaseModel):
    """Data required to create a training pair."""

    episode_id: int
    anchor_content_id: UUID
    positive_content_id: UUID
    negative_content_id: UUID | None = None
    pair_type: str
    mining_method: str
    source_family: str
    anchor_text: str
    positive_text: str
    negative_text: str | None = None
    is_in_replay_buffer: bool = False


class TrainingPair(TrainingPairCreate):
    """Complete training pair model with database fields."""

    id: int
    created_at: datetime


# =============================================================================
# ADAPTERS
# =============================================================================


class AdapterCreate(BaseModel):
    """Data required to create a new adapter."""

    adapter_id: str
    base_model: str
    episode_id: int
    sources: list[str]
    objective: str
    train_start_date: datetime
    train_end_date: datetime
    replay_pct: float
    hnsw_index_id: str | None = None
    lora_config: dict
    training_config: dict
    metrics: dict
    stability_delta: float | None = None
    status: str = "training"
    storage_path: str
    created_by: str


class Adapter(AdapterCreate):
    """Complete adapter model with database fields."""

    id: int
    created_at: datetime
    promoted_at: datetime | None = None

    def __init__(self, **data):
        # Parse JSON strings if needed (asyncpg returns JSONB as strings)
        import json

        if isinstance(data.get("sources"), str):
            data["sources"] = json.loads(data["sources"])
        if isinstance(data.get("lora_config"), str):
            data["lora_config"] = json.loads(data["lora_config"])
        if isinstance(data.get("training_config"), str):
            data["training_config"] = json.loads(data["training_config"])
        if isinstance(data.get("metrics"), str):
            data["metrics"] = json.loads(data["metrics"])

        super().__init__(**data)


# =============================================================================
# EVAL QUERIES
# =============================================================================


class EvalQueryCreate(BaseModel):
    """Data required to create an evaluation query."""

    query_text: str
    query_type: str
    difficulty: str
    expected_sources: list[str] | None = None
    ground_truth_content_ids: list[UUID] | None = None
    tags: list[str] = Field(default_factory=list)
    is_in_stability_set: bool = False


class EvalQuery(EvalQueryCreate):
    """Complete eval query model with database fields."""

    id: int
    created_at: datetime


# =============================================================================
# EVAL RESULTS
# =============================================================================


class EvalResultCreate(BaseModel):
    """Data required to create an evaluation result."""

    adapter_id: int | None = None
    eval_query_id: int
    retrieved_content_ids: list[UUID]
    scores: list[float]
    adapters_used: list[int] | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    recall_at_20: float | None = None
    ndcg_at_10: float | None = None
    ndcg_at_20: float | None = None
    latency_ms: int | None = None


class EvalResult(EvalResultCreate):
    """Complete eval result model with database fields."""

    id: int
    created_at: datetime


# =============================================================================
# CONTENT (from existing table)
# =============================================================================


class Content(BaseModel):
    """Content model from existing content table."""

    id: UUID
    created_at: datetime
    updated_at: datetime
    category: str
    source: str
    source_id: str
    title: str
    index_content: str
    author: str | None = None
    metadata: dict | str  # Can be dict or JSON string from asyncpg
    organization_id: UUID
    content_type: str
    source_url: str
    preview_content: str | None = None
    sharing: str
    tags: list[str] | str = Field(default_factory=list)  # Can be list or JSON string

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        # Parse JSON strings if needed
        import json

        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        if isinstance(data.get("tags"), str):
            data["tags"] = json.loads(data["tags"])

        super().__init__(**data)
