"""Tortoise ORM models for AlignSim run persistence."""

from uuid import uuid4

from tortoise import fields
from tortoise.models import Model


class RunModel(Model):
    """A single game run (LLM or human)."""

    id = fields.UUIDField(pk=True, default=uuid4)
    scenario_name = fields.CharField(max_length=100)
    condition = fields.CharField(max_length=50)  # "condition1", "human_web", "engine_test"
    player_type = fields.CharField(max_length=20)  # "llm" or "human"
    model = fields.CharField(max_length=100, null=True)  # LLM model name, null for human
    harness = fields.CharField(max_length=20, null=True)  # "claude-code" or "pi"
    thinking = fields.CharField(max_length=20, null=True)  # reasoning level: off|minimal|low|medium|high|xhigh
    seed = fields.IntField()
    max_turns = fields.IntField()
    turns_played = fields.IntField(default=0)
    game_over_reason = fields.CharField(max_length=100, null=True)

    # Final scores (populated at game end)
    score_composite = fields.FloatField(null=True)
    score_mrr = fields.FloatField(null=True)
    score_churn = fields.FloatField(null=True)
    score_runway = fields.FloatField(null=True)
    final_mrr = fields.IntField(null=True)
    final_runway_turns = fields.FloatField(null=True)
    score_pareto = fields.FloatField(null=True)
    function_scores = fields.JSONField(null=True)

    # Layer 2 hidden alignment scores: nested dict per metric with score and raw values.
    # Columns added after the table's first creation (alignment_scores, harness, thinking) are
    # reconciled on existing DBs by the ADD COLUMN IF NOT EXISTS pass in database.py init_db()
    # (or `python -m alignsim migrate-db`) — generate_schemas() only CREATEs, it never ALTERs.
    alignment_scores = fields.JSONField(null=True)

    # Codebase version — last commit touching alignsim/
    engine_commit = fields.CharField(max_length=40, null=True)

    # Token usage per model: {model: {input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}}
    token_usage = fields.JSONField(null=True)

    # Flexible storage
    config = fields.JSONField(default=dict)  # calibration params, temperature, etc.
    metadata = fields.JSONField(default=dict)

    # Timestamps
    started_at = fields.DatetimeField(auto_now_add=True)
    finished_at = fields.DatetimeField(null=True)

    class Meta:
        table = "alignsim_runs"
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"Run({self.id}): {self.condition} seed={self.seed}"


class TurnSnapshotModel(Model):
    """Per-turn metrics snapshot."""

    id = fields.UUIDField(pk=True, default=uuid4)
    run = fields.ForeignKeyField("models.RunModel", related_name="turn_snapshots", on_delete=fields.CASCADE)
    turn = fields.IntField()

    # Core metrics
    mrr = fields.IntField()
    budget = fields.IntField()
    runway_turns = fields.FloatField()
    capacity_used = fields.IntField()
    capacity_available = fields.IntField()
    tech_debt_level = fields.FloatField()

    # Counts
    active_customers = fields.IntField()
    pipeline_customers = fields.IntField()
    bugs_injected = fields.IntField(default=0)
    bugs_fixed = fields.IntField(default=0)
    churn_count = fields.IntField(default=0)

    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "alignsim_turn_snapshots"
        unique_together = [("run", "turn")]
        indexes = [("run", "turn")]

    def __str__(self) -> str:
        return f"Snapshot(turn={self.turn}, mrr={self.mrr})"


class TurnActionModel(Model):
    """Individual action submitted during a turn."""

    id = fields.UUIDField(pk=True, default=uuid4)
    run = fields.ForeignKeyField("models.RunModel", related_name="turn_actions", on_delete=fields.CASCADE)
    turn = fields.IntField()

    action_type = fields.CharField(max_length=20)  # "build", "sell", "support", etc.
    action_data = fields.JSONField()  # Full action via model_dump()
    capacity = fields.IntField(default=0)
    was_valid = fields.BooleanField(default=True)
    rejection_reason = fields.TextField(null=True)

    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "alignsim_turn_actions"
        indexes = [("run", "turn"), ("run", "action_type")]

    def __str__(self) -> str:
        return f"Action(turn={self.turn}, type={self.action_type})"


class TurnEventModel(Model):
    """Narrative event from a turn."""

    id = fields.UUIDField(pk=True, default=uuid4)
    run = fields.ForeignKeyField("models.RunModel", related_name="turn_events", on_delete=fields.CASCADE)
    turn = fields.IntField()

    event_text = fields.TextField()
    event_type = fields.CharField(max_length=50, null=True)  # Parsed: "deal_won", "churn", etc.
    entity_id = fields.CharField(max_length=50, null=True)  # Parsed: customer/feature/bug ID

    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "alignsim_turn_events"
        indexes = [("run", "turn"), ("run", "event_type")]

    def __str__(self) -> str:
        return f"Event(turn={self.turn}, {self.event_text[:50]})"


class LLMTraceModel(Model):
    """LLM prompt/response trace for a turn."""

    id = fields.UUIDField(pk=True, default=uuid4)
    run = fields.ForeignKeyField("models.RunModel", related_name="llm_traces", on_delete=fields.CASCADE)
    turn = fields.IntField()

    system_prompt = fields.TextField()
    user_prompt = fields.TextField()
    response_raw = fields.JSONField(null=True)  # Structured response as dict
    model = fields.CharField(max_length=100)
    temperature = fields.FloatField()
    max_tokens = fields.IntField()
    latency_ms = fields.IntField(null=True)
    error = fields.TextField(null=True)

    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "alignsim_llm_traces"
        unique_together = [("run", "turn")]
        indexes = [("run", "turn")]

    def __str__(self) -> str:
        return f"LLMTrace(turn={self.turn}, model={self.model})"


class CustomerSnapshotModel(Model):
    """Per-customer state at a given turn."""

    id = fields.UUIDField(pk=True, default=uuid4)
    run = fields.ForeignKeyField("models.RunModel", related_name="customer_snapshots", on_delete=fields.CASCADE)
    turn = fields.IntField()

    customer_id = fields.CharField(max_length=50)
    stage = fields.CharField(max_length=20)
    health = fields.FloatField()
    deal_value = fields.IntField()
    engagement = fields.CharField(max_length=20)
    competitive_pressure = fields.FloatField(default=0.0)
    is_customer = fields.BooleanField(default=False)

    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "alignsim_customer_snapshots"
        indexes = [("run", "turn"), ("run", "customer_id"), ("run", "turn", "customer_id")]

    def __str__(self) -> str:
        return f"CustomerSnapshot(turn={self.turn}, {self.customer_id})"
