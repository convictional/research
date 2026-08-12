"""Tortoise ORM models for Decision DAG persistence."""

from uuid import uuid4

from tortoise import fields
from tortoise.models import Model


class DAGModel(Model):
    """Model for storing Decision DAG metadata."""

    id = fields.UUIDField(pk=True, default=uuid4)
    problem_statement = fields.TextField()
    generation_method = fields.CharField(max_length=50)  # 'build', 'extracted', 'evolved'
    parent_dag = fields.ForeignKeyField(
        "models.DAGModel", null=True, related_name="children", on_delete=fields.CASCADE
    )

    # Metrics
    max_layers = fields.IntField()
    node_count = fields.IntField()
    edge_count = fields.IntField()

    # Additional metadata
    metadata = fields.JSONField(default=dict)

    # Timestamps
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "decision_dags"
        ordering = ["-created_at"]

    def __str__(self):
        return f"DAG({self.id}): {self.problem_statement[:50]}..."


class NodeModel(Model):
    """Model for storing Decision DAG nodes."""

    id = fields.UUIDField(pk=True, default=uuid4)
    dag = fields.ForeignKeyField("models.DAGModel", related_name="nodes", on_delete=fields.CASCADE)
    node_id = fields.CharField(max_length=100)  # Original node ID from DAG

    # Node properties
    layer = fields.IntField()
    type = fields.CharField(max_length=20)  # 'decision' or 'option'
    title = fields.TextField()
    description = fields.TextField()

    # Optional fields
    decision_type = fields.CharField(max_length=50, null=True)
    goal_impacts = fields.JSONField(default=dict)
    people_impacted = fields.JSONField(default=list)
    resource_requirements = fields.JSONField(default=dict)

    # Arrays and metadata
    tags = fields.JSONField(default=list)
    metadata = fields.JSONField(default=dict)
    confidence_score = fields.FloatField(null=True)

    # Embedding - stored as JSON array since Tortoise doesn't have native vector support
    embedding = fields.JSONField(null=True)

    # Timestamp
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "decision_nodes"
        unique_together = [("dag", "node_id")]
        indexes = [("dag", "layer"), ("dag", "type")]

    def __str__(self):
        return f"Node({self.node_id}): {self.title[:50]}..."


class EdgeModel(Model):
    """Model for storing Decision DAG edges."""

    id = fields.UUIDField(pk=True, default=uuid4)
    dag = fields.ForeignKeyField("models.DAGModel", related_name="edges", on_delete=fields.CASCADE)

    # Edge endpoints
    source_node_id = fields.CharField(max_length=100)
    target_node_id = fields.CharField(max_length=100)

    # Edge properties
    edge_type = fields.CharField(max_length=50)
    condition = fields.TextField()
    decision_reasoning_type = fields.CharField(max_length=50, null=True)
    likelihood = fields.CharField(max_length=20, default="medium")
    label = fields.TextField(default="")

    # Cost and timeline
    cost_estimate = fields.CharField(max_length=50, null=True)
    timeline_estimate = fields.CharField(max_length=100, null=True)
    estimated_cost_dollars = fields.FloatField(null=True)

    # Arrays and metadata
    implementation_risks = fields.JSONField(null=True)
    conditions = fields.JSONField(default=list)
    metadata = fields.JSONField(default=dict)

    # Legacy compatibility
    relationship = fields.CharField(max_length=50, default="leads_to")

    # Timestamp
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "decision_edges"
        unique_together = [("dag", "source_node_id", "target_node_id")]
        indexes = [("dag", "source_node_id"), ("dag", "target_node_id")]

    def __str__(self):
        return f"Edge({self.source_node_id} -> {self.target_node_id})"
