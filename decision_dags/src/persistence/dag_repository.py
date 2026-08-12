"""Repository for DAG persistence operations."""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from tortoise.exceptions import DoesNotExist
from tortoise.transactions import in_transaction

from ..models import DecisionDAG, DecisionNode, DecisionEdge, NodeType, EdgeType, DecisionType, DecisionReasoningType
from .models import DAGModel, NodeModel, EdgeModel

logger = logging.getLogger(__name__)


def serialize_for_json(obj: Any) -> Any:
    """Recursively serialize objects for JSON storage, handling datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj


class DAGRepository:
    """Repository for saving and loading Decision DAGs."""

    async def save_dag(
        self,
        dag: DecisionDAG,
        problem_statement: str,
        generation_method: str = "build",
        parent_dag_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """
        Save a Decision DAG to the database.

        Args:
            dag: The DecisionDAG to save
            problem_statement: The problem statement for this DAG
            generation_method: How this DAG was generated ('build', 'extracted', 'evolved')
            parent_dag_id: ID of parent DAG if this is derived
            metadata: Additional metadata to store

        Returns:
            UUID of the saved DAG
        """
        async with in_transaction() as _transaction:
            try:
                # Create DAG record
                dag_model = await DAGModel.create(
                    problem_statement=problem_statement,
                    generation_method=generation_method,
                    parent_dag_id=parent_dag_id,
                    max_layers=dag.get_max_layer(),
                    node_count=len(dag.all_nodes),
                    edge_count=len(dag.edges),
                    metadata=serialize_for_json(
                        {
                            **(metadata or {}),
                            "original_dag_id": dag.id,
                            "created_at": dag.created_at.isoformat() if dag.created_at else datetime.now().isoformat(),
                            "dag_metadata": dag.metadata,
                            "original_extracted_path_id": dag.original_extracted_path_id,
                        }
                    ),
                )

                # Save all nodes
                for node_id, node in dag.all_nodes.items():
                    await NodeModel.create(
                        dag=dag_model,
                        node_id=node.id,
                        layer=node.layer,
                        type=node.type.value,
                        title=node.title,
                        description=node.description,
                        decision_type=node.decision_type.value if node.decision_type else None,
                        goal_impacts=serialize_for_json(node.goal_impacts),
                        people_impacted=serialize_for_json(node.people_impacted),
                        resource_requirements=serialize_for_json(node.resource_requirements),
                        tags=serialize_for_json(node.tags),
                        metadata=serialize_for_json(node.metadata),
                        embedding=node.embedding if node.embedding else None,
                        confidence_score=node.confidence_score,
                    )

                # Save all edges
                for edge in dag.edges:
                    await EdgeModel.create(
                        dag=dag_model,
                        source_node_id=edge.source_id,
                        target_node_id=edge.target_id,
                        edge_type=edge.edge_type.value,
                        condition=edge.condition,
                        decision_reasoning_type=edge.decision_reasoning_type.value
                        if edge.decision_reasoning_type
                        else None,
                        likelihood=edge.likelihood,
                        label=edge.label,
                        cost_estimate=edge.cost_estimate,
                        timeline_estimate=edge.timeline_estimate,
                        estimated_cost_dollars=edge.estimated_cost_dollars,
                        implementation_risks=serialize_for_json(edge.implementation_risks),
                        conditions=serialize_for_json(edge.conditions),
                        metadata=serialize_for_json(edge.metadata),
                        relationship=edge.relationship,
                    )

                logger.info(f"Saved DAG {dag_model.id} with {len(dag.all_nodes)} nodes and {len(dag.edges)} edges")
                return dag_model.id

            except Exception as e:
                logger.error(f"Failed to save DAG: {e}")
                raise

    async def load_dag(self, dag_id: UUID) -> DecisionDAG:
        """
        Load a Decision DAG from the database.

        Args:
            dag_id: UUID of the DAG to load

        Returns:
            Reconstructed DecisionDAG
        """
        try:
            # Load DAG with related nodes and edges
            dag_model = await DAGModel.get(id=dag_id).prefetch_related("nodes", "edges")

            # Create DAG instance
            dag = DecisionDAG(
                id=str(dag_model.metadata.get("original_dag_id", dag_model.id)),
                metadata=dag_model.metadata.get("dag_metadata", {}),
                created_at=datetime.fromisoformat(
                    dag_model.metadata.get("created_at", dag_model.created_at.isoformat())
                ),
                original_extracted_path_id=dag_model.metadata.get("original_extracted_path_id"),
            )

            # Load all nodes
            nodes_map = {}
            for node_model in dag_model.nodes:
                # Handle decision_type properly
                node_type = NodeType(node_model.type)
                decision_type = None
                if node_model.decision_type:
                    decision_type = DecisionType(node_model.decision_type)
                elif node_type == NodeType.DECISION:
                    # Default decision_type for decision nodes if missing from DB
                    decision_type = DecisionType.STRATEGIC

                node = DecisionNode(
                    id=node_model.node_id,
                    layer=node_model.layer,
                    type=node_type,
                    title=node_model.title,
                    description=node_model.description,
                    decision_type=decision_type,
                    goal_impacts=node_model.goal_impacts,
                    people_impacted=node_model.people_impacted,
                    resource_requirements=node_model.resource_requirements,
                    tags=node_model.tags,
                    metadata=node_model.metadata,
                    embedding=node_model.embedding,
                    confidence_score=node_model.confidence_score,
                )
                nodes_map[node.id] = node
                dag.add_node(node)

            # Load all edges
            for edge_model in dag_model.edges:
                # Handle decision_reasoning_type properly
                edge_type = EdgeType(edge_model.edge_type)
                decision_reasoning_type = None
                if edge_model.decision_reasoning_type:
                    decision_reasoning_type = DecisionReasoningType(edge_model.decision_reasoning_type)
                elif edge_type == EdgeType.OPTION_TO_DECISION:
                    # Default for option_to_decision edges if missing from DB
                    decision_reasoning_type = DecisionReasoningType.LOGICAL

                edge = DecisionEdge(
                    source_id=edge_model.source_node_id,
                    target_id=edge_model.target_node_id,
                    edge_type=edge_type,
                    condition=edge_model.condition,
                    decision_reasoning_type=decision_reasoning_type,
                    likelihood=edge_model.likelihood,
                    label=edge_model.label,
                    cost_estimate=edge_model.cost_estimate,
                    timeline_estimate=edge_model.timeline_estimate,
                    estimated_cost_dollars=edge_model.estimated_cost_dollars,
                    implementation_risks=edge_model.implementation_risks,
                    conditions=edge_model.conditions,
                    metadata=edge_model.metadata,
                    relationship=edge_model.relationship,
                )
                dag.add_edge(edge)

            logger.info(f"Loaded DAG {dag_id} with {len(dag.all_nodes)} nodes and {len(dag.edges)} edges")
            return dag

        except DoesNotExist:
            logger.error(f"DAG {dag_id} not found")
            raise ValueError(f"DAG with ID {dag_id} not found")
        except Exception as e:
            logger.error(f"Failed to load DAG: {e}")
            raise

    async def list_dags(
        self,
        filter_by: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: str = "created_at",
        ascending: bool = False,
    ) -> List[DAGModel]:
        """
        List DAGs with optional filtering.

        Args:
            filter_by: Filter by generation_method ('build', 'extracted', 'evolved')
            limit: Maximum number of results
            offset: Number of results to skip
            sort_by: Field to sort by
            ascending: Sort order

        Returns:
            List of DAG models
        """
        query = DAGModel.all()

        if filter_by:
            query = query.filter(generation_method=filter_by)

        # Apply sorting
        order_field = f"{'' if ascending else '-'}{sort_by}"
        query = query.order_by(order_field)

        # Apply pagination
        query = query.offset(offset).limit(limit)

        return await query

    async def update_dag_metadata(
        self, dag_id: UUID, generation_method: Optional[str] = None, metadata_update: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update DAG metadata.

        Args:
            dag_id: UUID of the DAG to update
            generation_method: New generation method
            metadata_update: Dictionary of metadata to update
        """
        try:
            dag_model = await DAGModel.get(id=dag_id)

            if generation_method:
                dag_model.generation_method = generation_method

            if metadata_update:
                dag_model.metadata.update(metadata_update)

            dag_model.updated_at = datetime.now()
            await dag_model.save()

            logger.info(f"Updated metadata for DAG {dag_id}")

        except DoesNotExist:
            logger.error(f"DAG {dag_id} not found")
            raise ValueError(f"DAG with ID {dag_id} not found")

    async def delete_dag(self, dag_id: UUID, cascade: bool = True) -> None:
        """
        Delete a DAG and optionally its children.

        Args:
            dag_id: UUID of the DAG to delete
            cascade: Whether to delete child DAGs
        """
        try:
            dag_model = await DAGModel.get(id=dag_id)

            if cascade:
                # Delete will cascade to nodes and edges automatically
                await dag_model.delete()
                logger.info(f"Deleted DAG {dag_id} and all related data")
            else:
                # Check if there are children
                children_count = await DAGModel.filter(parent_dag_id=dag_id).count()
                if children_count > 0:
                    raise ValueError(f"DAG {dag_id} has {children_count} child DAGs. Use cascade=True to delete them.")
                await dag_model.delete()
                logger.info(f"Deleted DAG {dag_id}")

        except DoesNotExist:
            logger.error(f"DAG {dag_id} not found")
            raise ValueError(f"DAG with ID {dag_id} not found")

    async def get_dag_info(self, dag_id: UUID) -> Dict[str, Any]:
        """
        Get detailed information about a DAG.

        Args:
            dag_id: UUID of the DAG

        Returns:
            Dictionary with DAG information
        """
        try:
            dag_model = await DAGModel.get(id=dag_id)

            # Get counts
            node_count = await NodeModel.filter(dag_id=dag_id).count()
            edge_count = await EdgeModel.filter(dag_id=dag_id).count()

            # Get parent info
            parent_info = None
            if dag_model.parent_dag_id:
                parent = await DAGModel.get_or_none(id=dag_model.parent_dag_id)
                if parent:
                    parent_info = {
                        "id": str(parent.id),
                        "problem_statement": parent.problem_statement,
                        "generation_method": parent.generation_method,
                    }

            # Get children info
            children = await DAGModel.filter(parent_dag_id=dag_id).all()
            children_info = [
                {
                    "id": str(child.id),
                    "problem_statement": child.problem_statement,
                    "generation_method": child.generation_method,
                    "created_at": child.created_at.isoformat(),
                }
                for child in children
            ]

            return {
                "id": str(dag_model.id),
                "problem_statement": dag_model.problem_statement,
                "generation_method": dag_model.generation_method,
                "max_layers": dag_model.max_layers,
                "node_count": node_count,
                "edge_count": edge_count,
                "parent": parent_info,
                "children": children_info,
                "metadata": dag_model.metadata,
                "created_at": dag_model.created_at.isoformat(),
                "updated_at": dag_model.updated_at.isoformat(),
            }

        except DoesNotExist:
            logger.error(f"DAG {dag_id} not found")
            raise ValueError(f"DAG with ID {dag_id} not found")


# Create singleton repository instance
dag_repository = DAGRepository()
