import asyncio
import pandas as pd
from typing import List, Dict, Set, Tuple
from datetime import timezone

from .knowledge_graph import KnowledgeGraph
from .utils.source_data import get_app_decisions_as_df
from .ontology.ontology import (
    Edge,
    StructuredNode,
    DecisionNode,
    OptionNode,
    CriteriaNode,
    GoalNode,
    UserNode,
    InsightNode,
)
from .config.prompts import Neo4jNodeCategory
from .utils.neo4j_graph_functions import update_neo4j_graph


def build_decision_graph() -> KnowledgeGraph:
    print("Getting source decision data from BigQuery...")
    decisions_df = get_app_decisions_as_df()
    print(f"Building decision graph with {len(decisions_df)} decisions...")

    nodes: List[StructuredNode] = []
    edges: List[Edge] = []

    user_nodes_dict: Dict[str, StructuredNode] = {}
    edge_set: Set[Tuple[str, str, str]] = set()

    for index, row in decisions_df.iterrows():
        decision = row["decision_data"]
        print(f"Working on decision number {index+1} of {len(decisions_df)}...")
        # Create decision node
        structured_decision_node = StructuredNode(
            node_category=Neo4jNodeCategory.BUSINESS_DECISIONS,
            decision_node=DecisionNode(
                name=decision["decision_title"],
                category=Neo4jNodeCategory.BUSINESS_DECISIONS.value,
                decision_id=decision["decision_id"],
                description=decision["summary"] or decision["goals"],
                created_at=decision["created_at"].replace(tzinfo=timezone.utc).isoformat(),
                updated_at=decision["updated_at"].replace(tzinfo=timezone.utc).isoformat(),
                other_fields={},
                status="decided" if decision["is_decided"] else "undecided",
                impact="",
                decision_type="",
                reversibility="",
            ),
        )
        nodes.append(structured_decision_node)

        # Create criteria nodes first
        criteria_nodes: Dict[str, StructuredNode] = {}
        for criterion in decision["criteria"]:
            if criterion["criterion_id"] is None:
                continue
            structured_criterion_node = StructuredNode(
                node_category=Neo4jNodeCategory.CRITERIA,
                criteria_node=CriteriaNode(
                    name=criterion["criterion_title"],
                    category=Neo4jNodeCategory.CRITERIA.value,
                    description=criterion["criterion_description"],
                    created_at=decision["created_at"].replace(tzinfo=timezone.utc).isoformat(),
                    updated_at=decision["updated_at"].replace(tzinfo=timezone.utc).isoformat(),
                    other_fields={"criterion_id": criterion["criterion_id"]},
                ),
            )
            criteria_nodes[criterion["criterion_id"]] = structured_criterion_node
            nodes.append(structured_criterion_node)

        # Create options nodes and link them to criteria
        for option in decision["options"].values():
            if option == {}:
                continue
            structured_option_node = StructuredNode(
                node_category=Neo4jNodeCategory.OPTIONS,
                options_node=OptionNode(
                    name=option["option_title"],
                    category=Neo4jNodeCategory.OPTIONS.value,
                    description=option["option_description"],
                    created_at=decision["created_at"].replace(tzinfo=timezone.utc).isoformat(),
                    updated_at=decision["updated_at"].replace(tzinfo=timezone.utc).isoformat(),
                    other_fields={"option_id": option["option_id"]},
                ),
            )
            nodes.append(structured_option_node)
            edges.append(
                Edge(
                    source=structured_decision_node.decision_node.name,
                    target=structured_option_node.options_node.name,
                    name="HAS_OPTION",
                    source_node_id=structured_decision_node.decision_node.node_id,
                    target_node_id=structured_option_node.options_node.node_id,
                    other_fields={},
                )
            )

            for criteria_eval in option["criteria_evaluations"]:
                if criteria_eval["criterion_id"] is None:
                    continue
                structured_criterion_node = criteria_nodes[criteria_eval["criterion_id"]]
                # Check if the criterion has been evaluated
                if pd.notna(criteria_eval["rating"]):
                    edge_name = f"EVALUATED_AS_{criteria_eval['rating'].upper()}"
                    other_fields = {
                        "evaluation_id": criteria_eval["evaluation_id"],
                    }
                else:
                    edge_name = "NOT_EVALUATED"
                    other_fields = {}

                # Check if this edge already exists
                edge_key = (
                    structured_option_node.options_node.node_id,
                    structured_criterion_node.criteria_node.node_id,
                    edge_name,
                )
                if edge_key not in edge_set:
                    edges.append(
                        Edge(
                            source=structured_option_node.options_node.name,
                            target=structured_criterion_node.criteria_node.name,
                            name=edge_name,
                            source_node_id=structured_option_node.options_node.node_id,
                            target_node_id=structured_criterion_node.criteria_node.node_id,
                            other_fields=other_fields,
                        )
                    )
                edge_set.add(edge_key)

        # Create goal node
        if decision["goals"]:
            structured_goals_node = StructuredNode(
                node_category=Neo4jNodeCategory.GOALS,
                goals_node=GoalNode(
                    name=f"Goal for {decision['decision_title']}",
                    category=Neo4jNodeCategory.GOALS.value,
                    description=decision["goals"],
                    created_at=decision["created_at"].replace(tzinfo=timezone.utc).isoformat(),
                    updated_at=decision["updated_at"].replace(tzinfo=timezone.utc).isoformat(),
                    goal_type="",
                ),
            )
            nodes.append(structured_goals_node)
            edges.append(
                Edge(
                    source=structured_decision_node.decision_node.name,
                    target=structured_goals_node.goals_node.name,
                    name="HAS_GOAL",
                    source_node_id=structured_decision_node.decision_node.node_id,
                    target_node_id=structured_goals_node.goals_node.node_id,
                    other_fields={},
                )
            )

        # Create or get user nodes for creator and decider
        for role, name in [("creator", decision["creator_name"]), ("decider", decision["decider_name"])]:
            if name not in user_nodes_dict:
                structured_users_node = StructuredNode(
                    node_category=Neo4jNodeCategory.USERS,
                    users_node=UserNode(
                        name=name,
                        category=Neo4jNodeCategory.USERS.value,
                        description=f"{name} - {role.capitalize()} for decision: {decision['decision_title']}",
                        created_at=decision["created_at"].replace(tzinfo=timezone.utc).isoformat(),
                        updated_at=decision["updated_at"].replace(tzinfo=timezone.utc).isoformat(),
                        title="",
                        location="",
                        department="",
                    ),
                )
                nodes.append(structured_users_node)
                user_nodes_dict[name] = structured_users_node
            else:
                structured_users_node = user_nodes_dict[name]

            edges.append(
                Edge(
                    source=structured_decision_node.decision_node.name,
                    target=structured_users_node.users_node.name,
                    name=f"HAS_{role.upper()}",
                    source_node_id=structured_decision_node.decision_node.node_id,
                    target_node_id=structured_users_node.users_node.node_id,
                    other_fields={},
                )
            )

        # Create or get user nodes for collaborators
        for collaborator_name in decision["collaborators"]:
            if collaborator_name not in user_nodes_dict:
                structured_collaborator_node = StructuredNode(
                    node_category=Neo4jNodeCategory.USERS,
                    users_node=UserNode(
                        name=collaborator_name,
                        category=Neo4jNodeCategory.USERS.value,
                        description=f"{collaborator_name} - Collaborator for decision: {decision['decision_title']}",
                        created_at=decision["created_at"].replace(tzinfo=timezone.utc).isoformat(),
                        updated_at=decision["updated_at"].replace(tzinfo=timezone.utc).isoformat(),
                        title="",
                        location="",
                        department="",
                    ),
                )
                nodes.append(structured_collaborator_node)
                user_nodes_dict[collaborator_name] = structured_collaborator_node
            else:
                structured_collaborator_node = user_nodes_dict[collaborator_name]

            edges.append(
                Edge(
                    source=structured_decision_node.decision_node.name,
                    target=structured_collaborator_node.users_node.name,
                    name="HAS_COLLABORATOR",
                    source_node_id=structured_decision_node.decision_node.node_id,
                    target_node_id=structured_collaborator_node.users_node.node_id,
                    other_fields={},
                )
            )

        # Create insight nodes
        # TODO: Make citations an edge to a new or existing source (node) they are citing
        for insight in decision["insights"].values():
            if not insight:
                continue
            structured_insight_node = StructuredNode(
                node_category=Neo4jNodeCategory.INSIGHTS,
                insights_node=InsightNode(
                    name=insight["insight_title"],
                    category=Neo4jNodeCategory.INSIGHTS.value,
                    description=insight["insight_description"] or "",
                    created_at=insight["insight_created_at"].replace(tzinfo=timezone.utc).isoformat(),
                    updated_at=insight["insight_updated_at"].replace(tzinfo=timezone.utc).isoformat(),
                    other_fields={
                        "insight_id": insight["insight_id"],
                        "citations": insight["citations"],
                        "position": insight["position"],
                        "source": insight["insight_source"],
                        "subtitle": insight["subtitle"] or "",
                    },
                ),
            )
            nodes.append(structured_insight_node)

            # Relate insight to decision
            edges.append(
                Edge(
                    source=structured_decision_node.decision_node.name,
                    target=structured_insight_node.insights_node.name,
                    name="HAS_INSIGHT",
                    source_node_id=structured_decision_node.decision_node.node_id,
                    target_node_id=structured_insight_node.insights_node.node_id,
                    other_fields={},
                )
            )

            # Create or get user nodes and relate to insight
            for role, name in [
                ("CREATED", insight["creator_name"]),
                ("ASSIGNED_TO", insight["assignee_name"]),
                ("COMPLETED", insight["completor_name"]),
            ]:
                if not name:
                    continue
                if name not in user_nodes_dict:
                    structured_users_node = StructuredNode(
                        node_category=Neo4jNodeCategory.USERS,
                        users_node=UserNode(
                            name=name,
                            category=Neo4jNodeCategory.USERS.value,
                            description=f"{name} - {role} for insight: {insight['insight_title']}",
                            created_at=insight["insight_created_at"].replace(tzinfo=timezone.utc).isoformat(),
                            updated_at=insight["insight_updated_at"].replace(tzinfo=timezone.utc).isoformat(),
                            title="",
                            location="",
                            department="",
                        ),
                    )
                    nodes.append(structured_users_node)
                    user_nodes_dict[name] = structured_users_node
                else:
                    structured_users_node = user_nodes_dict[name]

                edges.append(
                    Edge(
                        source=structured_users_node.users_node.name,
                        target=structured_insight_node.insights_node.name,
                        name=role,
                        source_node_id=structured_users_node.users_node.node_id,
                        target_node_id=structured_insight_node.insights_node.node_id,
                        other_fields={},
                    )
                )

    # Create KnowledgeGraph object
    knowledge_graph = KnowledgeGraph(structured_nodes=nodes, edges=edges)

    knowledge_graph.convert_to_generic_nodes()
    asyncio.run(knowledge_graph.aaugment_all_nodes_with_embedding())

    print(f"Updating neo4j with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} edges...")

    # Update Neo4j graph
    update_neo4j_graph(knowledge_graph, update_with_merge=False)

    return knowledge_graph
