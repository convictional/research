"""Database context integration for decision DAGs."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import asyncpg
from openai import AsyncOpenAI

from common.embeddings import aembed_query
from ..settings import settings

logger = logging.getLogger(__name__)


class DatabaseContextProvider:
    """Provides organizational context from the decide_development database."""

    def __init__(self):
        self._connection_pool: Optional[asyncpg.Pool] = None
        self._openai_client: Optional[AsyncOpenAI] = None

    async def _get_connection_pool(self) -> asyncpg.Pool:
        """Get or create database connection pool."""
        if not self._connection_pool:
            # Use the same connection string pattern as agentic_sql
            connection_string = (
                f"postgresql://{settings.local_postgres_user}:{settings.local_postgres_password}"
                f"@{settings.local_postgres_host}:{settings.local_postgres_port}/decide_development"
            )
            self._connection_pool = await asyncpg.create_pool(
                connection_string, min_size=1, max_size=5, command_timeout=30
            )
        return self._connection_pool

    def _get_openai_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if not self._openai_client:
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        return self._openai_client

    async def get_organizational_context(
        self, problem_statement: str, organization_id: Optional[str] = None, top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Get comprehensive organizational context for DAG building.

        Args:
            problem_statement: The strategic problem to analyze
            organization_id: Optional organization filter
            top_k: Number of results per category

        Returns:
            Dict containing organizational goals, past decisions, content, and activity insights
        """
        try:
            # Generate embedding for similarity search
            embedding = await self._generate_embedding(problem_statement)

            # Fetch context in parallel
            context_tasks = [
                self._get_organizational_goals(organization_id, top_k),
                self._get_relevant_decisions(problem_statement, embedding, organization_id, top_k),
                self._get_relevant_content(problem_statement, embedding, organization_id, top_k),
                self._get_activity_insights(problem_statement, organization_id, top_k),
            ]

            goals, decisions, content, activities = await asyncio.gather(*context_tasks)

            return {
                "organizational_goals": goals,
                "past_decisions": decisions,
                "relevant_content": content,
                "activity_insights": activities,
                "context_summary": self._create_context_summary(goals, decisions, content, activities),
            }

        except Exception as e:
            logger.error(f"Error fetching organizational context: {e}")
            return {
                "organizational_goals": [],
                "past_decisions": [],
                "relevant_content": [],
                "activity_insights": {},
                "context_summary": "Unable to fetch organizational context",
            }

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for similarity search."""
        try:
            embedding = await aembed_query(
                async_openai_client=self._get_openai_client(),
                text=text,
                embedding_model=settings.embedding_model,
                embedding_dim=1536,
            )
            return embedding.tolist() if hasattr(embedding, "tolist") else embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    async def _get_organizational_goals(
        self, organization_id: Optional[str] = None, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get organizational goals, prioritizing active/incomplete ones."""
        try:
            pool = await self._get_connection_pool()

            # Build query with optional organization filter
            # Note: goal table doesn't have deleted_at column based on schema
            where_clause = "WHERE 1=1"
            params = []
            if organization_id:
                where_clause += " AND organization_id = $1"
                params.append(organization_id)

            query = f"""
                SELECT
                    id, title, status, target_date, current_metric_value,
                    created_at, updated_at, completed_at
                FROM goal
                {where_clause}
                ORDER BY
                    CASE WHEN status = 'completed' THEN 1 ELSE 0 END,
                    created_at DESC
                LIMIT {top_k * 2}
            """

            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

                goals = []
                for row in rows:
                    # Get success conditions if they exist
                    # Note: successcondition table doesn't have deleted_at column based on schema
                    conditions_query = """
                        SELECT description FROM successcondition
                        WHERE goal_id = $1
                    """
                    conditions = await conn.fetch(conditions_query, row["id"])

                    goals.append(
                        {
                            "id": str(row["id"]),
                            "title": row["title"],
                            "status": row["status"],
                            "target_date": row["target_date"].isoformat() if row["target_date"] else None,
                            "current_metric_value": row["current_metric_value"],
                            "success_conditions": [{"description": c["description"]} for c in conditions],
                            "is_active": row["status"] != "completed",
                            "created_at": row["created_at"].isoformat(),
                        }
                    )

                logger.info(f"Retrieved {len(goals)} organizational goals")
                return goals[:top_k]

        except Exception as e:
            logger.error(f"Error fetching organizational goals: {e}")
            return []

    async def _get_relevant_decisions(
        self,
        problem_statement: str,
        embedding: List[float],
        organization_id: Optional[str] = None,
        top_k: int = 20,  # Increased for better filtering
    ) -> List[Dict[str, Any]]:
        """Get past decisions relevant to the problem statement using similarity search."""
        try:
            pool = await self._get_connection_pool()

            # Extract keywords from problem statement for text search
            keywords = self._extract_keywords(problem_statement)
            search_terms = " | ".join(keywords[:5])  # Use top 5 keywords for search

            # Build query with optional organization filter
            where_clause = "WHERE deleted_at IS NULL"
            params = []
            if organization_id:
                where_clause += " AND organization_id = $1"
                params.append(organization_id)
                param_offset = 1
            else:
                param_offset = 0

            # Combine text similarity and recency scoring
            search_param = param_offset + 1
            tsquery_param = param_offset + 2
            query = f"""
                WITH decision_search AS (
                    SELECT
                        id, title, explanation, source, created_at, updated_at,
                        CASE
                            WHEN title ILIKE ${search_param} OR explanation ILIKE ${search_param}
                            THEN 2.0
                            WHEN to_tsvector('english', title || ' ' || COALESCE(explanation, ''))
                                 @@ plainto_tsquery('english', ${tsquery_param})
                            THEN ts_rank_cd(to_tsvector('english', title || ' ' || COALESCE(explanation, '')),
                                          plainto_tsquery('english', ${tsquery_param}))
                            ELSE 0.0
                        END as text_relevance_score,
                        EXTRACT(EPOCH FROM (NOW() - created_at)) / (365.25 * 24 * 3600) as years_old,
                        LENGTH(COALESCE(explanation, '')) as explanation_length
                    FROM decision
                    {where_clause}
                ),
                scored_decisions AS (
                    SELECT *,
                        -- Combine relevance, recency, and content quality
                        (text_relevance_score * 0.5) +
                        (GREATEST(0, 1.0 - years_old/2.0) * 0.3) +  -- Decay over 2 years
                        (LEAST(1.0, explanation_length/200.0) * 0.2) as combined_score  -- Quality bonus
                    FROM decision_search
                    WHERE text_relevance_score > 0.0  -- Only include relevant results
                )
                SELECT
                    id, title, explanation, source, created_at, updated_at,
                    text_relevance_score, combined_score
                FROM scored_decisions
                ORDER BY combined_score DESC
                LIMIT {top_k}
            """

            # Prepare search parameters
            search_term = f"%{problem_statement.split()[0] if problem_statement.split() else problem_statement}%"
            search_params = params + [search_term, search_terms]

            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *search_params)

                decisions = []
                for row in rows:
                    decisions.append(
                        {
                            "id": str(row["id"]),
                            "title": row["title"],
                            "explanation": row["explanation"],
                            "source": row["source"],
                            "created_at": row["created_at"].isoformat(),
                            "relevance_score": float(row["text_relevance_score"]),
                            "combined_score": float(row["combined_score"]),
                            "type": "decision",
                        }
                    )

                logger.info(f"Retrieved {len(decisions)} relevant past decisions with scores")
                return decisions

        except Exception as e:
            logger.error(f"Error fetching past decisions: {e}")
            return []

    async def _get_relevant_content(
        self,
        problem_statement: str,
        embedding: List[float],
        organization_id: Optional[str] = None,
        top_k: int = 15,  # Increased for better filtering
    ) -> List[Dict[str, Any]]:
        """Get content relevant to the problem statement using sophisticated search."""
        try:
            pool = await self._get_connection_pool()

            # Extract keywords from problem statement for text search
            keywords = self._extract_keywords(problem_statement)
            search_terms = " | ".join(keywords[:5])  # Use top 5 keywords for search

            # Build query with optional organization filter
            where_clause = "WHERE 1=1"
            params = []
            if organization_id:
                where_clause += " AND organization_id = $1"
                params.append(organization_id)
                param_offset = 1
            else:
                param_offset = 0

            # Sophisticated content search with scoring
            search_param = param_offset + 1
            tsquery_param = param_offset + 2
            query = f"""
                WITH content_search AS (
                    SELECT
                        id, title, content_type, category, author, preview_content,
                        index_content, source, created_at, metadata,
                        CASE
                            WHEN title ILIKE ${search_param}
                            THEN 3.0
                            WHEN preview_content ILIKE ${search_param} OR index_content ILIKE ${search_param}
                            THEN 2.0
                            WHEN to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(index_content, ''))
                                 @@ plainto_tsquery('english', ${tsquery_param})
                            THEN ts_rank_cd(to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(index_content, '')),
                                          plainto_tsquery('english', ${tsquery_param}))
                            ELSE 0.0
                        END as text_relevance_score,
                        EXTRACT(EPOCH FROM (NOW() - created_at)) / (365.25 * 24 * 3600) as years_old,
                        CASE content_type
                            WHEN 'decision_process' THEN 1.0
                            WHEN 'goal' THEN 0.9
                            WHEN 'meeting_transcript' THEN 0.8
                            WHEN 'discussion_comment' THEN 0.7
                            ELSE 0.6
                        END as content_type_weight,
                        LENGTH(COALESCE(index_content, preview_content, '')) as content_length
                    FROM content
                    {where_clause}
                ),
                scored_content AS (
                    SELECT *,
                        -- Combine relevance, recency, content type importance, and quality
                        (text_relevance_score * 0.4) +
                        (content_type_weight * 0.25) +
                        (GREATEST(0, 1.0 - years_old/3.0) * 0.25) +  -- Decay over 3 years
                        (LEAST(1.0, content_length/500.0) * 0.1) as combined_score  -- Quality bonus
                    FROM content_search
                    WHERE text_relevance_score > 0.0  -- Only include relevant results
                )
                SELECT
                    id, title, content_type, category, author, preview_content,
                    index_content, source, created_at, metadata,
                    text_relevance_score, combined_score
                FROM scored_content
                ORDER BY combined_score DESC
                LIMIT {top_k}
            """

            # Prepare search parameters
            search_term = f"%{problem_statement.split()[0] if problem_statement.split() else problem_statement}%"
            search_params = params + [search_term, search_terms]

            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *search_params)

                content = []
                for row in rows:
                    # Use preview_content if available, otherwise truncate index_content
                    preview = row["preview_content"] or (
                        row["index_content"][:1000] + "..."
                        if row["index_content"] and len(row["index_content"]) > 1000
                        else row["index_content"] or "No content available"
                    )

                    content.append(
                        {
                            "id": str(row["id"]),
                            "title": row["title"],
                            "content_type": row["content_type"],
                            "category": row["category"],
                            "author": row["author"],
                            "preview": preview,
                            "source": row["source"],
                            "created_at": row["created_at"].isoformat(),
                            "metadata": row["metadata"] or {},
                            "relevance_score": float(row["text_relevance_score"]),
                            "combined_score": float(row["combined_score"]),
                        }
                    )

                logger.info(f"Retrieved {len(content)} relevant content items with scores")
                return content

        except Exception as e:
            logger.error(f"Error fetching relevant content: {e}")
            return []

    async def _get_activity_insights(
        self, problem_statement: str, organization_id: Optional[str] = None, top_k: int = 10
    ) -> Dict[str, Any]:
        """Get activity-based insights for resource planning."""
        try:
            pool = await self._get_connection_pool()

            # Build query with optional organization filter
            where_clause = "WHERE 1=1"
            params = []
            if organization_id:
                where_clause += " AND organization_id = $1"
                params.append(organization_id)

            # Get recent activities by type
            activity_query = f"""
                SELECT
                    type, content_type, COUNT(*) as activity_count,
                    array_agg(DISTINCT actor_id) as actors,
                    array_agg(snippet ORDER BY created_at DESC) as recent_snippets
                FROM activity
                {where_clause}
                AND created_at > NOW() - INTERVAL '90 days'
                GROUP BY type, content_type
                ORDER BY activity_count DESC
                LIMIT {top_k}
            """

            async with pool.acquire() as conn:
                activity_rows = await conn.fetch(activity_query, *params)

                # Get user engagement patterns
                engagement_query = f"""
                    SELECT
                        actor_id, COUNT(*) as activity_count,
                        array_agg(DISTINCT type) as activity_types
                    FROM activity
                    {where_clause}
                    AND created_at > NOW() - INTERVAL '30 days'
                    AND actor_id IS NOT NULL
                    GROUP BY actor_id
                    ORDER BY activity_count DESC
                    LIMIT 10
                """

                engagement_rows = await conn.fetch(engagement_query, *params)

                # Process activity insights
                activity_patterns = []
                for row in activity_rows:
                    activity_patterns.append(
                        {
                            "type": row["type"],
                            "content_type": row["content_type"],
                            "activity_count": row["activity_count"],
                            "unique_actors": len(set(row["actors"])) if row["actors"] else 0,
                            "recent_snippets": [s for s in row["recent_snippets"][:3] if s],
                        }
                    )

                # Process engagement patterns
                engagement_patterns = []
                for row in engagement_rows:
                    engagement_patterns.append(
                        {
                            "actor_id": str(row["actor_id"]),
                            "activity_count": row["activity_count"],
                            "activity_types": list(row["activity_types"]) if row["activity_types"] else [],
                        }
                    )

                insights = {
                    "activity_patterns": activity_patterns,
                    "engagement_patterns": engagement_patterns,
                    "collaboration_indicators": self._analyze_collaboration_patterns(activity_patterns),
                    "resource_indicators": self._analyze_resource_patterns(activity_patterns, engagement_patterns),
                }

                logger.info(f"Generated activity insights with {len(activity_patterns)} patterns")
                return insights

        except Exception as e:
            logger.error(f"Error fetching activity insights: {e}")
            return {
                "activity_patterns": [],
                "engagement_patterns": [],
                "collaboration_indicators": {},
                "resource_indicators": {},
            }

    def _analyze_collaboration_patterns(self, activity_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze collaboration patterns from activity data."""
        # Count collaborative vs individual activities
        collaborative_types = {"discussion_comment", "github_discussion_comment", "meeting_transcript"}
        individual_types = {"email", "document", "goal"}

        collaborative_count = sum(
            p["activity_count"] for p in activity_patterns if p["content_type"] in collaborative_types
        )
        individual_count = sum(p["activity_count"] for p in activity_patterns if p["content_type"] in individual_types)

        total_activities = collaborative_count + individual_count
        collaboration_ratio = collaborative_count / total_activities if total_activities > 0 else 0

        return {
            "collaboration_ratio": collaboration_ratio,
            "collaborative_activities": collaborative_count,
            "individual_activities": individual_count,
            "most_collaborative_types": [
                p["content_type"] for p in activity_patterns if p["content_type"] in collaborative_types
            ][:3],
        }

    def _analyze_resource_patterns(
        self, activity_patterns: List[Dict[str, Any]], engagement_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze resource allocation patterns."""
        # Calculate engagement distribution
        total_actors = len(engagement_patterns)
        high_engagement = len([p for p in engagement_patterns if p["activity_count"] > 10])
        medium_engagement = len([p for p in engagement_patterns if 5 <= p["activity_count"] <= 10])
        low_engagement = total_actors - high_engagement - medium_engagement

        # Identify resource-intensive activity types
        resource_intensive = sorted(
            activity_patterns, key=lambda x: x["activity_count"] * x["unique_actors"], reverse=True
        )[:3]

        return {
            "total_active_users": total_actors,
            "engagement_distribution": {
                "high_engagement": high_engagement,
                "medium_engagement": medium_engagement,
                "low_engagement": low_engagement,
            },
            "resource_intensive_activities": [
                {
                    "type": p["content_type"],
                    "total_activity": p["activity_count"],
                    "unique_contributors": p["unique_actors"],
                }
                for p in resource_intensive
            ],
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text for search."""
        import re

        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "throughout",
            "despite",
            "towards",
            "upon",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "this",
            "that",
            "these",
            "those",
        }

        # Extract words, convert to lowercase, filter stop words and short words
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) >= 3]

        # Return unique keywords, preserving order
        seen = set()
        result = []
        for word in keywords:
            if word not in seen:
                seen.add(word)
                result.append(word)

        return result[:10]  # Return top 10 keywords

    def _create_context_summary(
        self,
        goals: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]],
        content: List[Dict[str, Any]],
        activities: Dict[str, Any],
    ) -> str:
        """Create a human-readable summary of the organizational context."""
        active_goals = len([g for g in goals if g.get("is_active", True)])
        recent_decisions = len(decisions)
        content_types = set(c.get("content_type", "unknown") for c in content)
        collaboration_ratio = activities.get("collaboration_indicators", {}).get("collaboration_ratio", 0)

        summary = f"""
        Organizational Context Summary:
        - {active_goals} active goals out of {len(goals)} total goals
        - {recent_decisions} recent strategic decisions available
        - {len(content)} relevant content items from {len(content_types)} different sources
        - Collaboration ratio: {collaboration_ratio:.1%} (indicates team collaboration level)
        - {activities.get("resource_indicators", {}).get("total_active_users", 0)} actively engaged team members
        """

        return summary.strip()

    async def close(self):
        """Close database connections."""
        if self._connection_pool:
            await self._connection_pool.close()
