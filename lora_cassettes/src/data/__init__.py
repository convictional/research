"""
Data layer for LoRA Cassettes experiment.

Provides Pydantic models and database access helpers using the hybrid approach:
- Raw SQL + asyncpg for bulk operations and flexibility
- Pydantic models for type safety in Python
- Helper functions wrapping common query patterns
"""
