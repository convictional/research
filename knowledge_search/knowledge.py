import asyncio
import getpass
import json
import os
import time
from datetime import datetime

import asyncpg
from asyncpg.exceptions import DuplicateObjectError
from config import settings
from google.cloud import storage  # type: ignore
from infra.db import init_db
from infra.llm import instructor_client
from infra.vectors import embed
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from tortoise import Tortoise

TABLE_NAME = "knowledgev3"
MODEL = "gpt-4o"
RANKING_WEIGHTS = {
    "keywords": 0.65,
    "named_entities": 0.1,
    "sparse_vector": 0.05,
    "dense_vector": 0.1,
    "page_rank": 0.1,
}

SCORE_THRESHOLD = 0.0

STORAGE_BUCKET = "decide_storage_staging"
STORAGE_BLOB = "knowledgev3.dump"
LOCAL_FILE = "/tmp/knowledge.dump"


#
# Setup
#
#


async def setup_db():
    #
    # Enable pgvector extension
    #
    #

    try:
        admin_user = os.getenv("ADMIN_POSTGRES_USER", getpass.getuser())
        admin_password = os.getenv("ADMIN_POSTGRES_PASSWORD", "")
        admin_connection = await asyncpg.connect(
            database=settings.postgres_dict["database"], user=admin_user, password=admin_password
        )
        await admin_connection.execute("CREATE EXTENSION vector;")
        await admin_connection.close()
    except DuplicateObjectError:
        print("pgvector already exists.")

    #
    # Create the table
    #
    #

    await init_db()
    connection = Tortoise.get_connection("default")
    await connection.execute_script(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            "id" UUID NOT NULL  PRIMARY KEY,
            source_id TEXT,
            source TEXT,
            raw_content TEXT,
            content TEXT,
            title TEXT,
            keywords JSONB,
            named_entities JSONB,
            embedding vector(1536),
            search_vector TSVECTOR,
            page_rank DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_search_vector ON {TABLE_NAME} USING GIN(search_vector);

        CREATE TABLE IF NOT EXISTS {TABLE_NAME}_person (
            "id" UUID NOT NULL PRIMARY KEY,
            "name" TEXT,
            "role" TEXT,
            "team" TEXT,
            "department" TEXT,
            embedding vector(1536),
            search_vector TSVECTOR,
            "created_at" TIMESTAMP DEFAULT NOW(),
            "updated_at" TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_person_search_vector ON {TABLE_NAME}_person USING GIN(search_vector);

        CREATE TABLE IF NOT EXISTS {TABLE_NAME}_activity (
            "id" UUID NOT NULL PRIMARY KEY,
            "actor" TEXT,
            "action" TEXT,
            "resource" TEXT,
            embedding vector(1536),
            search_vector TSVECTOR,
            "created_at" TIMESTAMP DEFAULT NOW(),
            "updated_at" TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_activity_search_vector ON {TABLE_NAME}_activity USING GIN(search_vector);
    """)

    #
    # Download the file
    #
    #

    storage_client = storage.Client()
    bucket = storage_client.bucket(STORAGE_BUCKET)
    bucket.blob(STORAGE_BLOB).download_to_filename(LOCAL_FILE)

    #
    # Restore the table
    #
    #

    command = [
        "pg_restore",
        "--host",
        settings.postgres_dict["host"],
        "--port",
        str(settings.postgres_dict["port"]),
        "--username",
        settings.postgres_dict["user"],
        "--dbname",
        settings.postgres_dict["database"],
        "--clean",
        "--no-owner",
        LOCAL_FILE,
    ]

    env = os.environ.copy()
    env["PGPASSWORD"] = settings.postgres_dict["password"] or ""

    process = await asyncio.create_subprocess_exec(
        *command, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        print(f"Error restoring database: {stderr.decode()}")

    return connection


async def extract_context(decision: str, knowledge_base_context: str):
    class Context(BaseModel):
        context_1: str
        context_2: str
        context_3: str
        context_4: str
        context_5: str
        context_6: str
        context_7: str
        context_8: str
        context_9: str
        context_10: str

    messages: list[ChatCompletionMessageParam] = []
    messages.append(
        {
            "role": "system",
            "content": """It's your job to assist decision makers. Return 10 paragraphs of context from the
            knowledge_base, success depends on you being thorough. Clean up, summarize, and make the information
            markdown-formatted and presentable to users. You must not add any information that didn't come from the
            knowledge_base and don't duplicate results.""",
        }
    )
    messages.append(
        {
            "role": "user",
            "content": f"""Based on this knowledge <knowledge_base>{knowledge_base_context}</knowledge_base> What's the
            most helpful context for deciding '{decision}'?""",
        }
    )

    response = await instructor_client.chat.completions.create(
        messages=messages, model=MODEL, response_model=Context, temperature=0
    )

    return "\n* ".join(
        [
            f"\n* {response.context_1}",
            response.context_2,
            response.context_3,
            response.context_4,
            response.context_5,
            response.context_6,
            response.context_7,
            response.context_8,
            response.context_9,
            response.context_10,
        ]
    )


async def get_relevant_keyword_recommendations_from_db(decision: str):
    class RelevantKeywords(BaseModel):
        keywords: list[str] = Field(..., description="Top ten relevant keywords exactly as provided in the list.")

    connection = Tortoise.get_connection("default")

    # Define the query
    search_query = f"""
        SELECT keyword, COUNT(keyword) AS keyword_count
        FROM {TABLE_NAME},
        jsonb_array_elements_text(keywords) AS keyword
        GROUP BY keyword
        ORDER BY keyword_count DESC
        LIMIT 100;
    """

    # Execute the query
    results = await connection.execute_query_dict(search_query)

    # Ask the LLM to pick the most relevant keywords
    messages: list[ChatCompletionMessageParam] = []
    messages.append(
        {
            "role": "system",
            "content": f"Pick the top ten most relevant keywords to the question '{decision}' from the list below.",
        }
    )
    messages.append({"role": "user", "content": f"<keywords>{[result["keyword"] for result in results]}</keywords>"})

    response = await instructor_client.chat.completions.create(
        messages=messages, model=MODEL, response_model=RelevantKeywords, temperature=0
    )
    return response.keywords


async def get_named_entities_recommendations_from_question(decision: str):
    class RelevantNamedEntities(BaseModel):
        named_entities: list[str] = Field(..., description="Top ten named entities in the question.")

    connection = Tortoise.get_connection("default")

    # Define the query
    search_query = f"""
        SELECT named_entity, COUNT(named_entity) AS named_entity_count
        FROM {TABLE_NAME},
        jsonb_array_elements_text(named_entities) AS named_entity
        GROUP BY named_entity
        ORDER BY named_entity_count DESC
        LIMIT 100;
    """

    # Execute the query
    results = await connection.execute_query_dict(search_query)

    # Ask the LLM to pick the most relevant named_entities
    messages: list[ChatCompletionMessageParam] = []
    messages.append(
        {
            "role": "system",
            "content": f"""Extract the top ten most relevant named entities from the question '{decision}' from the
            list below. If it isn't present in the question you must not return it. Double-check this before returning.
            """,
        }
    )
    messages.append(
        {
            "role": "user",
            "content": f"<named_entities>{[result["named_entity"] for result in results]}</named_entities>",
        }
    )

    response = await instructor_client.chat.completions.create(
        messages=messages, model=MODEL, response_model=RelevantNamedEntities, temperature=0
    )
    return response.named_entities


async def query_knowledge_base(decision: str, relevant_keywords: list[str], named_entities: list[str]):
    connection = Tortoise.get_connection("default")

    # Embed the query
    query_embedding = await embed(decision)

    # Convert the embedding list to a string
    query_embedding_str = json.dumps(query_embedding)

    # Specific keywords and named entities to boost
    keywords_str = " | ".join(relevant_keywords)
    named_entities_str = " | ".join(named_entities)

    # Define the query
    search_query = f"""
        WITH stats AS (
            SELECT
                MIN(ts_rank_cd(search_vector, plainto_tsquery('english', $1))) AS min_search_rank,
                MAX(ts_rank_cd(search_vector, plainto_tsquery('english', $1))) AS max_search_rank,
                MIN((SELECT COUNT(*) FROM jsonb_array_elements_text(keywords::jsonb) WHERE value ILIKE ANY (ARRAY[$4])))::FLOAT AS min_keywords_count,
                MAX((SELECT COUNT(*) FROM jsonb_array_elements_text(keywords::jsonb) WHERE value ILIKE ANY (ARRAY[$4])))::FLOAT AS max_keywords_count,
                MIN((SELECT COUNT(*) FROM jsonb_array_elements_text(named_entities::jsonb) WHERE value ILIKE ANY (ARRAY[$5])))::FLOAT AS min_entities_count,
                MAX((SELECT COUNT(*) FROM jsonb_array_elements_text(named_entities::jsonb) WHERE value ILIKE ANY (ARRAY[$5])))::FLOAT AS max_entities_count,
                MIN(COALESCE(page_rank, 0))::FLOAT AS min_page_rank,
                MAX(COALESCE(page_rank, 0))::FLOAT AS max_page_rank,
                MIN((1 - (embedding <=> $2::vector)))::FLOAT AS min_embedding_sim,
                MAX((1 - (embedding <=> $2::vector)))::FLOAT AS max_embedding_sim
            FROM
                {TABLE_NAME}
        )
        SELECT
            id,
            source_id,
            source,
            content,
            keywords,
            named_entities,
            embedding,
            (
                CASE
                    WHEN stats.max_search_rank != stats.min_search_rank THEN
                        ((ts_rank_cd(search_vector, plainto_tsquery('english', $1)) - stats.min_search_rank) / (stats.max_search_rank - stats.min_search_rank)) * $3
                    ELSE 0
                END +
                CASE
                    WHEN stats.max_keywords_count != stats.min_keywords_count THEN
                        (((SELECT COUNT(*) FROM jsonb_array_elements_text(keywords::jsonb) WHERE value ILIKE ANY (ARRAY[$4]))::FLOAT - stats.min_keywords_count) / (stats.max_keywords_count - stats.min_keywords_count)) * $6
                    ELSE 0
                END +
                CASE
                    WHEN stats.max_entities_count != stats.min_entities_count THEN
                        (((SELECT COUNT(*) FROM jsonb_array_elements_text(named_entities::jsonb) WHERE value ILIKE ANY (ARRAY[$5]))::FLOAT - stats.min_entities_count) / (stats.max_entities_count - stats.min_entities_count)) * $7
                    ELSE 0
                END +
                CASE
                    WHEN stats.max_page_rank != stats.min_page_rank THEN
                        ((COALESCE(page_rank, 0)::FLOAT - stats.min_page_rank) / (stats.max_page_rank - stats.min_page_rank)) * $8
                    ELSE 0
                END +
                CASE
                    WHEN stats.max_embedding_sim != stats.min_embedding_sim THEN
                        (((1 - (embedding <=> $2::vector))::FLOAT - stats.min_embedding_sim) / (stats.max_embedding_sim - stats.min_embedding_sim)) * $9
                    ELSE 0
                END
            ) AS score
        FROM
            {TABLE_NAME}, stats
        ORDER BY
            score DESC
        LIMIT 100;
    """

    # Execute the query
    start_time = time.time()
    results = await connection.execute_query_dict(
        search_query,
        [
            decision,
            query_embedding_str,
            RANKING_WEIGHTS["sparse_vector"],
            keywords_str,
            named_entities_str,
            RANKING_WEIGHTS["keywords"],
            RANKING_WEIGHTS["named_entities"],
            RANKING_WEIGHTS["page_rank"],
            RANKING_WEIGHTS["dense_vector"],
        ],
    )
    print(f"Query time: {time.time() - start_time}")

    print("WHY SO FEW RESULTS?")
    print(len(results))

    return [result for result in results if result["score"] > SCORE_THRESHOLD]


async def query_people_knowledge(query: str):
    connection = Tortoise.get_connection("default")

    # Embed the query
    query_embedding = await embed(query)

    # Convert the embedding list to a string
    query_embedding_str = json.dumps(query_embedding)

    # Define the query
    search_query = f"""
        WITH ranked_persons AS (
            SELECT
                id,
                name,
                role,
                team,
                department,
                embedding,
                ts_rank_cd(search_vector, plainto_tsquery('english', $1)) + (1 - (embedding <=> $2::vector))::FLOAT AS score,
                ROW_NUMBER() OVER (PARTITION BY name, role, team, department ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', $1)) + (1 - (embedding <=> $2::vector))::FLOAT DESC) AS row_num
            FROM
                {TABLE_NAME}_person
        )
        SELECT
            id,
            name,
            role,
            team,
            department,
            embedding,
            score + (COUNT(*) OVER (PARTITION BY name, role, team, department) - 1) * 0.1 AS boosted_score
        FROM
            ranked_persons
        WHERE
            row_num = 1
        ORDER BY
            boosted_score DESC
        LIMIT 50;
    """

    # Execute the query
    results = await connection.execute_query_dict(search_query, [query, query_embedding_str])
    print("PEOPLE RESULTS")
    print(results)
    return results


async def query_activity_knowledge(query: str):
    connection = Tortoise.get_connection("default")

    # Embed the query
    query_embedding = await embed(query)

    # Convert the embedding list to a string
    query_embedding_str = json.dumps(query_embedding)

    # Define the query
    search_query = f"""
        SELECT
            id,
            actor,
            action,
            resource,
            embedding,
            ts_rank_cd(search_vector, plainto_tsquery('english', $1)) + (1 - (embedding <=> $2::vector))::FLOAT AS score
        FROM
            {TABLE_NAME}_activity
        ORDER BY
            score DESC
        LIMIT 50;
    """

    # Execute the query
    results = await connection.execute_query_dict(search_query, [query, query_embedding_str])
    print("ACTIVITY RESULTS")
    print(results)
    return results
