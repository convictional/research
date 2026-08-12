import asyncio
import json
import time
from uuid import uuid4 as generate_uuid

import numpy as np
from bs4 import BeautifulSoup
from dateutil import parser
from experiments.knowledge_search.knowledge import TABLE_NAME, setup_db
from google.cloud import bigquery
from infra.llm import instructor_client
from infra.vectors import embed
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

MODEL = "gpt-4o"

SYSTEM_PROMPT = """
It's your job to extract features from the content provided to help build a knowledge base.
"""


class Person(BaseModel):
    name: str = Field(..., description="Name of the person.")
    role: str = Field(..., description="Role of the person.")
    team: str = Field(..., description="Team of the person.")
    department: str = Field(..., description="Department of the person.")


class Activity(BaseModel):
    actor: str = Field(..., description="Who performed the action.")
    action: str = Field(..., description="Action extracted from the content.")
    resource: str = Field(..., description="Resource extracted from the content.")


class KnowledgeFeatures(BaseModel):
    title: str = Field(..., description="Title extracted from the content.")
    keywords: list[str] = Field(..., description="Keywords extracted from the content.")
    named_entities: list[str] = Field(..., description="All named entities extracted from the content.")
    people: list[Person] = Field(..., description="People extracted from the content.")
    activity: list[Activity] = Field(..., description="Activities extracted from the content.")


async def extract_features(content: str):
    messages: list[ChatCompletionMessageParam] = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": content})

    try:
        return await instructor_client.chat.completions.create(
            messages=messages, model=MODEL, response_model=KnowledgeFeatures
        )
    except Exception:
        return await extract_features(content[: len(content) // 2])


def strip_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ")


def parse_date(date_str):
    try:
        parsed_date = parser.parse(date_str)
        return parsed_date
    except (ValueError, TypeError):
        return None


async def embed_content(content: str):
    try:
        return await embed(content)
    except Exception as e:
        print(f"Content is too long to embed: {e}")

        # Split the content into smaller chunks recursively
        half_length = len(content) // 2
        part1 = content[:half_length]
        part2 = content[half_length:]

        embedding1 = await embed_content(part1)
        embedding2 = await embed_content(part2)

        # Combine the embeddings by averaging them
        combined_embedding = np.mean([embedding1, embedding2], axis=0)
        return combined_embedding.tolist()


async def process_row(row, connection, source):
    print("Processing ", row)

    # Check if source_id already exists
    existing_entry = await connection.execute_query_dict(
        f"SELECT 1 FROM {TABLE_NAME} WHERE source_id = $1",
        [str(row["id"])],
    )
    if existing_entry:
        print(f"Skipping row with source_id {row['id']} as it already exists.")
        return

    cleaned_content = strip_html(row["content"])
    if len(cleaned_content) < 100:
        return

    features = await extract_features(cleaned_content)
    print("Features: ", features)

    embedding = await embed_content(cleaned_content)

    tsvector_result = await connection.execute_query_dict(
        "SELECT to_tsvector('english', $1) AS tsvector", [cleaned_content]
    )
    tsvector = tsvector_result[0]["tsvector"]

    await connection.execute_query(
        f"""
        INSERT INTO {TABLE_NAME} (
            id,
            source_id,
            source,
            raw_content,
            content,
            title,
            keywords,
            named_entities,
            embedding,
            search_vector
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
        )
        """,
        [
            generate_uuid(),
            str(row["id"]),
            source,
            row["content"],
            cleaned_content,
            features.title,
            json.dumps(features.keywords),
            json.dumps(features.named_entities),
            json.dumps(embedding),
            tsvector,
        ],
    )

    for person in features.people:
        content = f"{person.name} {person.role} {person.team} {person.department}"
        embedding = await embed_content(content)

        tsvector_result = await connection.execute_query_dict(
            "SELECT to_tsvector('english', $1) AS tsvector", [content]
        )
        tsvector = tsvector_result[0]["tsvector"]

        await connection.execute_query(
            f"""
            INSERT INTO {TABLE_NAME}_person (
                id,
                name,
                role,
                team,
                department,
                embedding,
                search_vector
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7
            )
            """,
            [
                generate_uuid(),
                person.name,
                person.role,
                person.team,
                person.department,
                json.dumps(embedding),
                tsvector,
            ],
        )

    for activity in features.activity:
        content = f"{activity.actor} {activity.action} {activity.resource}"
        embedding = await embed_content(content)

        tsvector_result = await connection.execute_query_dict(
            "SELECT to_tsvector('english', $1) AS tsvector", [cleaned_content]
        )
        tsvector = tsvector_result[0]["tsvector"]

        await connection.execute_query(
            f"""
            INSERT INTO {TABLE_NAME}_activity (
                id,
                actor,
                action,
                resource,
                embedding,
                search_vector
            ) VALUES (
                $1, $2, $3, $4, $5, $6
            )
            """,
            [
                generate_uuid(),
                activity.actor,
                activity.action,
                activity.resource,
                json.dumps(embedding),
                tsvector,
            ],
        )


async def process_guru_cards(connection):
    client = bigquery.Client()
    query_job = client.query("SELECT * FROM `${GCP_PROJECT}.guru.card`")
    query_results = query_job.result()

    semaphore = asyncio.Semaphore(3)

    async def sem_process_row(row):
        async with semaphore:
            await process_row(row, connection, "${GCP_PROJECT}.guru.card")

    tasks = [sem_process_row(row) for row in query_results]
    await asyncio.gather(*tasks)


async def process_github_issues(connection):
    client = bigquery.Client()
    query_job = client.query("""
        WITH IssueComments AS (
    SELECT
        i.id AS issue_id,
        i.body AS issue_body,
        ic.body AS comment_body,
        i.created_at AS issue_created_at,
        ic.created_at AS comment_created_at,
        i.user_id AS issue_user_id,
        ic.user_id AS comment_user_id
    FROM
        `${GCP_PROJECT}.github.issue` i
    LEFT JOIN
        `${GCP_PROJECT}.github.issue_comment` ic
    ON
        i.id = ic.issue_id
    WHERE
        i.body IS NOT NULL AND ic.body IS NOT NULL AND
        (i.created_at >= '2022-01-01' OR ic.created_at >= '2022-01-01')
),
CombinedContent AS (
    SELECT
        issue_id,
        content,
        content_created_at,
        user_id
    FROM (
        SELECT
            issue_id,
            issue_body AS content,
            issue_created_at AS content_created_at,
            issue_user_id AS user_id
        FROM
            IssueComments
        UNION ALL
        SELECT
            issue_id,
            comment_body AS content,
            comment_created_at AS content_created_at,
            comment_user_id AS user_id
        FROM
            IssueComments
        WHERE
            comment_body IS NOT NULL
    )
),
ContentWithUserNames AS (
    SELECT
        cc.issue_id,
        SUBSTRING(cc.content, 0, 10000) AS content,  -- Truncate content per comment to 10,000 characters
        cc.content_created_at,
        u.name AS user_name,
        ic.issue_created_at
    FROM
        CombinedContent cc
    LEFT JOIN
        `${GCP_PROJECT}.github.user` u
    ON
        cc.user_id = u.id
    LEFT JOIN
        IssueComments ic
    ON
        cc.issue_id = ic.issue_id
)
SELECT
    issue_id,
    issue_created_at,
    STRING_AGG(CONCAT(user_name, ': ', content), '\\n' ORDER BY content_created_at ASC LIMIT 1000) AS combined_content  -- Limit the number of concatenated entries
FROM
    ContentWithUserNames
GROUP BY
    issue_id, issue_created_at;

""")
    query_results = query_job.result(page_size=100)
    pages = query_results.pages

    processed_rows = []
    while True:
        page = next(pages, None)
        if page is None:
            print("No more pages.")
            break

        print("Processing page...")
        for row in page:
            if not row["combined_content"]:
                continue
            content_with_date = f'Created at {row["issue_created_at"]}\n\n{row["combined_content"]}'
            processed_rows.append({"id": row["issue_id"], "content": content_with_date})

    semaphore = asyncio.Semaphore(3)

    async def sem_process_row(row):
        async with semaphore:
            await process_row(row, connection, "${GCP_PROJECT}.github.issue")

    tasks = [sem_process_row(row) for row in processed_rows]
    await asyncio.gather(*tasks)


async def main():
    connection = await setup_db()

    start_time = time.time()
    await process_guru_cards(connection)
    done_time = time.time()
    print(f"Processed Guru cards in {done_time - start_time} seconds")

    start_time = time.time()
    await process_github_issues(connection)
    done_time = time.time()
    print(f"Processed GitHub issues in {done_time - start_time} seconds")
