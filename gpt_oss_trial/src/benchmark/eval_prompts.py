"""Cherry-picked request IDs for evaluation."""

import asyncpg
import json

EVAL_REQUEST_IDS = [
    "78e06638-a471-44fb-bd48-cf35c572f28c",
    "aea58cff-ef4e-45ea-b77a-97ef497a0e25",
    "73cdb411-f298-4db9-b834-449ac4fc74ef",
    "c54689a5-a2df-4db6-8098-528aae3500ef",
    "2670dd05-6402-4e53-a593-9d776dbdfa24",
    "7abade01-890a-4c80-97d9-98047e3e3a41",
]


async def fetch_eval_request_bodies(db_name: str) -> list[dict]:
    """
    Fetch the cherry-picked evaluation request bodies from database.

    Args:
        db_name: Name of the PostgreSQL database

    Returns:
        List of request body dicts in the same order as EVAL_REQUEST_IDS
    """
    conn = await asyncpg.connect(database=db_name, user="decide", host="localhost")

    try:
        rows = await conn.fetch(
            """
            SELECT id, request_body
            FROM llmrequest
            WHERE id = ANY($1::uuid[])
            """,
            EVAL_REQUEST_IDS,
        )

        id_to_body = {}
        for row in rows:
            request_body_json = row["request_body"]
            request_body = json.loads(request_body_json) if isinstance(request_body_json, str) else request_body_json
            id_to_body[str(row["id"])] = request_body

        request_bodies = []
        for req_id in EVAL_REQUEST_IDS:
            if req_id in id_to_body:
                request_bodies.append(id_to_body[req_id])

        return request_bodies
    finally:
        await conn.close()
