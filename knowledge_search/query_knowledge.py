import asyncio
from typing import List, Dict
from pydantic import BaseModel
import tiktoken
from experiments.knowledge_search.knowledge import (
    RANKING_WEIGHTS,
    SCORE_THRESHOLD,
    get_named_entities_recommendations_from_question,
    get_relevant_keyword_recommendations_from_db,
    query_activity_knowledge,
    query_knowledge_base,
    query_people_knowledge,
    setup_db,
)

from experiments.knowledge_graphs.src.utils.instruct_llm import instruct_llm
from experiments.knowledge_graphs.src.config.experiment_settings import settings

MODEL = "gpt-4o"
MAX_KNOWLEDGE_TOKENS = 90000
MAX_KNOWLEDGE_CHARS = 1048576


def print_dict_except_keys(d: dict, keys_to_exclude):
    for key, value in d.items():
        if key not in keys_to_exclude:
            print(f"{key}: {value}")


class KnowledgeSearchV3:
    def __init__(self, use_ranking: bool = False, name: str = "KnowledgeSearchV3"):
        self.use_ranking = use_ranking
        self.name = f"{name}{'WithLLMRanking' if use_ranking else ''}"

    async def setup(self) -> None:
        await setup_db()

    async def hyperparameters(self):
        return {
            "model": MODEL,
            "keywords_weight": RANKING_WEIGHTS["keywords"],
            "named_entities_weight": RANKING_WEIGHTS["named_entities"],
            "sparse_vector_weight": RANKING_WEIGHTS["sparse_vector"],
            "dense_vector_weight": RANKING_WEIGHTS["dense_vector"],
            "page_rank_weight": RANKING_WEIGHTS["page_rank"],
            "score_threshold": SCORE_THRESHOLD,
        }

    async def find_knowledge(self, question: str) -> list[str]:
        tasks = [
            get_relevant_keyword_recommendations_from_db(question),
            get_named_entities_recommendations_from_question(question),
        ]
        relevant_keywords, named_entities = await asyncio.gather(*tasks)

        knowledge_base_docs = await query_knowledge_base(
            decision=question, relevant_keywords=relevant_keywords, named_entities=named_entities
        )

        print("*" * 80)
        print("SCORES:")
        for result in knowledge_base_docs:
            print_dict_except_keys(result, ["embedding", "content"])
            print(result["content"][:100])
            print("---")

        print("*" * 80)
        print(f"RELEVANT KEYWORDS FOR {question}:")
        for keyword in relevant_keywords:
            print(keyword)
        print("*" * 80)

        print(f"RELEVANT NAMED ENTITIES FOR {question}:")
        for entity in named_entities:
            print(entity)
        print("*" * 80)

        related_activity = await query_activity_knowledge(query=question)
        related_activity_string = "Most related activity, use this activity to influence the response, considering its relevance, but do not treat their information as the only or absolute truth. Provide a balanced and nuanced answer influenced by this alongside other relevant information.:\n"
        for activity in related_activity:
            related_activity_string += (
                f"Actor: {activity['actor']}, Action: {activity['action']}, Resource: {activity['resource']}\n"
            )

        print("RELATED ACTIVITY:")
        print(related_activity_string)
        activity_knowledge = {"content": related_activity_string}

        related_people = await query_people_knowledge(query=question)
        related_people_string = "Most related people, use these people to influence the response, considering their relevance and expertise, but do not treat their information as the only or absolute truth. Provide a balanced and nuanced answer incorporating their perspectives alongside other relevant information.:\n"
        for person in related_people:
            related_people_string += f"Name: {person['name']}, Role: {person['role']}, Team: {person['team']}, Department: {person['department']}\n"
        people_knowledge = {"content": related_people_string}

        print("RELATED PEOPLE:")
        print(related_people_string)

        enc = tiktoken.encoding_for_model(MODEL)

        current_tokens = len(enc.encode(activity_knowledge["content"])) + len(enc.encode(people_knowledge["content"]))
        current_length = len(activity_knowledge["content"]) + len(people_knowledge["content"])
        knowledge_subset = [activity_knowledge, people_knowledge]

        for doc in knowledge_base_docs:
            doc_tokens = len(enc.encode(doc["content"]))
            doc_length = len(doc["content"])

            is_too_many_tokens = current_tokens + doc_tokens >= MAX_KNOWLEDGE_TOKENS
            is_too_long = current_length + doc_length >= MAX_KNOWLEDGE_CHARS

            if is_too_many_tokens or is_too_long:
                break

            knowledge_subset.append(doc)
            current_tokens += doc_tokens
            current_length += doc_length

        if not knowledge_subset:
            raise RuntimeError("No knowledge base documents fit within the token limit")

        print(f"Number of knowledge base docs being used: {len(knowledge_subset)}")
        print(f"Total tokens: {current_tokens}")
        print(f"Total length: {current_length}")
        print("*" * 80)

        _ranked_context = [doc["content"] for doc in knowledge_subset]

        if self.use_ranking:
            _ranked_context = rank_results(question, _ranked_context)

        return _ranked_context


class fRagResults(BaseModel):
    results: List[str]


RANK_RESULTS_SYSTEM_PROMPT = """
You are tasked with re-ranking the following results based on the relevance to the user's query. The results were retrieved from a feature and named entity index using a hybrid search approach. The results may include:
- chunks of content
- entire documents
- code snippets
- data
and have had some deterministic ranking effort put into them based on individual retrieval scores using normalization and weighting.

Your goal is to return the most relevant results that directly assist in answering the user's query. You should:
1. Evaluate and rank the results based on their relevance to the query.
2. Return results at the granularity of a path or a node.
3. Return only results that directly answer the question or provide relevant context for reasoning about such.

Guidelines:
- Do not modify the content of the results; only re-rank and format them as needed.
- You may exclude less relevant results, but ensure to return at least one result, even if it appears irrelevant.
- Aim for precision, recall, relevancy and clarity in your rankings to facilitate an effective response to the user's query.
- The most relevant or important information should be ranked higher than less relevant or important information.
"""


def rank_results(user_query: str, graph_traversal_results: Dict) -> List[str]:
    # Call instruct_llm to rank the results
    system_prompt = RANK_RESULTS_SYSTEM_PROMPT
    user_prompt = f"User query: {user_query}\nResults: {graph_traversal_results}"

    response, _ = instruct_llm(
        system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0, response_model=fRagResults
    )

    # Assuming the response contains a ranked list of results
    return response.results
