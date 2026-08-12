from pathlib import Path
import pandas as pd
import faiss
from pydantic import BaseModel, Field

from .settings import settings
from .helpers.embeddings import aembed_to_faiss, aquery_faiss_index
from common.io import dump_to_pickle_file, load_pickle_file


async def initialize_named_entity_knowledge_store(input_file_path: Path, load_faiss_index_from_cache: bool = False):
    """
    Initialize a named entity knowledge store.
    """
    print("Initializing named entity knowledge store...")

    knowledge_store = NamedEntityKnowledgeStore(input_file_path)
    await knowledge_store.ainit_faiss_index(load_faiss_index_from_cache)

    return knowledge_store


class NamedEntityQueryResult(BaseModel):
    entity_name: str = Field(..., title="The name of the named entity")
    category: str = Field(None, title="The category of the named entity")
    description: str = Field(None, title="The description of the named entity")
    index: int = Field(None, title="The index of the named entity in the knowledge store")
    distance: float = Field(None, title="The distance of the named entity from the query string")


class NamedEntityKnowledgeStore:
    """
    This is an object for storing named entity knowledge.

    This is just a dummy knowledge store, thus, the logic is not meant to be "production" worthy.

    The mock data was generated using Claude: https://claude.ai/share/6cccabd7-52da-47bc-b779-5d0f60e9973f
    """

    def __init__(self, input_file_path: Path):
        self.knowledge_store_data: list[dict] = self._load_raw_knowledge_store_data(input_file_path)
        print(f"Loaded {len(self.knowledge_store_data)} entities into knowledge store.")
        self.knowledge_store_index: faiss.IndexFlatL2 | None = None

    def _load_raw_knowledge_store_data(self, input_file_path: Path) -> list[dict]:
        """
        Load raw knowledge store data from csv.
        The result is a list of dictionaries, where each dictionary represents a row in the csv.
        """
        print(f"Loading knowledge store data from {input_file_path}...")

        df = pd.read_csv(input_file_path)
        return df.to_dict(orient="records")

    async def ainit_faiss_index(self, load_faiss_index_from_cache: bool):
        """
        Initialize a faiss index.
        """
        print("Initializing faiss index...")

        pickle_path = settings.output_path / "named_entity_knowledge_store_faiss_index.pkl"

        if load_faiss_index_from_cache:
            print("Loading faiss index from cache...")
            self.knowledge_store_index = load_pickle_file(pickle_path)
        else:
            print("Initializing faiss index from knowledge store data...")
            self.knowledge_store_index = faiss.IndexFlatL2(settings.faiss_embedding_dimension)
            texts_to_embed = [f"{data["entity_name"]} {data["description"]}" for data in self.knowledge_store_data]
            self.knowledge_store_index = await aembed_to_faiss(
                texts_to_embed,
                self.knowledge_store_index,
                settings.faiss_embedding_model,
                settings.faiss_embedding_dimension,
            )

            print("Dumping faiss index to cache...")
            dump_to_pickle_file(self.knowledge_store_index, pickle_path)

    async def asearch_similar_entities(
        self, user_query: str, num_entities: int = 10
    ) -> tuple[list[float], list[dict]]:
        """
        Get similar entities to a user query, by querying the knowledge store faiss index.
        """
        distances, similar_entity_indices = await aquery_faiss_index(
            query_text=user_query,
            index=self.knowledge_store_index,
            top_k=num_entities,
            embedding_model=settings.faiss_embedding_model,
            embedding_dim=settings.faiss_embedding_dimension,
        )
        if all(x == -1 for x in similar_entity_indices):
            return [], []
        similar_entity_indices = list(map(int, similar_entity_indices))
        similar_entities = [self.knowledge_store_data[idx] for idx in similar_entity_indices]

        query_results = [
            NamedEntityQueryResult(
                entity_name=entity["entity_name"],
                category=entity["category"],
                description=entity["description"],
                index=index,
                distance=distance,
            )
            for entity, distance, index in zip(similar_entities, distances, similar_entity_indices)
        ]

        return query_results
