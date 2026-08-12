from datetime import datetime
from uuid import UUID
from typing import List
from itertools import islice

from .settings import settings, logger
from .build_knowledge_store.get_context import get_source_content_from_bq
from .build_knowledge_store.extract_named_entities import EntityExtractor, EntityStore
from .helpers.async_helper import execute_tasks_with_manual_pbar
from .helpers.io import save_checkpoint, load_checkpoint

named_entity_knowledge_store_csv_path = settings.output_path / f"named_entity_knowledge_store-{datetime.now()}.csv"

LLM_TEMPERATURE = 0.0


async def process_content_chunk(chunk: dict, entity_extractor: EntityExtractor) -> List:
    # Use the datetime object directly if it's already in the correct format
    created_at = chunk["created_at"]
    if isinstance(created_at, str):
        created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")

    entities = await entity_extractor.extract_entities_and_facts(
        chunk["content"], UUID(chunk["content_id"]), created_at
    )
    return entities


def batch_iterator(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


async def build_named_entity_knowledge_store_csv():
    """Build the named entity knowledge store CSV from content in BigQuery"""
    logger.info("Starting named entity extraction from BigQuery content...")

    content = get_source_content_from_bq()
    entity_extractor = EntityExtractor()
    entity_store = EntityStore()

    # Process content in batches
    batch_size = 50
    total_batches = (len(content) + batch_size - 1) // batch_size

    # Load last completed batch checkpoint
    last_checkpoint = load_checkpoint("batch_progress")
    start_batch = 0
    if last_checkpoint:
        checkpoint_batch, saved_entities = last_checkpoint
        # Validate checkpoint batch number
        if checkpoint_batch >= total_batches:
            logger.info("All batches were already processed. Proceeding to deduplication...")
            entity_store.collected_entities = saved_entities
            await entity_store.deduplicate_collected_entities()
            entity_store.export_to_csv(named_entity_knowledge_store_csv_path)
            logger.info("Process complete!")
            return

        start_batch = checkpoint_batch
        entity_store.collected_entities = saved_entities
        logger.info(f"Resuming from batch {start_batch + 1} with {len(saved_entities)} entities")

    logger.info(f"Processing {len(content)} documents in {total_batches} batches...")

    for i, content_batch in enumerate(batch_iterator(content, batch_size), start=start_batch):
        if i >= total_batches:
            break

        logger.info(f"\nBatch {i+1}/{total_batches}:")

        tasks = [process_content_chunk(chunk, entity_extractor) for chunk in content_batch]
        results = await execute_tasks_with_manual_pbar(tasks)

        all_entities = [entity for chunk_entities in results for entity in chunk_entities]
        entity_store.collected_entities.extend(all_entities)

        # Save checkpoint after each batch
        save_checkpoint((i + 1, entity_store.collected_entities), "batch_progress")
        logger.info(f"Saved checkpoint for batch {i+1} with {len(entity_store.collected_entities)} total entities")

    logger.info("\nPerforming final de-duplication...")
    await entity_store.deduplicate_collected_entities()

    entity_store.export_to_csv(named_entity_knowledge_store_csv_path)
    logger.info("\nProcess complete!")
