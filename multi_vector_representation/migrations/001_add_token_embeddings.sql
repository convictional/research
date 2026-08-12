-- Add token_embeddings column for ColBERT multi-vector storage
-- Using pgvector array syntax: vector(1024)[] stores multiple 1024-dim vectors per document
-- Note: jina-colbert-v2 uses 1024 dimensions, not 128

ALTER TABLE content
ADD COLUMN IF NOT EXISTS token_embeddings vector(1024)[];

-- Add index for metadata tracking
CREATE INDEX IF NOT EXISTS idx_content_token_embeddings_exists
ON content ((token_embeddings IS NOT NULL));

-- Rollback script:
-- ALTER TABLE content DROP COLUMN IF EXISTS token_embeddings;
-- DROP INDEX IF EXISTS idx_content_token_embeddings_exists;
