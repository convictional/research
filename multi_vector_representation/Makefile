.PHONY: help install index search stats compare test-reranker clean generate-queries classify-queries pool-results

help:
	@echo "Multi-Vector Search Experiment"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make index         - Index all content with ColBERT embeddings"
	@echo "  make index-small   - Index 50 records (for testing)"
	@echo "  make stats         - Show database statistics"
	@echo "  make search QUERY='your query' - Search using multi-vector"
	@echo "  make compare QUERY='your query' - Compare MV vs SV search"
	@echo "  make test-reranker QUERY='your query' - Test production reranker (verbose)"
	@echo "  make clean         - Remove token embeddings from database"
	@echo ""
	@echo "Evaluation:"
	@echo "  make generate-queries     - Generate test queries with Claude"
	@echo "  make classify-queries     - Classify queries (factoid/navigational/exploratory)"
	@echo "  make pool-results         - Pool search results for annotation"
	@echo "  make annotate             - Start annotation UI (http://localhost:8000)"
	@echo "  make evaluate             - Run full evaluation with latency benchmark"
	@echo "  make evaluate-quick       - Run evaluation without latency benchmark"
	@echo "  make evaluate-stratified  - Run evaluation by query type"

install:
	uv sync

index:
	uv run python main.py index

index-small:
	uv run python main.py index --limit 500

index-cpu:
	uv run python main.py index --device cpu

index-small-cpu:
	uv run python main.py index --limit 50 --device cpu

stats:
	uv run python main.py stats

search:
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make search QUERY='your query'"; \
		exit 1; \
	fi
	uv run python main.py search "$(QUERY)"

compare:
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make compare QUERY='your query'"; \
		exit 1; \
	fi
	uv run python main.py compare "$(QUERY)"

test-reranker:
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make test-reranker QUERY='your query'"; \
		exit 1; \
	fi
	uv run python main.py test-reranker "$(QUERY)"

clean:
	psql decide_development -c "UPDATE content SET token_embeddings = NULL;"
	@echo "Removed all token embeddings from database"

generate-queries:
	uv run python main.py generate-queries

classify-queries:
	uv run python main.py classify-queries

pool-results:
	uv run python main.py pool-results

pool-results-cpu:
	uv run python main.py pool-results --device cpu

annotate:
	uv run python annotation_server.py

evaluate:
	uv run python main.py evaluate

evaluate-quick:
	uv run python main.py evaluate --skip-latency

evaluate-stratified:
	uv run python main.py evaluate-stratified
