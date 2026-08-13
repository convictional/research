# Data Integration Experimentation

**Author:** Adam McCabe

This mini-codebase constructs Decision and Metric classes and includes 'database' context on our BigQuery instance - specifically our `star_schema` and `staging` tables, with minimal specificity in the prompts and all context retrieved via RAG. We augment the knowledge with RAG against Guru. The code here uses Langchain as it allows for more easily swapping out foundation models between OpenAI and Anthropic (although that doesn't work as well as you'd like without significant prompt updates).

## Process
### Decision:
1. User input with as little or as much context as the user wants to input - although, more is better (to a point).
2. LLM Brainstormed Metrics and Supportability:
3. LLM is responsible for leveraging Guru to brainstorm metrics that a data analyst would potentially need to investigate to inform the decision being considered. For each of the metrics brainstormed, we retrieve relevant database context and have the LLM return a determination on if the metric can be investigated, and if not, what is required.

### Metrics Analysis
For each metric deemed to be supportable, a Metrics chain is initiated which works through individual LLM calls to:
1. Given the metric, get relevant database context and ask the LLM to return only the needed tables and columns to query the metric (database_schema)
2. Given the database context, we (deterministically) put together a simple select returned_llm_columns from returned_llm_table limit 25 and include these sample results along with the database_schema produced in step 1
3. Given the updated schema from 2, now ask the LLM to write a valid query to allow a data analyst to analyze the metric
4. Attempt to run the query
    - If it fails, pass the query, the error and retrieved database context (against the query and error) to the LLM, now with memory enabled, with the task to remedy the error.
    - Repeat up to 2 times (3 total attempts)
5. Once (if) the query is successful, pass the results, the metric, the decision and retrieved context (against the metric) to the LLM with the task to write a summary analysis of the results that could accompany a chart in an analyst report including follow up questions that may be needed

### Final Report
Finally, given the completed metrics analysis, we pass this along with the original decision and retrieved context (against the decision) to the LLM one more time with the ask to return a report with the following headings:
- Decision overview - what is being considered and based on context, what has been previously decided related to this area.
- Arguments For the Decision: Using all of the context, your experience and broad knowledge, craft an argument in favor of the decision.
- Arguments Against the Decision: Using all of the context, your experience and broad knowledge, craft an argument in opposition of the decision.
- Analysis Summary: Summarize the above analysis as it relates to the decision.
- Potential Alternatives: List at least 3 potential alternatives based on the above context, arguments for and against and decision itself.
- _In this chain, we ask the LLM to assume the identity of a Harvard MBA grad with experience working at McKinsey._

## Running the code:

This code needs to be better optimized for usability - at present, you will need to:
1. From the main `decide` directory, run `poetry install` and then `poetry shell` to initiate a virtual environment with the production dependencies
2. Run `pip install -r experiments/integration_research/requirements.txt` to install the remaining dependencies (_TODO, update the sandbox pyproject deps_)
3. Open `experiments/integration_research/decision_metric_examples.py` and modify the decision description or metric description
4. Ensure you have ENV variables set for OpenAI and BigQuery - I'd recommend using Doppler
5. Run either the example chains with `doppler run -- python experiments/integration_research/decision_metric_examples.py`
