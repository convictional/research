import asyncio
import faiss
import tiktoken
import pandas as pd
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from tqdm import tqdm

from ..knowledge_graph import KnowledgeGraph

from ..config.experiment_settings import settings
from ..config.prompts import (
    DECISION_OPTION_PREDICTION_INPUT_SYSTEM_PROMPT,
    DECISION_OPTION_PREDICTION_RESULTS_SYSTEM_PROMPT,
)

from ..utils.async_helper import limited_task, wrap_task_progress_bar
from ..utils.embeddings import aembed_to_faiss, query_faiss_index, embed
from ..utils.tokens import split_chunks_by_tokens, trunc_on_tokens
from ..utils.source_data import get_app_decisions_as_df, get_source_content_from_bq, query_local_postgres_db
from ..utils.instruct_llm import ainstruct_llm

from ..graph_traversal.traversal_tools import (
    InducedSubGraphTool,
    AllShortestPathsToLabelTool,
    SingleNodeOneHopNeighborsTool,
)
from ..graph_traversal.traverse_graph import aget_and_init_current_graph, get_vss_keywords

tokenizer = tiktoken.encoding_for_model("gpt-4")


class GraphPredictionToolInputs(BaseModel):
    ISG: InducedSubGraphTool = Field(
        ...,
        description="Induced Subgraph Tool to generate a subgraph from the current graph state based on a list of passed nodes and all paths of length 2 or less between them. Must provide at least two distinct nodes.",
    )

    OHN: SingleNodeOneHopNeighborsTool = Field(
        ...,
        description="Single Node One-Hop Neighbors tool to find all the one-hop neighbors of a central node. Great for exploring the immediate connections of a node (both ingoing and outgoing).",
    )

    people_paths: AllShortestPathsToLabelTool = Field(
        ...,
        description="All Shortest Paths tool to find all the shortest paths from the user provided identified decision node to 'People' nodes. You must use the user provided identified decision node as the start node and 'People' as the target node label. Focus primarily on impacts to People nodes who are employees of Convictional.",
    )

    tool_paths: AllShortestPathsToLabelTool = Field(
        ...,
        description="All Shortest Paths tool to find all the shortest paths from the user provided identified decision node to 'BusinessTools' nodes. You must use the user provided identified decision node as the start node and 'BusinessTools' as the target node label.",
    )

    process_paths: AllShortestPathsToLabelTool = Field(
        ...,
        description="All Shortest Paths tool to find all the shortest paths from the user provided identified decision node to 'BusinessProcesses' nodes. You must use the user provided identified decision node as the start node and 'BusinessProcesses' as the target node label.",
    )

    org_paths: AllShortestPathsToLabelTool = Field(
        ...,
        description="All Shortest Paths tool to find all the shortest paths from the user provided identified decision node to 'Organizations' nodes. You must use the user provided identified decision node as the start node and 'Organizations' as the target node label.",
    )

    principles_paths: AllShortestPathsToLabelTool = Field(
        ...,
        description="All Shortest Paths tool to find all the shortest paths from the user provided identified decision node to 'BusinessPrinciples' nodes. You must use the user provided identified decision node as the start node and 'BusinessPrinciples' as the target node label.",
    )

    measures_paths: AllShortestPathsToLabelTool = Field(
        ...,
        description="All Shortest Paths tool to find all the shortest paths from the user provided identified decision node to 'BusinessMeasures' nodes. You must use the user provided identified decision node as the start node and 'BusinessMeasures' as the target node label.",
    )

    objectives_paths: AllShortestPathsToLabelTool = Field(
        ...,
        description="All Shortest Paths tool to find all the shortest paths from the user provided identified decision node to 'BusinessObjectives' nodes. You must use the user provided identified decision node as the start node and 'BusinessObjectives' as the target node label.",
    )

    unidentified_decision_paths: AllShortestPathsToLabelTool = Field(
        ...,
        description="All Shortest Paths tool to find all the shortest paths from the user provided identified decision node to 'UnidentifiedBusinessDecisions' nodes. You must use the user provided identified decision node as the start node and 'UnidentifiedBusinessDecisions' as the target node label.",
    )


class DecisionOptionPrediction(BaseModel):
    objectives_impacted: str = Field(
        ...,
        description="Describe what, if any, impact the decision will have on Convictional's objectives. Only reference objectives found in context. Only provide the top 3 objectives impacted reasoned by support in context and your own intuition.",
    )

    people_impacted: str = Field(
        ...,
        description="Describe the people who will be impacted by this decision option. This can include employees, customers, partners, etc. You must provide specific information about the impact on each individual. Only reference people found in context. Only provide the top 3 people impacted reasoned by support in context and your own intuition.",
    )

    tools_impact: str = Field(
        ...,
        description="Describe what, if any, impact the decision will have on the tools and systems used at Convictional. This can include internal tools, external tools, and integrations. Only reference tools found in context. Only provide the top 3 tools impacted reasoned by support in context and your own intuition.",
    )

    processes_impact: str = Field(
        ...,
        description="Describe what, if any, impact the decision will have on the processes at Convictional. Only reference processes found in context. Only provide the top 3 processes impacted reasoned by support in context and your own intuition.",
    )

    org_impact: str = Field(
        ...,
        description="Describe what, if any, impact the decision will have on organizations other than Convictional. This can include partners, customers, competitors, etc. Only reference organizations found in context. Only provide the top 3 organizations impacted reasoned by support in context and your own intuition.",
    )

    financial_impact: str = Field(
        ...,
        description="Describe what, if any, financial impact the decision will have on Convictional. This can include costs, revenue, profit, etc. Only reference financial impacts found in context.",
    )

    principles_conflicts: str = Field(
        ...,
        description="Describe any conflicts between the decision option and Convictional's principles. Only reference principles found in context.",
    )

    measures_to_measure: str = Field(
        ...,
        description="Describe what measures should be considered in tracking the outcome of this decision. Only reference the top 3-5 measures to be considered, reasoned by support in context and your own intuition.",
    )

    risks: str = Field(
        ...,
        description="Describe the top three risks to be considered given all other context. Be specific, informative and concise.",
    )

    potential_gains: str = Field(
        ...,
        description="Describe the top three potential gains to be considered given all other context. Be specific, informative and concise.",
    )

    implications: str = Field(
        ...,
        description="Describe the top 3-5 implications of the decision option, reasoned by support in context and your own intuition. This can include the impact on Convictional's objectives, people, tools, processes, organizations, financials, principles, measures, risks, and potential gains. Only reference implications found in context.",
    )


class Judgement(BaseModel):
    score: int = Field(
        ...,
        description="The score given to the prediction. Should be between 1 and 5 with 5 being the highest and representing a prediction that is specific, informative and rooted in context.",
    )

    explanation: str = Field(
        ...,
        description="The explanation for the score given to the prediction. This should explain why the prediction was given the score it was.",
    )


class PredictionJudgements(BaseModel):
    app_only_judgement: Judgement = Field(
        ...,
        description="The judgement for the prediction made using the app decision data only.",
    )

    vss_judgement: Judgement = Field(
        ...,
        description="The judgement for the prediction made using the VSS data only.",
    )

    frag_judgement: Judgement = Field(
        ...,
        description="The judgement for the prediction made using the fRAG data only.",
    )

    graph_judgement: Judgement = Field(
        ...,
        description="The judgement for the prediction made using the graph data only.",
    )


# Constants


app_only_path = settings.output_path / "app_decision_option_predictions.csv"
vss_path = settings.output_path / "vss_decision_option_predictions.csv"
frag_path = settings.output_path / "frag_decision_option_predictions.csv"
graph_path = settings.output_path / "graph_decision_option_predictions.csv"
judge_output_path = settings.output_path / "judged_decision_option_predictions.csv"


FRAG_SEARCH_QUERY = """
    WITH full_text_results AS (
        SELECT {search_fields},
            ts_rank_cd(text_search, query) AS content_rank,
            ts_rank_cd(to_tsvector('english', title), query) AS title_rank,
            CASE
                WHEN REGEXP_REPLACE(LOWER(title), '\\s+', '', 'g') =
                    REGEXP_REPLACE(LOWER($1), '\\s+', '', 'g')
                THEN 1
                ELSE 0
            END AS exact_title_match
        FROM "content" as searchable_table, plainto_tsquery('english', $1) query
        WHERE (text_search @@ query OR to_tsvector('english', title) @@ query)
        AND organization_id = $3
    ),
    vector_results AS (
        SELECT {search_fields},
            embedding <=> $2::vector AS distance
        FROM "content" as searchable_table
        WHERE organization_id = $3
    )
    SELECT {search_fields}, combined_score
    FROM (
        SELECT DISTINCT ON (id) {search_fields},
            (COALESCE(1 / NULLIF(title_rank, 0), 0) +
                COALESCE(1 / NULLIF(content_rank, 0), 0) +
                COALESCE(1 / NULLIF(distance + 1, 0), 0) +
                (CASE WHEN exact_title_match = 1 THEN 100000 ELSE 0 END) -
                (CASE WHEN source = 'github' THEN 100 ELSE 0 END)
            ) AS combined_score
        FROM (
            SELECT {search_fields}, content_rank, title_rank,
                exact_title_match, NULL::float AS distance
            FROM full_text_results
            UNION
            SELECT {search_fields}, NULL::float AS content_rank,
                NULL::float AS title_rank, NULL::int as exact_title_match, distance
            FROM vector_results
        ) combined
        ORDER BY id, combined_score DESC
    ) unique_results
    ORDER BY combined_score DESC
    LIMIT {RESULTS_LIMIT};
"""

# Helpers


async def task_manager(tasks: List[asyncio.Task]) -> None:
    pbar = tqdm(total=len(tasks), desc="Processing predictions...")
    wrapped_tasks = [wrap_task_progress_bar(task, pbar) for task in tasks]

    _ = await asyncio.gather(*wrapped_tasks)
    pbar.close()


async def get_prediction_df(option: dict, context: dict, idx: int, decision_options_df: pd.DataFrame) -> pd.DataFrame:
    print("Getting prediction results from LLM...")
    trunced_context = trunc_on_tokens(context)
    results_system_prompt = DECISION_OPTION_PREDICTION_RESULTS_SYSTEM_PROMPT.format(tool_results=trunced_context)

    prediction_results, _ = await ainstruct_llm(
        system_prompt=results_system_prompt,
        user_prompt=f"Decision option and context for which we're making predictions about implications, risks and potential gains: {option.to_json()}",
        temperature=0.0,
        response_model=DecisionOptionPrediction,
    )

    print("Prediction results:")
    print(prediction_results)

    decision_options_df.loc[idx, "risks"] = prediction_results.risks
    decision_options_df.loc[idx, "potential_gains"] = prediction_results.potential_gains
    decision_options_df.loc[idx, "implications"] = prediction_results.implications
    decision_options_df.loc[idx, "financial_impact"] = prediction_results.financial_impact
    decision_options_df.loc[idx, "people_impacted"] = prediction_results.people_impacted
    decision_options_df.loc[idx, "tools_impact"] = prediction_results.tools_impact
    decision_options_df.loc[idx, "processes_impact"] = prediction_results.processes_impact
    decision_options_df.loc[idx, "org_impact"] = prediction_results.org_impact
    decision_options_df.loc[idx, "objectives_impacted"] = prediction_results.objectives_impacted
    decision_options_df.loc[idx, "principles_conflicts"] = prediction_results.principles_conflicts
    decision_options_df.loc[idx, "measures_to_measure"] = prediction_results.measures_to_measure

    return decision_options_df


# Asyncified get_graph_prediction_tool_inputs
async def get_graph_prediction_tool_inputs(option: dict, current_graph: KnowledgeGraph) -> GraphPredictionToolInputs:
    print(f"Processing decision option: {option['option_title']}")
    keyword_query = f"Decision Title: {option['decision_title']} - Option Title: {option['option_title']} Option Description: {option['option_description']}"
    vss_keywords = get_vss_keywords(keyword_query, [])
    print(f"Searching graph with keywords: {vss_keywords}")
    similar_subgraph = await current_graph.get_similar_subgraph(vss_keywords)

    inputs_system_prompt = DECISION_OPTION_PREDICTION_INPUT_SYSTEM_PROMPT.format(similar_subgraph=similar_subgraph)

    print("Getting prediction tool inputs from LLM...")
    prediction_inputs, _ = await ainstruct_llm(
        system_prompt=inputs_system_prompt,
        user_prompt=f"Decision Details: {option.to_json()}",
        temperature=0.0,
        response_model=GraphPredictionToolInputs,
    )
    print("Prediction inputs:")
    print(prediction_inputs)

    return prediction_inputs


def get_graph_prediction_tool_results(
    tool: GraphPredictionToolInputs, current_graph: KnowledgeGraph
) -> Dict[str, Any]:
    """Execute all the available traversal tools and return the results along with LLM summary."""
    subgraph = tool.ISG.get_subgraph(current_graph)
    neighbours = tool.OHN.get_neighbours()
    people_paths = tool.people_paths.get_paths()
    tool_paths = tool.tool_paths.get_paths()
    process_paths = tool.process_paths.get_paths()
    org_paths = tool.org_paths.get_paths()
    principles_paths = tool.principles_paths.get_paths()
    measures_paths = tool.measures_paths.get_paths()
    objectives_paths = tool.objectives_paths.get_paths()
    unidentified_decision_paths = tool.unidentified_decision_paths.get_paths()

    results = {
        "Induced Subgraph Tool": subgraph,
        "Single Node One-Hop Neighbors Tool": neighbours,
        "People Paths": people_paths,
        "Tool Paths": tool_paths,
        "Process Paths": process_paths,
        "Org Paths": org_paths,
        "Principles Paths": principles_paths,
        "Measures Paths": measures_paths,
        "Objectives Paths": objectives_paths,
        "Unidentified Decision Paths": unidentified_decision_paths,
    }
    return results


# Asyncified get_graph_predictions
async def get_graph_predictions(concurrent_tasks: int = 25, delay_between_tasks: float = 0.1) -> None:
    print("Getting and initializing current graph...")
    current_graph = await aget_and_init_current_graph()
    print("Getting decision options from the database...")
    decision_options_df = get_app_decisions_as_df()

    semaphore = asyncio.Semaphore(concurrent_tasks)

    tasks = [
        limited_task(
            process_graph_prediction(option, current_graph, idx, decision_options_df),
            semaphore,
            delay_between_tasks,
        )
        for idx, option in decision_options_df.iterrows()
    ]

    task_manager(tasks)

    decision_options_df.to_csv(graph_path, index=False)


async def process_graph_prediction(option, current_graph, idx, decision_options_df):
    prediction_inputs = await get_graph_prediction_tool_inputs(option, current_graph)

    print("Querying neo4j for prediction tool results...")
    graph_context = get_graph_prediction_tool_results(prediction_inputs, current_graph)

    decision_options_df = await get_prediction_df(option, graph_context, idx, decision_options_df)
    return decision_options_df


# App Only
async def get_app_only_predictions(concurrent_tasks: int = 25, delay_between_tasks: float = 0.1):
    decision_options_df = get_app_decisions_as_df()

    semaphore = asyncio.Semaphore(concurrent_tasks)

    tasks = [
        limited_task(
            process_app_only_prediction(option, idx, decision_options_df),
            semaphore,
            delay_between_tasks,
        )
        for idx, option in decision_options_df.iterrows()
    ]

    task_manager(tasks)

    decision_options_df.to_csv(app_only_path, index=False)


async def process_app_only_prediction(option, idx, decision_options_df):
    tool_results = {}
    decision_options_df = await get_prediction_df(option, tool_results, idx, decision_options_df)
    return decision_options_df


# VSS
async def init_content_faiss_index():
    print("Getting content from BigQuery...")
    raw_content = get_source_content_from_bq()
    content = split_chunks_by_tokens(raw_content, 8000)
    content_index = faiss.IndexFlatL2(settings.faiss_embedding_dimension)

    content_to_embed = [f"{chunk['title']} {chunk['content']}" for chunk in content]
    content_index = await aembed_to_faiss(
        content_to_embed,
        content_index,
        settings.faiss_embedding_model,
        settings.faiss_embedding_dimension,
        max_concurrent_tasks=50,
        delay_between_tasks=0.2,
    )
    return content_index


async def get_vss_predictions(concurrent_tasks: int = 25, delay_between_tasks: float = 0.1):
    content_index = await init_content_faiss_index()
    decision_options_df = get_app_decisions_as_df()

    semaphore = asyncio.Semaphore(concurrent_tasks)

    tasks = [
        limited_task(
            process_vss_prediction(option, content_index, idx, decision_options_df),
            semaphore,
            delay_between_tasks,
        )
        for idx, option in decision_options_df.iterrows()
    ]

    task_manager(tasks)

    decision_options_df.to_csv(vss_path, index=False)


async def process_vss_prediction(option, content_index, idx, decision_options_df):
    option_query = f"Decision Title: {option['decision_title']} - Option Title: {option['option_title']} Option Description: {option['option_description']}"
    vss_keywords = get_vss_keywords(option_query, [])
    tool_results = await query_faiss_index(content_index, vss_keywords, k=15)

    decision_options_df = await get_prediction_df(option, tool_results, idx, decision_options_df)
    return decision_options_df


# fRAG
async def get_frag_results(query: str, top_k: int = 12) -> List[dict]:
    query_embedding = "[" + ", ".join(str(value) for value in embed(query, embedding_dim=1536)) + "]"
    search_fields = ", ".join(["id", "external_id", "source", "title", "keywords", "search_content"])

    search_query = FRAG_SEARCH_QUERY.format(search_fields=search_fields, RESULTS_LIMIT=top_k)
    results = await query_local_postgres_db(
        search_query,
        query,
        query_embedding,
        "00000000-0000-0000-0000-000000000000",
    )
    return results


async def get_frag_predictions(concurrent_tasks: int = 25, delay_between_tasks: float = 0.1):
    decision_options_df = get_app_decisions_as_df()

    semaphore = asyncio.Semaphore(concurrent_tasks)

    tasks = [
        limited_task(
            process_frag_prediction(option, idx, decision_options_df),
            semaphore,
            delay_between_tasks,
        )
        for idx, option in decision_options_df.iterrows()
    ]

    task_manager(tasks)

    decision_options_df.to_csv(frag_path, index=False)


async def process_frag_prediction(option, idx, decision_options_df):
    print(f"Processing decision option: {option['option_title']}")
    print("Querying postgres for results...")
    frag_query = f"Decision Title: {option['decision_title']} - Option Title: {option['option_title']} Option Description: {option['option_description']}"
    frag_results = await get_frag_results(frag_query, 20)

    decision_options_df = await get_prediction_df(option, frag_results, idx, decision_options_df)
    return decision_options_df


# LLM as a Judge
async def llm_judge(prediction: str) -> PredictionJudgements:
    print("Getting prediction judgement from LLM...")
    judgement_system_prompt = "We are comparing four different rag approaches to make a prediction about the implications, risks and potential gains of a decision option. Please provide a score between 1 and 5 for each approach, with 5 being the highest and representing a prediction that is specific, informative and rooted in context. You should always provide an explanation for your score."
    prediction_judgement, _ = await ainstruct_llm(
        system_prompt=judgement_system_prompt,
        user_prompt=f"Prediction: {prediction}",
        temperature=0.0,
        response_model=PredictionJudgements,
    )
    print("Prediction judgement:")
    print(prediction_judgement)
    return prediction_judgement


def combine_predictions(row):
    combined = {}
    combined["graphrag"] = row["graphrag"]
    combined["frag"] = row["frag"]
    combined["vss"] = row["vss"]
    combined["app_only"] = row["app_only"]
    return combined


async def judge_predictions(concurrent_tasks: int = 25, delay_between_tasks: float = 0.1):
    # load the predictions already made from csv

    app_only_df = pd.read_csv(app_only_path)
    vss_df = pd.read_csv(vss_path)
    frag_df = pd.read_csv(frag_path)
    graph_df = pd.read_csv(graph_path)

    # Create a merged DataFrame
    merged_df = app_only_df.copy()
    relevant_columns = merged_df.columns[merged_df.columns.get_loc("risks") :]

    # Iterate over relevant columns and combine the corresponding cells into dictionaries
    for column in relevant_columns:
        combined_column = pd.DataFrame(
            {
                "graphrag": graph_df[column],
                "frag": frag_df[column],
                "vss": vss_df[column],
                "app_only": app_only_df[column],
            }
        )
        merged_df[column] = combined_column.apply(combine_predictions, axis=1)

    # iterate over the dataframe and judge the predictions

    semaphore = asyncio.Semaphore(concurrent_tasks)

    tasks = [
        limited_task(
            process_judge_prediction(row, idx, relevant_columns, merged_df),
            semaphore,
            delay_between_tasks,
        )
        for idx, row in merged_df.iterrows()
    ]

    task_manager(tasks)

    # save the dataframe
    merged_df.to_csv(judge_output_path, index=False)


async def process_judge_prediction(row, idx, relevant_columns, merged_df):
    for column in relevant_columns:
        # get the predictions
        predictions = row[column]

        # get the judgement
        judgement = await llm_judge(predictions)

        # update the dataframe
        merged_df.loc[idx, f"{column}_judgement"] = str(judgement)
        merged_df.loc[idx, f"{column}app_only_judgement_score"] = judgement.app_only_judgement.score
        merged_df.loc[idx, f"{column}vss_judgement_score"] = judgement.vss_judgement.score
        merged_df.loc[idx, f"{column}frag_judgement_score"] = judgement.frag_judgement.score
        merged_df.loc[idx, f"{column}graph_judgement_score"] = judgement.graph_judgement.score
