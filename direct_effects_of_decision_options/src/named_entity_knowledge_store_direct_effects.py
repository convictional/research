from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from .settings import settings
from .knowledge_store import NamedEntityKnowledgeStore, NamedEntityQueryResult, initialize_named_entity_knowledge_store
from .decisions_data import get_decision_data, Decision
from common.prompt_template_engine import build_prompt
from common.instruct_llm import ainstruct_llm, set_async_instructor_client
from common.io import dump_list_of_objects_to_csv
from common.async_helper import execute_tasks_with_manual_pbar


named_entity_knowledge_store_csv_path = settings.input_path / "named_entity_knowledge_store.csv"


LLM_TEMPERATURE = 0.0


class NamedEntityKnowledgeStoreQueryStringResponse(BaseModel):
    query_string: str = Field(
        ..., title="The query to search the named entity knowledge store. The string is comma-separated."
    )
    reason: str = Field(..., title="The reason for why the query string was chosen. Limit to 2 sentences.")


class NamedEntityFilteringResponse(BaseModel):
    reason: str = Field(..., title="The reason for why the named entity is relevant or not. Limit to 2 sentences.")
    is_relevant: bool = Field(..., title="Whether the named entity is relevant to the decision and decision option")


class NamedEntityDirectEffect(BaseModel):
    title: str = Field(..., title="A simple overview of a direct effect of a decision option (5-10 words)")
    description: str = Field(..., title="A detailed description of a direct effect of a decision option")
    category: str = Field(..., title="The category of a direct effect of a decision option")
    impact: str = Field(
        ...,
        title="The business impact of a direct effect of a decision option. This must be either low, medium, or high.",
    )
    source_named_entity_indices: list[int] = Field(
        ..., title="The indices of the named entities that are related to this direct effect"
    )
    specificity_reasoning: SkipJsonSchema[str] = Field(
        "", title="The reasoning for the specificity refinements of the direct effect"
    )


class NamedEntityDirectEffectsResponse(BaseModel):
    direct_effects: list[NamedEntityDirectEffect] = Field(..., title="List of direct effects of a decision option")


class NamedEntityStatusQuoDirectEffectFilteringResponse(BaseModel):
    reason: str = Field(
        ...,
        title="The reason for why or why not the direct effect is a status quo direct effect. Limit to 2 sentences.",
    )
    is_status_quo: bool = Field(..., title="Whether or not the direct effect is a status quo direct effect")


class NamedEntityDirectEffectSpecificityRefinementResponse(BaseModel):
    description: str = Field(
        ..., title="The refined description of a direct effect of a decision option for specificity"
    )
    specificity_reasoning: str = Field(..., title="The reasoning for refining the direct effect for specificity")


class NamedEntityKnowledgeStorePrintableResults(BaseModel):
    decision_title: str = Field(..., title="The title of the decision")
    decision_goals: str = Field(..., title="The goals of the decision")
    criteria: str = Field(..., title="List of criteria of the decision")
    option_title: str = Field(..., title="The title of the decision option")
    option_description: str = Field(..., title="The description of the decision option")
    knowledge_store_query_string: str = Field(
        ..., title="The query string used to query the named entity knowledge store"
    )
    knowledge_store_query_string_reason: str = Field(..., title="The reason for why the query string was chosen")
    knowledge_store_query_indices: str = Field(
        ..., title="The indices of the resulting entities in the knowledge store from the knowledge store query"
    )
    knowledge_store_query_data: str = Field(
        ..., title="The resulting data of the entities in the knowledge store from the knwoledge store query"
    )
    knowledge_store_filtering_results: str = Field(
        ..., title="The filtering results of the named entities from the knowledge store query"
    )
    num_filtered_knowledge_store_query_results: int = Field(
        ..., title="The number of filtered entities from the knowledge store query"
    )
    filtered_knowledge_store_query_indices: str = Field(
        ..., title="The indices of the resulting entities in the knowledge store after filtering"
    )
    filtered_knowledge_store_query_data: str = Field(
        ..., title="The resulting data of the entities in the knowledge store after filtering"
    )
    initial_direct_effects: str = Field(
        ..., title="The (initial) direct effects of the decision option, before any refinement"
    )
    status_quo_filtering_results: str = Field(
        ..., title="The filtering results of the analysis of status quo direct effects"
    )
    status_quo_filtered_direct_effects: str = Field(
        ..., title="The status quo filtered direct effects of the decision option"
    )
    specificity_refined_direct_effects: str = Field(
        ..., title="The refined direct effects of the decision option for specificity"
    )


async def get_query_knowledge_store_query_strings_for_decision_options(
    decision: Decision,
) -> list[NamedEntityKnowledgeStoreQueryStringResponse]:
    """
    Get query strings for decision options to query the named entity knowledge store.
    Give an LLM details of of the decision to come up with a query string for each decision option.
    """
    print("Getting knowledge store query strings for decision options...")

    system_prompt = build_prompt("named_entity_knowledge_store/knowledge_store_query_system.txt.jinja")
    user_prompts = [
        build_prompt(
            "named_entity_knowledge_store/knowledge_store_query_user.txt.jinja", decision=decision, option=option
        )
        for option in decision.options
    ]

    tasks = [
        ainstruct_llm(
            system_prompt,
            user_prompt,
            response_model=NamedEntityKnowledgeStoreQueryStringResponse,
            llm_model=settings.llm_model,
            temperature=LLM_TEMPERATURE,
        )
        for user_prompt in user_prompts
    ]

    query_strings = await execute_tasks_with_manual_pbar(tasks)

    return query_strings


async def get_knowledge_store_query_results(
    knowledge_store: NamedEntityKnowledgeStore,
    query_strings: list[NamedEntityKnowledgeStoreQueryStringResponse],
    top_k_similar_entities: int,
) -> list[list[NamedEntityQueryResult]]:
    """
    For each query string, query the named entity knowledge store and get the list of similar entities.
    To query the named entity knowledge store, we use the faiss index.
    """
    print("Querying named entity knowledge store with query strings...")

    tasks = [
        knowledge_store.asearch_similar_entities(query_string.query_string, num_entities=top_k_similar_entities)
        for query_string in query_strings
    ]

    results = await execute_tasks_with_manual_pbar(tasks)

    return results


async def get_filtered_named_entity_query_results_using_llm(
    decision: Decision,
    knowledge_store_query_strings: list[NamedEntityKnowledgeStoreQueryStringResponse],
    named_entity_query_results: list[list[NamedEntityQueryResult]],
) -> list[list[NamedEntityFilteringResponse]]:
    """
    Filter named entity query results using LLM.

    We get the named entity query results from the named entity knowledge store using a FAISS index, which gives us the top k similar entities.
    Thus, some of the entities may not be relevant to the decision (since we get the top k similar entities).

    This function asks an LLM whether each entity is relevant to the decision and each decision option.
    """
    print("Getting named entity filtering results using LLM...")

    lists_of_filtering_results = []

    for index, option in enumerate(decision.options):
        print(f"Getting named entity filtering results for option {index+1}...")

        named_entities = named_entity_query_results[index]
        named_entities_query_string = knowledge_store_query_strings[index].query_string

        system_prompt = build_prompt(
            "named_entity_knowledge_store/filter_named_entities_system.txt.jinja",
            decision=decision,
            option=option,
            named_entities_query_string=named_entities_query_string,
        )
        user_prompts = [
            build_prompt(
                "named_entity_knowledge_store/filter_named_entities_user.txt.jinja",
                named_entity=named_entity,
            )
            for named_entity in named_entities
        ]

        tasks = [
            ainstruct_llm(
                system_prompt,
                user_prompt,
                response_model=NamedEntityFilteringResponse,
                llm_model=settings.llm_model,
                temperature=LLM_TEMPERATURE,
            )
            for user_prompt in user_prompts
        ]

        filtering_results = await execute_tasks_with_manual_pbar(tasks)
        lists_of_filtering_results.append(filtering_results)

    return lists_of_filtering_results


def get_filtered_named_entities(
    named_entity_query_results: list[list[NamedEntityQueryResult]],
    filtering_results: list[list[NamedEntityFilteringResponse]],
) -> list[list[NamedEntityQueryResult]]:
    """
    Get the filtered named entities based on the filtering results.
    That is, for each decision option, filter out the named entities that are not relevant, and return the relevant named entities.
    """
    print("Getting filtered named entities...")

    filtered_named_entities = []

    for index, (named_entities_for_option, filtering_results_for_option) in enumerate(
        zip(named_entity_query_results, filtering_results)
    ):
        print(f"Filtering named entities for option {index+1}...")
        print(f"Number of named entities before filtering: {len(named_entities_for_option)}")

        filtered_entities = [
            named_entity
            for named_entity, filtering_result in zip(named_entities_for_option, filtering_results_for_option)
            if filtering_result.is_relevant
        ]

        print(f"Number of named entities after filtering: {len(filtered_entities)}")

        filtered_named_entities.append(filtered_entities)

    return filtered_named_entities


async def get_initial_direct_effects_using_llm(
    decision: Decision,
    named_entity_query_results: list[list[NamedEntityQueryResult]],
) -> list[NamedEntityDirectEffectsResponse]:
    """
    Get initial direct effects of decision options using LLM.
    Give an LLM details of the decision and the resulting entities from the named entity knowledge store to come up with the direct effects.
    These direct effects will be refined in future steps.
    """
    print("Getting initial direct effects using LLM...")

    system_prompt = build_prompt("named_entity_knowledge_store/direct_effects_system.txt.jinja")
    user_prompts = [
        build_prompt(
            "named_entity_knowledge_store/direct_effects_user.txt.jinja",
            decision=decision,
            option=option,
            named_entities=named_entities,
        )
        for option, named_entities in zip(decision.options, named_entity_query_results)
    ]

    tasks = [
        ainstruct_llm(
            system_prompt,
            user_prompt,
            response_model=NamedEntityDirectEffectsResponse,
            llm_model=settings.llm_model,
            temperature=LLM_TEMPERATURE,
        )
        for user_prompt in user_prompts
    ]

    direct_effects = await execute_tasks_with_manual_pbar(tasks)

    for index, effects in enumerate(direct_effects):
        print(f"Number of direct effects for option {index+1}: {len(effects.direct_effects)}")

    return direct_effects


async def get_status_quo_filtered_results_using_llm(
    initial_direct_effects: list[NamedEntityDirectEffectsResponse], decision: Decision
) -> list[list[NamedEntityStatusQuoDirectEffectFilteringResponse]]:
    """
    This step filters out status quo direct effects.
    Status quo direct effects are direct effects that are already known and do not provide any new information.
    """
    print("Filtering out status quo direct effects...")

    filtering_results_for_options = []

    for index, (option, effects_response_for_option) in enumerate(zip(decision.options, initial_direct_effects)):
        print(f"Filtering out status quo direct effects for option {index+1}...")
        effects_for_option = effects_response_for_option.direct_effects

        system_prompt = build_prompt(
            "named_entity_knowledge_store/filter_out_status_quo_direct_effects_system.txt.jinja"
        )

        user_prompts = [
            build_prompt(
                "named_entity_knowledge_store/filter_out_status_quo_direct_effects_user.txt.jinja",
                decision=decision,
                option=option,
                direct_effect=effect,
            )
            for effect in effects_for_option
        ]

        tasks = [
            ainstruct_llm(
                system_prompt,
                user_prompt,
                response_model=NamedEntityStatusQuoDirectEffectFilteringResponse,
                llm_model=settings.llm_model,
                temperature=LLM_TEMPERATURE,
            )
            for user_prompt in user_prompts
        ]

        status_quo_filtering_results = await execute_tasks_with_manual_pbar(tasks)

        filtering_results_for_options.append(status_quo_filtering_results)

    return filtering_results_for_options


def get_status_quo_filtered_direct_effects(
    initial_direct_effects: list[NamedEntityDirectEffectsResponse],
    status_quo_filtering_results: list[list[NamedEntityStatusQuoDirectEffectFilteringResponse]],
) -> list[NamedEntityDirectEffectsResponse]:
    print("Filtering out status quo direct effects...")

    status_quo_filtered_direct_effects = []

    for index, (effects_response_for_option, filtering_results_for_option) in enumerate(
        zip(initial_direct_effects, status_quo_filtering_results)
    ):
        print(f"Filtering out status quo direct effects for option {index+1}...")
        effects_for_option = effects_response_for_option.direct_effects

        filtered_effects_for_option = [
            effect
            for effect, filtering_result in zip(effects_for_option, filtering_results_for_option)
            if not filtering_result.is_status_quo
        ]

        status_quo_filtered_direct_effects.append(
            NamedEntityDirectEffectsResponse(direct_effects=filtered_effects_for_option)
        )

    for index, effects in enumerate(status_quo_filtered_direct_effects):
        print(f"Number of status quo filtered direct effects for option {index+1}: {len(effects.direct_effects)}")

    return status_quo_filtered_direct_effects


async def refine_direct_effects_for_specificity_using_llm(
    initial_direct_effects: list[NamedEntityDirectEffectsResponse],
    decision: Decision,
    named_entity_query_results: list[list[NamedEntityQueryResult]],
) -> list[NamedEntityDirectEffectsResponse]:
    """
    This step refines the initial direct effects to be more specific using LLM.
    """
    print("Refining direct effects for specificity using LLM...")

    lists_of_refined_direct_effects = []

    # Loop over each option and its associated objects
    for index, (option, effects_response_for_option, named_entities_for_option) in enumerate(
        zip(decision.options, initial_direct_effects, named_entity_query_results)
    ):
        print(f"Refining direct effects for specificity for option {index+1}...")
        effects_for_option = effects_response_for_option.direct_effects

        system_prompts = [
            build_prompt(
                "named_entity_knowledge_store/refine_direct_effects_specificity_system.txt.jinja",
                decision=decision,
                option=option,
                named_entities=[
                    named_entity
                    for named_entity in named_entities_for_option
                    if named_entity.index in effect.source_named_entity_indices
                ],
            )
            for effect in effects_for_option
        ]

        user_prompts = [
            build_prompt(
                "named_entity_knowledge_store/refine_direct_effects_specificity_user.txt.jinja",
                direct_effect=effect,
            )
            for effect in effects_for_option
        ]

        tasks = [
            ainstruct_llm(
                system_prompt,
                user_prompt,
                response_model=NamedEntityDirectEffectSpecificityRefinementResponse,
                llm_model=settings.llm_model,
                temperature=LLM_TEMPERATURE,
            )
            for user_prompt, system_prompt in zip(user_prompts, system_prompts)
        ]

        refined_direct_effects = await execute_tasks_with_manual_pbar(tasks)

        refined_effects_for_option = [
            NamedEntityDirectEffect(
                title=effect.title,
                description=refined_effect.description,
                specificity_reasoning=refined_effect.specificity_reasoning,
                category=effect.category,
                impact=effect.impact,
                source_named_entity_indices=effect.source_named_entity_indices,
            )
            for effect, refined_effect in zip(effects_for_option, refined_direct_effects)
        ]

        lists_of_refined_direct_effects.append(
            NamedEntityDirectEffectsResponse(direct_effects=refined_effects_for_option)
        )

    for index, effects in enumerate(lists_of_refined_direct_effects):
        print(f"Number of specificity refined direct effects for option {index+1}: {len(effects.direct_effects)}")

    return lists_of_refined_direct_effects


def dump_results_to_csv(
    decision_id: int,
    decision: Decision,
    query_strings: list[NamedEntityKnowledgeStoreQueryStringResponse],
    named_entity_query_results: list[list[NamedEntityQueryResult]],
    filtering_results: list[list[NamedEntityFilteringResponse]],
    filtered_named_entities: list[list[NamedEntityQueryResult]],
    initial_direct_effects: list[NamedEntityDirectEffectsResponse],
    status_quo_filtering_results: list[list[NamedEntityStatusQuoDirectEffectFilteringResponse]],
    status_quo_filtered_direct_effects: list[NamedEntityDirectEffectsResponse],
    specificity_refined_direct_effects: list[NamedEntityDirectEffectsResponse],
):
    """
    Dump all resulting data to a csv file.
    """
    print("Dumping results to csv...")

    results = [
        NamedEntityKnowledgeStorePrintableResults(
            decision_title=decision.title,
            decision_goals=decision.goals,
            criteria="\n".join(
                [
                    f"Criteria {index+1}:\nTitle: {c.title}\nDescription: {c.description}\n"
                    for index, c in enumerate(decision.criteria)
                ]
            ),
            option_title=option.title,
            option_description=option.description,
            knowledge_store_query_string=query_string.query_string,
            knowledge_store_query_string_reason=query_string.reason,
            knowledge_store_query_indices=str([result.index for result in query_results]),
            knowledge_store_query_data="\n".join(
                [
                    f"Entity index: {result.index}\nEntity Name: {result.entity_name}\nCategory: {result.category}\nDescription: {result.description}\nDistance: {result.distance}\n"
                    for result in query_results
                ]
            ),
            knowledge_store_filtering_results="\n".join(
                [
                    f"Entity index: {qr.index}\nEntity Name: {qr.entity_name}\nIs relevant: {fr.is_relevant}\nReason: {fr.reason}\n"
                    for fr, qr in zip(filtering_result, query_results)
                ]
            ),
            num_filtered_knowledge_store_query_results=len(filtered_entities),
            filtered_knowledge_store_query_indices=str([result.index for result in filtered_entities]),
            filtered_knowledge_store_query_data="\n".join(
                [
                    f"Entity index: {result.index}\nEntity Name: {result.entity_name}\nCategory: {result.category}\nDescription: {result.description}\nDistance: {result.distance}\n"
                    for result in filtered_entities
                ]
            ),
            initial_direct_effects="\n".join(
                [
                    f"Title: {effect.title}\nEffect: {effect.description}\nCategory: {effect.category}\nBusiness impact: {effect.impact}\nSource named entity indices: {effect.source_named_entity_indices}\n"
                    for effect in initial_effects.direct_effects
                ]
            ),
            status_quo_filtering_results="\n".join(
                [
                    f"Effect Title: {effect.title}\nEffect Description: {effect.description}\nIs status quo?: {filtering_result.is_status_quo}\nReason: {filtering_result.reason}\n"
                    for filtering_result, effect in zip(status_quo_results, initial_effects.direct_effects)
                ]
            ),
            status_quo_filtered_direct_effects="\n".join(
                [
                    f"Title: {effect.title}\nEffect: {effect.description}\nCategory: {effect.category}\nBusiness impact: {effect.impact}\nSource named entity indices: {effect.source_named_entity_indices}\n"
                    for effect in status_quo_filtered_effects.direct_effects
                ]
            ),
            specificity_refined_direct_effects="\n".join(
                [
                    f"Title: {effect.title}\nEffect: {effect.description}\nSpecificity Reasoning: {effect.specificity_reasoning}\nCategory: {effect.category}\nBusiness impact: {effect.impact}\nSource named entity indices: {effect.source_named_entity_indices}\n"
                    for effect in specificity_refined_effects.direct_effects
                ]
            ),
        )
        for option, query_string, query_results, filtering_result, filtered_entities, initial_effects, status_quo_results, status_quo_filtered_effects, specificity_refined_effects in zip(
            decision.options,
            query_strings,
            named_entity_query_results,
            filtering_results,
            filtered_named_entities,
            initial_direct_effects,
            status_quo_filtering_results,
            status_quo_filtered_direct_effects,
            specificity_refined_direct_effects,
        )
    ]

    results_csv_path = (
        settings.output_path / f"named_entity_knowledge_store_direct_effects_results_decision_{decision_id}.csv"
    )

    dump_list_of_objects_to_csv(results, results_csv_path)


async def named_entity_knowledge_store_direct_effects_of_decision_options():
    """
    This function runs the experiment to find direct effects of decision options using a named entity knowledge store.

    1. Initialize a named entity knowledge store. This is just a dummy knowledge store, thus, the logic is not meant to be "production" worthy.
    2. Load a decision.
    3. Ask an LLM to come up a query string for each decision option to query the named entity knowledge store.
    4. Query the named entity knowledge store with the query strings.
    5. Double check relevance from knowledge store using LLM, i.e. filter out irrelevant entities.
    6. Get direct effects of decision options using LLM.
    7. Dump all results to a csv file.
    """
    print("Running named entity knowledge store direct effects of decision options experiment...")

    # Set up async instructor client
    set_async_instructor_client(llm_model=settings.llm_model, api_key=settings.anthropic_api_key)

    # Initialize named entity knowledge store
    knowledge_store: NamedEntityKnowledgeStore = await initialize_named_entity_knowledge_store(
        input_file_path=named_entity_knowledge_store_csv_path,
        load_faiss_index_from_cache=True,
    )

    # Load a decision
    decision_id = 1
    decision: Decision = get_decision_data(decision_id)

    # Get knowledge store query string for decision options
    knowledge_store_query_strings: list[
        NamedEntityKnowledgeStoreQueryStringResponse
    ] = await get_query_knowledge_store_query_strings_for_decision_options(decision)

    # Query named entity knowledge store with query strings
    top_k_similar_entities = 10
    named_entity_query_results: list[list[NamedEntityQueryResult]] = await get_knowledge_store_query_results(
        knowledge_store, knowledge_store_query_strings, top_k_similar_entities
    )

    # Double check relevance from knowledge store using LLM
    filtering_results: list[
        list[NamedEntityFilteringResponse]
    ] = await get_filtered_named_entity_query_results_using_llm(
        decision, knowledge_store_query_strings, named_entity_query_results
    )
    filtered_named_entities: list[list[NamedEntityQueryResult]] = get_filtered_named_entities(
        named_entity_query_results, filtering_results
    )

    # Get initial direct effects using LLM
    initial_direct_effects: list[NamedEntityDirectEffectsResponse] = await get_initial_direct_effects_using_llm(
        decision, filtered_named_entities
    )

    # Filter out status quo direct effects
    status_quo_filtering_results: list[
        list[NamedEntityStatusQuoDirectEffectFilteringResponse]
    ] = await get_status_quo_filtered_results_using_llm(initial_direct_effects, decision)
    status_quo_filtered_direct_effects: list[NamedEntityDirectEffectsResponse] = (
        get_status_quo_filtered_direct_effects(initial_direct_effects, status_quo_filtering_results)
    )

    # Refine initial direct effects to be more specific using LLM
    specificity_refined_direct_effects: list[
        NamedEntityDirectEffectsResponse
    ] = await refine_direct_effects_for_specificity_using_llm(
        status_quo_filtered_direct_effects, decision, filtered_named_entities
    )

    # Dump results to csv
    dump_results_to_csv(
        decision_id,
        decision,
        knowledge_store_query_strings,
        named_entity_query_results,
        filtering_results,
        filtered_named_entities,
        initial_direct_effects,
        status_quo_filtering_results,
        status_quo_filtered_direct_effects,
        specificity_refined_direct_effects,
    )
