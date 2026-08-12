# Overview

This `commmon/` directory is meant to house common helper functions and logic that are useful across multiple experiments.

For example, function definitions for sending LLM requests, pickle file I/O, prompt templating setup, etc, are pieces of logic that is used across multiple experiments. Rather than defining that logic separately in each experiment, one can simply import from the `common` module.

# Usage warnings

- Before calling `ainstruct_llm` from `instruct_llm.py`, the `set_async_instructor_client` function must be called prior.
    - Basically, the async instructor client must be initialized to something sensical before calling an LLM using that client
- To initialize and register prompt templates, execute the `initialize_and_register_prompt_templates` function in `prompt_template_engine.py`.
    - The function takes a path as an argument, and is the path to the prompts directory for a given experiment
- The functions in `embeddings.py` require an async OpenAI client as an argument.
    - Thus, the client needs to be initiaited and provided to the function to make the embeddings-related calls.
