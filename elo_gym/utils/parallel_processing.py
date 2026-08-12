import concurrent.futures
from models.simple_model import SimpleModel
# Import other model modules as needed


def process_prompt_parallel(prompt, selected_models):
    models = {
        "Simple Model": SimpleModel(),
        "GPT4 Wrapper": SimpleModel(),
        "Complex Chain": SimpleModel(),
        # Add other model instances as needed
    }

    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(models[model].process_prompt, prompt): model for model in selected_models}
        for future in concurrent.futures.as_completed(futures):
            model = futures[future]
            results[model] = future.result()

    return results
