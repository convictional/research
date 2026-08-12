import panel as pn
import datetime
from utils.parallel_processing import process_prompt_parallel
from utils.data_storage import save_results_to_csv, save_elo_ratings, load_elo_ratings
from utils.elo_calculation import update_elo_ratings

# Create and serve the Panel application
pn.extension(
    raw_css=[
        """
    .model-response {
        padding: 10px;
        margin: 5px;
        background: #f0f0f0;
    }
    """
    ]
)

# Create the widgets
prompt_input = pn.widgets.TextInput(name="Prompt", placeholder="Enter your prompt here...")
model_select = pn.widgets.MultiChoice(
    name="Models", options=["Simple Model", "GPT4 Wrapper", "Complex Chain"], value=["Simple Model", "GPT4 Wrapper"]
)
process_button = pn.widgets.Button(name="Process Prompt", button_type="primary")
submit_feedback_button = pn.widgets.Button(name="Submit Feedback", button_type="primary")

# Create placeholders for model responses, rankings, and feedback
model_responses = pn.Column()
model_rankings = pn.pane.Markdown()
model_feedback = pn.pane.Markdown()

# Create dictionaries to store ranking and feedback widgets
ranking_widgets = {}
feedback_widgets = {}

# Load ELO ratings from CSV file or initialize with default values
elo_ratings = load_elo_ratings("./temp/elo_ratings.csv") or {model: 1000 for model in model_select.options}

# Create a placeholder for displaying current ELO ratings
current_elo_ratings = pn.pane.Markdown()


def update_ui():
    # Update ranking widgets based on the selected models
    for model in model_select.value:
        if model not in ranking_widgets:
            ranking_widgets[model] = pn.widgets.Select(
                name=f"{model} Ranking", options=list(range(1, len(model_select.value) + 1)), value=1
            )
        else:
            ranking_widgets[model].options = list(range(1, len(model_select.value) + 1))
            ranking_widgets[model].value = 1

    # Remove ranking widgets for unselected models
    for model in list(ranking_widgets.keys()):
        if model not in model_select.value:
            del ranking_widgets[model]

    # Update feedback widgets based on the selected models
    for model in model_select.value:
        if model not in feedback_widgets:
            feedback_widgets[model] = pn.widgets.TextInput(name=f"{model} Feedback", placeholder="Enter feedback...")

    # Remove feedback widgets for unselected models
    for model in list(feedback_widgets.keys()):
        if model not in model_select.value:
            del feedback_widgets[model]

    # Update the displayed current ELO ratings
    elo_ratings_text = "### Current ELO Ratings\n"
    for model in model_select.value:
        if model in elo_ratings:
            elo_ratings_text += f"- **{model}**: {round(elo_ratings[model])}\n"
        else:
            elo_ratings_text += f"- **{model}**: 1000\n"
            elo_ratings[model] = 1000
    current_elo_ratings.object = elo_ratings_text

    # Update the layout with the new widgets
    layout.objects = [
        pn.Row(prompt_input, model_select, process_button),
        current_elo_ratings,
        model_responses,
        pn.Row(*[ranking_widgets[model] for model in model_select.value]),
        pn.Row(*[feedback_widgets[model] for model in model_select.value]),
        submit_feedback_button,
    ]


def process_prompt_click(event):
    prompt = prompt_input.value
    selected_models = model_select.value
    if prompt and selected_models:
        results = process_prompt_parallel(prompt, selected_models)
        # Update model responses
        model_responses.objects = [
            pn.Column(
                pn.pane.Markdown(f"### {model}"),
                pn.pane.Markdown(response["response"]),
                css_classes=["model-response"],
            )
            for model, response in results.items()
        ]
        model_responses.objects = [pn.Row(*model_responses.objects)]

        # Update model rankings
        rankings = {model: ranking_widgets[model].value for model in selected_models}
        rankings_text = "### Model Rankings\n"
        for model, ranking in rankings.items():
            rankings_text += f"- **{model}**: {ranking}\n"
        model_rankings.object = rankings_text

        # Update model feedback
        feedback = {model: feedback_widgets[model].value for model in selected_models}
        feedback_text = "### Model Feedback\n"
        for model, feedback_text in feedback.items():
            feedback_text += f"- **{model}**: {feedback_text}\n"
        model_feedback.object = feedback_text

        return results, rankings, feedback


def submit_feedback_click(event):
    results, rankings, feedback = process_prompt_click(event)
    save_results_to_csv(results=results, rankings=rankings, feedback=feedback, file_path="./temp/results.csv")

    # Update the ELO ratings based on user rankings
    global elo_ratings
    elo_ratings = update_elo_ratings(rankings, elo_ratings)

    # Save the updated ELO ratings to CSV file with timestamp
    save_elo_ratings(elo_ratings, "./temp/elo_ratings.csv")

    # Update the displayed current ELO ratings
    update_ui()

    # Clear the feedback widgets
    for model in model_select.value:
        feedback_widgets[model].value = ""


# Define the initial layout
layout = pn.Column(pn.Row(prompt_input, model_select, process_button), current_elo_ratings, model_responses)

# Attach event handlers
model_select.param.watch(lambda event: update_ui(), "value")
process_button.on_click(process_prompt_click)
submit_feedback_button.on_click(submit_feedback_click)

# Update the UI initially
update_ui()

pn.serve(layout)
