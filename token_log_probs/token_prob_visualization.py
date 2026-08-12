import re

import numpy as np
import panel as pn
import tastymap as tm
from openai import AsyncOpenAI

# from context import guru_vectorized_content, db_vectorized_context

pn.extension()
pn.config.global_css = [
    """
                         .custom-background {
                         background-color: #1e1e1e;
                         white-space: pre-wrap;
                         word-wrap: break-word;
                         }
                         """
]


COLOR_MAP = "RdBu_r"
NUM_COLORS = 4

SYSTEM_PROMPT = """
Assume the identities of Stefano Puntoni and Bart De Langhe as it relates to their decision driven analytics work. 

Your job is to help the User evaluate and think through a strategic business decision. The most important aspect
of your job is to:
- identify alternatives - ask the user for input on your thinking if needed
- given an alternative option, what is the data and/or context the user would need to gather in order to properly
answer the question
""".strip()


def color_by_logprob(text, log_prob):
    linear_prob = np.round(np.exp(log_prob) * 100, 2)
    color_index = int(linear_prob // (100 / (len(colors) - 1)))

    return f'<span style="color: {colors[color_index]}">{text}</span>'


def custom_serializer(content):
    pattern = r"<span.*?>(.*?)</span>"
    matches = re.findall(pattern, content)
    if not matches:
        return content
    return matches[0]


async def respond_to_input(contents: str, user: str, instance: pn.chat.ChatInterface):
    """
    This function responds to a user's input by calling the LLM and returning the response
    along with logprobs, and tokens colored by their linear probability.

    We also call our structure_decision_function to structure the response into a decision
    with experiments on how we can then tie the log-probs returned for tokens in the raw
    response to the tokens in the structured response - as we cannot return logprobs with
    a Tool call in OpenAI.
    """
    # add system prompt
    if system_input.value:
        system_message = {"role": "system", "content": system_input.value}
        messages = [system_message]
    else:
        messages = []

    # gather messages for memory
    if memory_toggle.value:
        messages += instance.serialize(custom_serializer=custom_serializer)
    else:
        messages.append({"role": "user", "content": contents})

    # call API
    response = await aclient.chat.completions.create(
        model=model_selector.value,
        messages=messages,
        stream=False,
        logprobs=True,
        temperature=temperature_input.value,
        max_tokens=max_tokens_input.value,
        seed=seed_input.value,
    )

    # process full response
    message = ""
    choice = response.choices[0]
    content = choice.message.content

    log_probs = choice.logprobs.content
    print(response)
    token_log_probs = []
    for token, log_prob in zip(content.split(), log_probs):
        message += color_by_logprob(token, log_prob.logprob)
        message += " "
        token_log_probs.append(log_prob.logprob)
    perplexity = np.exp(-np.mean(token_log_probs))

    message = f"<span><medium>Perplexity: {perplexity:.2f}</medium><br><small>"
    message += f"(Note, lower is better; 1 is the theoretical best)</small><br><br></span> {message}"
    yield message.strip()


# Generate HTML for the color scale with probabilities aligned
def generate_color_scale_with_probabilities(colors):
    color_scale_html = '<div style="display: flex;">'

    # Append the color blocks
    for color in colors:
        color_scale_html += f'<div style="flex: 1; background-color: {color}; height: 20px;"></div>'

    color_scale_html += '</div><div style="display: flex; justify-content: space-between; padding: 0 4px;">'

    # Append the probability labels
    for i in range(len(colors)):
        prob = (100 / (len(colors) - 1)) * i
        if i == 0 or i == len(colors) - 1:  # Add padding for the first and last label
            align = "left" if i == 0 else "right"
            color_scale_html += f'<div style="width: {100/len(colors)}%; text-align: {align};">{prob:.0f}%</div>'
        else:
            color_scale_html += f'<div style="width: {100/len(colors)}%; text-align: center;">{prob:.0f}%</div>'

    color_scale_html += "</div>"

    return color_scale_html


tmap = tm.cook_tmap(
    colors_or_cmap=COLOR_MAP,
    num_colors=NUM_COLORS,
    bad="k",
    under="k",
    over="k",
)

colors = tmap.to_model("hex")

# Generate the colormap HTML representation
tmap_html = tmap._repr_html_()


# Process the HTML to remove the 'under', 'bad', and 'over' labels
def clean_colormap_html(html):
    # Find and remove the unwanted segments using regex
    html = re.sub(r'<div style="float: left;.*?under</div>', "", html)
    html = re.sub(r'<div style="margin: 0 auto;.*?bad <.*?</div></div>', "", html)
    html = re.sub(r'<div style="float: right;.*?over <.*?</div>', "", html)
    return html


clean_tmap_html = clean_colormap_html(tmap_html)
color_scale_legend = pn.pane.HTML(clean_tmap_html + generate_color_scale_with_probabilities(colors), align="center")

aclient = AsyncOpenAI()

system_input = pn.widgets.TextAreaInput(
    name="System Prompt",
    value=SYSTEM_PROMPT,
    rows=2,
    auto_grow=True,
)
model_selector = pn.widgets.Select(
    name="Model",
    options=["gpt-4-turbo-preview", "gpt-3.5-turbo", "gpt-4-turbo-1106"],
    width=150,
)
temperature_input = pn.widgets.FloatInput(name="Temperature", start=0, end=2, step=0.01, value=0, width=100)
max_tokens_input = pn.widgets.IntInput(name="Max Tokens", start=0, value=2048, width=100)
seed_input = pn.widgets.IntInput(name="Seed", start=0, end=100, value=42, width=100)
memory_toggle = pn.widgets.Toggle(name="Include Memory", value=True, width=100, margin=(22, 5))
chat_interface = pn.chat.ChatInterface(
    callback=respond_to_input,
    callback_user="ChatGPT",
    callback_exception="verbose",
)
main_layout = pn.Column(
    pn.Row(
        system_input,
        model_selector,
        temperature_input,
        max_tokens_input,
        seed_input,
        memory_toggle,
        align="center",
    ),
    color_scale_legend,
    chat_interface,
).servable(title="Token Probabilities")
