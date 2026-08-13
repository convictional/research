# Token Probabilities

**Author:** Adam McCabe

Along with our tokens, we can ask OpenAI to return the log-probabilities of the individual tokens. By setting `temperature` to 0, we can force the language model to always select the most likely token. Linearlizing these log probabilities, we get an understandable probability distribution over the tokens which we can use to infer confidence by looking at the 'steepness' of the drop-off in probability.

Running the playground, starting from the `decide` directory, run the following commands:
```bash
make clean
make setup
poetry shell
doppler setup
doppler run -- panel serve experiments/token_log_probs/token_prob_visualization.py 
```
You should see a console return that includes something like, `Bokeh app running at: http://localhost:5006/token_prob_visualization`. Open this link in your browser to see the visualization.
