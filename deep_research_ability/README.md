# Deep Research Ability

**Author:** Adam McCabe

This experiment has been updated to match our production implementation of Research in Threads and runs over our content seed (which reflects Convictional's content indexed in threads). In particular, it has access to a search tool which mimics our production SearchContentJob allowing it to find content from our `content` table in postgres. This is inspired by the Deep Research features offered by [Google](https://blog.google/products/gemini/google-gemini-deep-research/) and [OpenAI](https://openai.com/index/introducing-deep-research/).

The original POC of the experiment was taken from the typescript implementation, [`dzhng/deep-research`](https://github.com/dzhng/deep-research), as well as an update to the search tool to mimic our production search over our content table.

See an example tree generated and visualized [here](https://drive.google.com/file/d/17ZbamoQjPVxTLOE8IEmcRuOy47I8kz6O/view?usp=drive_link).

## Running a Deep Research Report
To run this experiment you will need to:

1. Ensure you have postgres running locally
2. From the `decide` directory, seed your content database
    - `make install`
    - `make db_reset`
    - `make db_seed`
    - `make db_content_seed`
    - Consult the application Makefile for more info
2. Move to the `experiments` directory, `cd experiments`
3. Install experimental dependecies, `poetry install`
4. You should now be able to run the experiment:
   ```
   make run_experiment ARGS="deep_research_ability --topic 'Your research topic' --breadth 4 --depth 2"
   ```

Once you start the experiment, it will use your specified parameters or prompt you if needed:
- `Research Topic: string` Provide the topic you want the agent to research. Be specific as it will help follow up questions be more pointed and helpful for the LLM.
- `Breadth: int` How wide you want the research to go - this sets the maximum number of new queries that the agent will kick off at each iteration. Note, we leave it up to the LLM to return less queries if it does not need the full breadth. The Agent uses SERP queries (Search Engine Results Page) from search/SEO theory to search content.
- `Depth: int` How deep the agent will research a given query; for each query the LLM generates follow up questions and will follow that thread for `depth` iterations.

## Research Visualization

The experiment now includes features to visualize the research process as a tree structure. When you run a deep research report, the following visualization outputs are automatically generated:

1. **Tree Visualization HTML**: An interactive tree allowing you to visualize the research progress
2. **Tree JSON Structure**: A JSON file with the complete tree data that can be used for further analysis.
3. **CSV Export**: Detailed research data including iterations, queries, results, and learnings.

### Visualizing Past Research

You can visualize previous research runs using the following command:

```
cd experiments
make run_experiment ARGS="deep_research_ability --visualize_csv your_research_data.csv"
```

Where `your_research_data.csv` is the name of a CSV file in the `src/output/` directory. This will generate tree visualizations for existing CSV files and provide analysis without running a new research session.

## Tree Structure Explanation

The research framework creates a tree structure where:

1. The root node is the initial research topic
2. Each depth level represents an iteration of research
3. Each iteration contains multiple research queries
4. As the depth increases, the number of queries should typically decrease (narrowing pattern)

The visualization helps identify whether the research algorithm is working correctly by showing:
- The branching structure of research queries
- How the focus narrows as depth increases
- Relationships between iterations at different depths

See the original implementation [here](https://github.com/dzhng/deep-research).
