# Goal extraction research prototype

**Authors:** Matt Chequers, Adam McCabe

This experiment researches extracting company goals from content.

Furthermore, we want to distinguish between stated and unstated goals, and contrast the two to find alignments and tensions between what the company is working on and the company's formally stated goals.

The task for the research can be found [internal app task, not public].

# Sub-experiments

## Initial gut check

We started with an initial gut check/spike to see what we could get with current functionality in the app, i.e. research threads.

This is summarized in the app task.

Google doc with the results can be found [internal doc, not public].

For complete details of the methods, analysis, results, etc, see the above link. For convenience, a brief overview of the intitial gut check is given below.

### Overview of initial gut check

- Use Research Threads in the app
- Ask about unstated goals for a specific entity within the organization, e.g. a specific department
- Ask to contrast the unstated goals against the stated goals of the organization
- This is a quick way to see if revealed and unstated goals can be extracted and how well that extraction works
- If things look good here, it is a positive indicator for digging deeper and maybe implementing more customized solutions, e.g. coding experiments


## In-app experimental implementation (i.e. the goal extraction prototyping)

With the gut check confirmed, we then implemented an experiment in the app to prototype goal extraction and contrast stated and unstated goals.

This is summarized in the app task.

WIP PR for this work is [internal PR, not public]. Since this code work is branched off of the app, this PR is not to be merged, and so we will just reference the PR for the coding involved in the prototype.

Google doc for the write up of this work is [internal doc, not public].

For complete details of the methods, analysis, results, etc, see the links above. For convenience, a brief overview of the research prototyping is given below.

### Overview of research prototype work

Multi-step process:
- Step 1, researching content using a research thread
    - Kick off a research thread asking about stated and unstated goals
    - Explicitly provide the current stated goals as context in the user query for the LLM to anchor to for stated goals
- Step 2, mine activity event data for stated and unstated goals
    - Process activity event data for various resources in the app:
        - Tasks: related events, metadata, and comment threads
        - Decisions: all decision metadata
        - Meetings: meeting metadata, including the summary (but not the transcript)
        - Discussions: metadata, original discussion comment, and threads of comments in the discussion
    - Use the processed activity event data in the input prompts
    - Explicitly provide the current stated goals as context in the user query for the LLM to anchor to for stated goals
- Step 3, extract and consolidate stated goals and context from steps 1 and 2
- Step 4, extract and consolidate unstated goals and context from steps 1 and 2
    - Also, the LLM is asked to deduplicate similar unstated goals into a single goal
- Step 5, verbose goals comparison report
    - Construct a verbose report about the alignments and misalignments/tensions between stated and unstated goals
- Step 6, goals comparison summary
    - Summarize the output of step 5 into a more readable and digestible summary format to be consumed by the end user

Run this process for 3 scenarios:
- Stock Convictional seed data (2 stated goals)
- Organization with no stated goals
    - Convictional seed data, but remove any stated goals
- Organization with many stated goals
    - Synthesized 20 “goals” from Convictional content using a Research Thread in the app
    - Inject these 20 goals into the database
    - 20 goals seems like it might be “many” for our ICP
