# Overview

This experiment is for tailored web search results for a given user and org.

This is meant to showcase tailoring web search results, and is meant to compliment #2140 (internal PR, not public) for inspiration of applying web search in the app.

# Prerequisites

This routine makes use of the Google Custom Search API.
Steps to set up:
1. Enable and set up API key:
    - Go to https://console.cloud.google.com/
    - In Google Cloud Console, go to "APIs & Services" > "Library"
    - Search for "Custom Search API" and enable it
    - Go to "APIs & Services" > "Credentials"
    - Click "Create Credentials" > "API Key"
    - Copy the generated API key
    - Optional but recommended:
        - Click on the newly created API key
        - Under "API restrictions", choose "Restrict key"
        - Select only "Custom Search API"
        - Save
2. Create custom search engine
    - Go to https://programmablesearchengine.google.com/
    - Click "Create a search engine"
    - In the setup options:
        - Choose "Search the entire web" to search all websites
        - Make sure "Search the entire web" is enabled in the settings
    - Get your Search Engine ID (cx) - you'll need this

# Results

GSheet for initial results is [internal spreadsheet, not public].

Known issues:
- Extensive quality check of results has not been done
    - The motivation for this bet was to spark conversation and inspire any ideas related to tailoring Google search results
- Sometimes a valid Google search query string is not created by the LLM (i.e. missing OR statements, etc)
    - Could implement a recursive loop to test the query, and feed it back to the LLM to correct if it isn't valid
- Even though there is a filter for results "after" some date, some results don't have a date attached to them in the response
    - So, the query might be returning "stale" results from before the starting date in the search query
- LLM summaries for some search results are not able to be created
    - This can happen if the link hits a paywall or similar blocks to getting the webpage content
    - Could implement some logic to account for this and ensure that top-N search results are summarized every time
- Could include goals in the context to motivate the search query and summarization
