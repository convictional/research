# Overview

This "experiment" is for R&D project tax credit work that we have done.

The main idea is to use knowledge data (e.g. GitHub) to extract qualified R&D projects from, for the purposes of R&D tax credits.

This type of work was started in 2024 for 2023 US tax season. The GitHub issue for that work is [internal GitHub issue, not public] and the code is [internal PR, not public] (Note, that code was never merged).

The purpose of this directory is to hold the code and logic for extracting qualified R&D projects for a given tax credit use-case. The code and logic can be refined over time as needed.

Also note, the code in this directory is designed to be self-contained, i.e. no calls to common functions or modules. This is to have a record of all of the code that makes the routine actually work. Since we run this not that frequently, if breaking changes to those common function calls happen, it will add a lot more time to debug any issues. We can revisit this design decision in the future if we feel like we need to.

# Visual flow of project extraction flow

![Alt text](figures/project_extraction_flow.png)

# Other links

- GitHub issue for 2024 tax season Canadian SR&ED credits is [internal GitHub issue, not public].
