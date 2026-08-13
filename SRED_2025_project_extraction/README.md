# Overview

**Authors:** Matt Chequers, Adam McCabe

This "experiment" is for 2025 SR&ED R&D project tax credit work that we have done.

The main idea is to use knowledge data (e.g. GitHub, app Tasks) to extract qualified R&D projects from, for the purposes of SR&ED R&D tax credits.

This type of work was started in 2024 for 2023 US tax season. The GitHub issue for that work is [internal GitHub issue, not public] and the code is [internal PR, not public] (Note, that code was never merged). The GitHub issue for the 2024 tax year work is [internal GitHub issue, not public] (Note, that code was merged).

The purpose of this directory is to hold the code and logic for extracting qualified R&D projects for a given tax credit use-case. The code and logic can be refined over time as needed.

Also note, the code in this directory is designed to be self-contained, i.e. no calls to common functions or modules. This is to have a record of all of the code that makes the routine actually work. Since we run this not that frequently, if breaking changes to those common function calls happen, it will add a lot more time to debug any issues. We can revisit this design decision in the future if we feel like we need to.

# Updates for 2025 SR&ED project extraction

The process is largely the same as the 2024 project code, except with different/new data sources added.

That is, we shifted from GitHub for logging our issues in early 2025 to the app Tasks feature. Thus, for 2025, we use GitHub issues (for the first part of 2025), app Tasks, and GitHub PRs (the idea being that PRs might help supplement the app Tasks as a second, independent view of the same engineering work).

# Other links

- App Task for the 2025 project extraction work is [internal app task, not public]
