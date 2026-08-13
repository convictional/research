# Activity Querying

**Authors:** Ben Crouse, Adam McCabe

SQL queries for analyzing organizational activity data, knowledge flows, and collaboration patterns. These queries help understand how information moves through an organization, identify knowledge bridges, and measure department productivity.

## Contents

- Database schema (`schema.sql`) for person, artifact, and activity tracking
- SQL queries for different analysis scenarios:
  - Department-level metrics
  - Knowledge flow networks
  - Information bottlenecks
  - Organizational memory and expertise
  - Cross-team collaboration

## Usage

Run these queries against a database with the provided schema and populated with activity data to gain insights into organizational dynamics and knowledge transfer patterns.

> **Note:** The sample data file that originally accompanied these queries was a real database
> export and was removed before open-sourcing. Only `schema.sql` and the queries themselves ship
> here; you will need to populate the schema with your own data.
