from context import db_vectorized_context, guru_vectorized_content, setup_bigquery_client
from decision import Decision
from metric import Metric


def run_decision_example():
    example_decision_description = """
    We're trying to decide if we should hire an additional sales rep for the North American Market.
    - Reps sell Convictional to Buyers on a SaaS contract @ ~$24K
    - We pay reps an approximate 15 percent commission on the first year of the contract on top of a base salary
    """

    bq_client = setup_bigquery_client()

    # Initialize our example decision object
    example_decision = Decision(
        description=example_decision_description,
        company_vectorized_context=guru_vectorized_content,
        database_vectorized_context=db_vectorized_context,
        query_client=bq_client,
    )

    # Brainstorm metrics
    brainstormed_metrics = example_decision.brainstorm_metrics()
    print("Brainstormed Metrics:")
    print(brainstormed_metrics)

    # Extract metrics
    extracted_metrics = example_decision.extract_metrics()
    print("\nExtracted Metrics:")
    print(extracted_metrics)

    # Check metric supportability
    checked_metrics = example_decision.check_metric_supportability()
    print("\nChecked Metrics:")
    print(checked_metrics)

    # Analyze metrics
    analyzed_metrics = example_decision.analyze_metrics()
    print("\nAnalyzed Metrics:")
    print(analyzed_metrics)

    # Summarize findings
    summary = example_decision.summarize_findings()
    print("\nSummary:")
    print(summary)


def run_metric_example():
    example_metric = """
    Total Buyer GMV processed over the last 365 days, grouped monthly and by buyer name
    """

    bq_client = setup_bigquery_client()

    # Initialize our example metric object
    example_metric_obj = Metric(
        description=example_metric, query_client=bq_client, vectorized_context=db_vectorized_context
    )

    # Get database schemas
    schemas = example_metric_obj.get_database_schemas()
    print("Database Schemas:")
    print(schemas)

    # Write query
    query = example_metric_obj.write_query()
    print("\nQuery:")
    print(query)

    # Execute query
    results = example_metric_obj.execute_query()
    print("\nQuery Results:")
    print(results)

    # Analyze results
    analysis = example_metric_obj.data_analyst()
    print("\nAnalysis:")
    print(analysis)


if __name__ == "__main__":
    print("Running Decision Example:")
    run_decision_example()

    # print("\nRunning Metric Example:")
    # run_metric_example()
