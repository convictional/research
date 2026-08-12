import asyncio
import csv
import pandas as pd
from pathlib import Path
import subprocess
import sys
import warnings

from datetime import datetime
from tortoise import Tortoise
from typing import Dict, Any

from .bigquery_context import get_bigquery_tables_for_query, check_schema_cache_freshness
from .postgres_context import (
    ContentConfig,
    create_cache_tables,
    format_schemas_for_prompt,
    get_similar_queries_context,
    get_table_schemas,
    save_verified_query,
)
from .instruct_llm import set_async_instructor_client
from .llm import (
    generate_sql,
    get_sql_feedback,
    get_sql_self_reflection,
    generate_answer,
)
from .settings import settings, logger, CLAUDE_SONNET, output_dir
from common.bigquery import query_bq
from common.prompt_template_engine import initialize_and_register_prompt_templates

# Suppress the specific Google Auth warning about user credentials
warnings.filterwarnings(
    "ignore",
    message="Your application has authenticated using end user credentials from Google Cloud SDK without a quota project.*",
    module="google.auth._default",
)

# Create colorful section headers
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
GREEN = "\033[92m"
PURPLE = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Initialize the prompt template engine
initialize_and_register_prompt_templates(Path(__file__).parent / "prompts")

# CSV logging setup
CSV_LOG_PATH = output_dir / "sql_queries_benchmark.csv"
CSV_HEADERS = [
    "timestamp",
    "question",
    "generated_sql",
    "database_type",
    "explanation",
    "results_summary",
    "verified",
    "feedback",
    "suggested_sql",
    "attempts",
    "max_attempts",
    "answer",
    "caveats",
    "follow_up_questions",
    "thinking_budget_tokens",
    "git_commit_sha",
    "human_verified",
    "human_correction",
    "similar_queries_count",
    "similar_queries",
]


def get_git_commit_sha() -> str:
    """Get the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Error getting git commit SHA: {e}")
        return "unknown"


def initialize_csv_log():
    """Initialize the CSV log file with headers if it doesn't exist."""
    file_exists = CSV_LOG_PATH.exists()

    # If file doesn't exist, create it with headers
    if not file_exists:
        with open(CSV_LOG_PATH, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADERS)
        logger.info(f"Created new SQL query benchmark log at {CSV_LOG_PATH}")
    else:
        logger.info(f"Appending to existing SQL query benchmark log at {CSV_LOG_PATH}")


def log_to_csv(data: Dict[str, Any]):
    """Log query processing data to CSV file."""
    try:
        # Extract metadata if available
        metadata = data.get("metadata", {})
        database_type = metadata.get("database_type", "PostgreSQL")
        attempts = metadata.get("attempts", 1)
        max_attempts = metadata.get("max_attempts", settings.max_query_attempts)

        # Get git commit SHA
        git_commit_sha = get_git_commit_sha()

        # Extract similar queries data
        similar_queries_count = metadata.get("similar_queries_count", 0)
        similar_queries = metadata.get("similar_queries", "")

        # Prepare row data in the same order as headers
        row = [
            datetime.now().isoformat(),
            data.get("question", ""),
            data.get("sql", ""),
            database_type,
            data.get("explanation", ""),
            data.get("results", "")[:500] if data.get("results") else "",  # Truncate long results
            data.get("verified", False),
            data.get("feedback", ""),
            data.get("suggested_sql", ""),
            attempts,
            max_attempts,
            data.get("answer", ""),
            data.get("caveats", ""),
            data.get("follow_up_questions", ""),
            settings.thinking_budget_tokens,
            git_commit_sha,
            data.get("human_verified", None),  # Was it human verified
            data.get("human_correction", ""),  # Human correction if provided
            similar_queries_count,  # Count of similar queries found
            similar_queries,  # The similar queries as formatted string
        ]

        with open(CSV_LOG_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        logger.info(f"Logged {database_type} query data to CSV: {data.get('question', 'unknown question')}")
    except Exception as e:
        logger.error(f"Error logging to CSV: {e}")


async def setup_database():
    """Set up the database connection and create tables if needed."""
    await Tortoise.init(config=ContentConfig.tortoise_orm)
    await create_cache_tables()


async def run_query(sql: str, database_type: str = "BigQuery") -> str:
    """
    Execute a SQL query and return the results as a string.

    Args:
        sql: The SQL query to execute
        database_type: The type of database to run the query against (always "BigQuery" now)

    Returns:
        The query results as a formatted string
    """
    # We're now only using BigQuery for query execution
    try:
        # Run query using the BigQuery client
        logger.info(f"Executing BigQuery query: {sql}")

        # Run the query using the common helper
        df = await asyncio.to_thread(query_bq, sql, settings.gcp_project)

        if df.empty:
            return "No data returned"

        # Format pandas DataFrame as a string with limited column width
        with pd.option_context("display.max_colwidth", 40):
            result_str = df.to_string(index=False)

        # Limit display to first 10 rows if there are more
        if len(df) > 10:
            result_str += f"\n\n... and {len(df) - 10} more rows (showing first 10 of {len(df)} total rows)"

        return result_str

    except Exception as e:
        logger.error(f"Error executing BigQuery query: {str(e)}", exc_info=True)
        return f"Error executing BigQuery query: {str(e)}"


async def process_natural_language_query(question: str, auto_save: bool = False) -> Dict[str, Any]:
    """
    Process a natural language query, generate SQL, execute it, and verify the results.
    If the query is not verified, attempt refinement through self-reflection.

    Args:
        question: The natural language question to convert to SQL
        auto_save: Whether to automatically save verified queries (default is False since we now use human verification)

    Returns:
        Dictionary with the results of the process
    """

    # Initialize attempt counter for self-reflection loop
    attempt_counter = 1
    max_attempts = settings.max_query_attempts

    # Initial SQL generation and execution
    print(f"\n{YELLOW}=============== STEP 1: QUERY ANALYSIS (ATTEMPT #{attempt_counter}) ==============={RESET}")
    logger.info(f"🔎 Processing natural language query: '{question}'")

    # First get SQL context (tables and similar queries)
    print(f"{CYAN}> Fetching context for query...{RESET}")
    logger.info("🔍 Fetching database schema and similar queries...")

    # Get BigQuery tables for the query
    tables = await get_bigquery_tables_for_query(question)
    table_schemas = await format_schemas_for_prompt(tables) if tables else None

    # Get similar queries
    similar_context = await get_similar_queries_context(question)
    similar_queries = similar_context["similar_queries"]

    logger.info(f"Context prepared: {len(similar_queries)} similar queries, {len(tables) if tables else 0} tables")

    # Generate SQL from natural language
    print(f"{CYAN}> Generating SQL using LLM...{RESET}")
    logger.info("🤖 Generating SQL using LLM...")
    sql_response = await generate_sql(question, table_schemas=table_schemas, similar_queries=similar_queries)
    print(f"\n{PURPLE}Generated SQL:{RESET}\n{WHITE}{sql_response.sql}{RESET}")
    print(f"\n{PURPLE}Explanation:{RESET}\n{WHITE}{sql_response.explanation}{RESET}")
    logger.info(f"✅ SQL generated successfully: {len(sql_response.sql)} characters")

    # Execution and verification loop
    verified = False
    current_sql = sql_response.sql
    current_explanation = sql_response.explanation
    current_results = ""
    current_feedback = None

    while attempt_counter <= max_attempts:
        # Execute the current SQL query
        print(f"\n{YELLOW}=============== STEP 2: EXECUTING SQL (ATTEMPT #{attempt_counter}) ==============={RESET}")
        database_type = "BigQuery"
        print(f"{CYAN}> Executing SQL on {database_type}...{RESET}")
        logger.info(f"⚙️ Executing SQL on {database_type} (attempt #{attempt_counter})...")
        logger.info(f"SQL Query:\n{current_sql}")

        # Try to execute the query
        try:
            current_results = await run_query(current_sql, database_type)
            print(f"\n{PURPLE}Query results ({database_type}):{RESET}\n{WHITE}{current_results}{RESET}")
            logger.info(f"✅ Query executed successfully: {len(current_results.splitlines())} lines of results")
        except Exception as e:
            # If execution fails, set results to the error message
            error_message = str(e)
            current_results = f"Error executing query: {error_message}"
            print(f"\n{RED}Query execution failed:{RESET}\n{WHITE}{current_results}{RESET}")
            logger.error(f"❌ Query execution failed: {error_message}")

        # Get feedback on the query and results
        print(
            f"\n{YELLOW}=============== STEP 3: QUERY VERIFICATION (ATTEMPT #{attempt_counter}) ==============={RESET}"
        )
        print(f"{CYAN}> Getting feedback on query and results...{RESET}")
        logger.info(f"🔍 Getting feedback on query and results (attempt #{attempt_counter})...")

        # Get feedback (even if execution failed - the failure will be part of the feedback)
        current_feedback = await get_sql_feedback(
            question=question,
            sql=current_sql,
            results=current_results,
            table_schemas=table_schemas,
            similar_queries=similar_queries,
        )
        verified = current_feedback.verified
        verified_color = GREEN if verified else RED

        print(f"\n{PURPLE}Verified:{RESET} {verified_color}{verified}{RESET}")
        print(f"\n{PURPLE}Feedback:{RESET}\n{WHITE}{current_feedback.feedback}{RESET}")
        logger.info(f"✅ Feedback received - Query verified: {verified}")

        if current_feedback.suggested_sql:
            print(f"\n{PURPLE}Suggested SQL:{RESET}\n{CYAN}{current_feedback.suggested_sql}{RESET}")
            logger.info("📝 LLM suggested an improved SQL query")

        # If verified or reached max attempts, break out of the loop
        if verified or attempt_counter >= max_attempts:
            break

        # Otherwise, try to improve the query through self-reflection
        print(
            f"\n{YELLOW}=============== STEP 4: QUERY REFINEMENT (ATTEMPT #{attempt_counter}) ==============={RESET}"
        )
        print(f"{CYAN}> Performing SQL self-reflection and refinement...{RESET}")
        logger.info(f"🧠 Performing SQL self-reflection (attempt #{attempt_counter})...")

        # Get self-reflection to improve the query
        reflection = await get_sql_self_reflection(
            question=question,
            sql=current_sql,
            results=current_results,
            feedback=current_feedback.feedback,
            attempt_counter=attempt_counter,
            suggested_sql=current_feedback.suggested_sql,
            table_schemas=table_schemas,
            similar_queries=similar_queries,
        )

        # Update current SQL with the improved one
        current_sql = reflection.improved_sql
        print(f"\n{PURPLE}Improved SQL:{RESET}\n{WHITE}{current_sql}{RESET}")
        print(f"\n{PURPLE}Improvement explanation:{RESET}\n{WHITE}{reflection.explanation}{RESET}")
        logger.info("✅ Query refined through self-reflection")

        # Increment attempt counter for next iteration
        attempt_counter += 1

    # Generate a concise answer to the original question
    print(f"\n{YELLOW}=============== STEP 5: GENERATING ANSWER ==============={RESET}")
    print(f"{CYAN}> Generating concise answer to your question...{RESET}")
    logger.info("🧠 Generating concise answer from query results...")

    tables = await get_bigquery_tables_for_query(question)
    formatted_schemas = await format_schemas_for_prompt(tables) if tables else None

    answer_response = await generate_answer(
        question=question, sql=current_sql, results=current_results, table_schemas=formatted_schemas
    )

    # Display the answer
    print(f"\n{PURPLE}Answer:{RESET} {WHITE}{answer_response.answer}{RESET}")

    if answer_response.caveats:
        print(f"\n{PURPLE}Caveats:{RESET} {WHITE}{answer_response.caveats}{RESET}")

    if answer_response.follow_up_questions:
        print(f"\n{PURPLE}You might also want to ask:{RESET} {WHITE}{answer_response.follow_up_questions}{RESET}")

    logger.info("✅ Generated concise answer to question")

    # Only auto-save if specifically requested (most cases now use the human verification step)
    if auto_save and verified:
        print(f"\n{YELLOW}=============== STEP 6: SAVING VERIFIED QUERY ==============={RESET}")
        print(f"{CYAN}> Saving verified query to database...{RESET}")
        logger.info("💾 Saving verified query to database...")
        query_id = await save_verified_query(
            question=question, sql=current_sql, result_sample=current_results, verified=True
        )
        print(f"\n{GREEN}Saved verified query with ID: {query_id}{RESET}")
        logger.info(f"✅ Saved verified query with ID: {query_id}")

    # Prepare complete response
    response_data = {
        "question": question,
        "sql": current_sql,
        "explanation": current_explanation if attempt_counter == 1 else "See improvement explanation",
        "results": current_results,
        "verified": verified,
        "feedback": current_feedback.feedback if current_feedback else "",
        "suggested_sql": current_feedback.suggested_sql if current_feedback and current_feedback.suggested_sql else "",
        "answer": answer_response.answer,
        "caveats": answer_response.caveats if answer_response.caveats else "",
        "follow_up_questions": answer_response.follow_up_questions if answer_response.follow_up_questions else "",
        "metadata": {"database_type": database_type, "attempts": attempt_counter, "max_attempts": max_attempts},
    }

    logger.info(f"✅ Completed processing query: '{question}' after {attempt_counter} attempts. Verified: {verified}")

    # Return the complete response
    return response_data


async def handle_human_verification(response):
    """
    Handle human verification of a query result, with options to run suggested SQL.

    Args:
        response: Dictionary with query response data

    Returns:
        tuple: (bool, dict) - (True if verified and saved, Updated verification data)
    """
    # Initialize verification data with defaults
    verification_data = {
        "verified_in_handler": False,  # Did verification happen in this handler
        "final_sql": response["sql"],  # Default to original SQL
        "final_results": response["results"],  # Default to original results
        "human_verified": False,
        "human_correction": "",
    }

    # Check if query is verified or if there's still a suggested SQL after all attempts
    if not response["verified"] and response["suggested_sql"]:
        # Ask if user wants to run the suggested SQL
        run_suggested = input(
            f"\n{BOLD}{CYAN}Query not verified after {response['metadata']['attempts']} attempts. Run suggested SQL? (y/n): {RESET}"
        )
        if run_suggested.lower() == "y":
            print(f"\n{YELLOW}=============== EXECUTING SUGGESTED SQL ==============={RESET}")
            results = await run_query(response["suggested_sql"], "BigQuery")
            print(f"\n{PURPLE}Suggested SQL Results:{RESET}\n{WHITE}{results}{RESET}")

            # Update verification data for the suggested SQL
            verification_data["final_sql"] = response["suggested_sql"]
            verification_data["final_results"] = results
            verification_data["human_correction"] = "Used LLM-suggested SQL instead of original"

            # Display a summary before asking to save
            print(f"\n{PURPLE}Query to save:{RESET}\n{WHITE}{response['suggested_sql']}{RESET}")
            print(
                f"\n{PURPLE}Results summary:{RESET}\n{WHITE}{results[:500]}{'...' if len(results) > 500 else ''}{RESET}"
            )

            # Ask if user wants to save the suggested SQL
            save_suggested = input(
                f"\n{BOLD}{CYAN}Would you like to save this suggested SQL as verified? (y/n): {RESET}"
            )
            if save_suggested.lower() == "y":
                query_id = await save_verified_query(
                    question=response["question"], sql=response["suggested_sql"], result_sample=results, verified=True
                )

                # Mark as verified
                verification_data["verified_in_handler"] = True
                verification_data["human_verified"] = False

                print(f"\n{GREEN}Saved suggested SQL as verified with ID: {query_id}{RESET}")
                logger.info(f"✅ Saved suggested SQL as verified with ID: {query_id}")
                return True, verification_data

    # Allow user to enter custom SQL if still not satisfied
    run_custom = input(f"\n{BOLD}{CYAN}Would you like to enter custom SQL? (y/n): {RESET}")
    if run_custom.lower() == "y":
        custom_sql = input(f"\n{BOLD}{CYAN}Enter your SQL query: {RESET}")
        print(f"\n{YELLOW}=============== EXECUTING CUSTOM SQL ==============={RESET}")
        try:
            results = await run_query(custom_sql, "BigQuery")
            print(f"\n{PURPLE}Custom SQL Results:{RESET}\n{WHITE}{results}{RESET}")

            # Update verification data for custom SQL
            verification_data["final_sql"] = custom_sql
            verification_data["final_results"] = results
            verification_data["human_correction"] = "User provided custom SQL: \n" + custom_sql

            # Display a summary before asking to save
            print(f"\n{PURPLE}Custom SQL to save:{RESET}\n{WHITE}{custom_sql}{RESET}")
            print(
                f"\n{PURPLE}Results summary:{RESET}\n{WHITE}{results[:500]}{'...' if len(results) > 500 else ''}{RESET}"
            )

            # Ask if user wants to save the custom SQL
            save_custom = input(f"\n{BOLD}{CYAN}Would you like to save this custom SQL as verified? (y/n): {RESET}")
            if save_custom.lower() == "y":
                query_id = await save_verified_query(
                    question=response["question"], sql=custom_sql, result_sample=results, verified=True
                )

                # Mark as verified
                verification_data["verified_in_handler"] = True
                verification_data["human_verified"] = False

                print(f"\n{GREEN}Saved custom SQL as verified with ID: {query_id}{RESET}")
                logger.info(f"✅ Saved custom SQL as verified with ID: {query_id}")
                return True, verification_data
        except Exception as e:
            print(f"{RED}Error executing custom SQL: {str(e)}{RESET}")

    return False, verification_data


async def run_demo():
    """Run a demo of the agentic SQL system with example questions."""

    print(f"\n{BOLD}{YELLOW}================================================{RESET}")
    print(f"{BOLD}{YELLOW}           AGENTIC SQL DEMO                     {RESET}")
    print(f"{BOLD}{YELLOW}================================================{RESET}")
    print(f"{CYAN}This demo will process natural language questions and generate SQL queries{RESET}")

    # Set up the database
    print(f"\n{YELLOW}Setting up database connection...{RESET}")
    await setup_database()
    print(f"{GREEN}Database connected successfully{RESET}")

    # Check BigQuery schema cache status
    print(f"\n{YELLOW}Checking BigQuery schema cache status...{RESET}")

    is_cache_fresh = await check_schema_cache_freshness()

    if is_cache_fresh:
        print(f"{GREEN}BigQuery schema cache is fresh and ready to use{RESET}")
    else:
        print(f"{YELLOW}BigQuery schema cache is not available or stale{RESET}")
        print(f"{YELLOW}It will be automatically updated on first query{RESET}")

    # Initialize CSV logging
    initialize_csv_log()
    print(f"{CYAN}Logging data to benchmark file: {CSV_LOG_PATH}{RESET}")

    # Configure LLM clients
    print(f"\n{YELLOW}Configuring LLM client...{RESET}")
    try:
        set_async_instructor_client(
            CLAUDE_SONNET,
            api_key=settings.anthropic_api_key,
        )
        print(f"{GREEN}Using Anthropic API with model: {CLAUDE_SONNET}{RESET}")
    except Exception as e:
        print(f"\n{RED}WARNING: Error configuring LLM client: {e}{RESET}")
        print(f"{RED}SQL generation and feedback may not work properly.{RESET}")
        print(f"{RED}You can still explore database schema and run queries directly.{RESET}")

    # Example questions for BigQuery
    example_questions = [
        "How many active users does our platform have?",
        "How many orgs were created in Q1 2025?",
        "How many non Convictional, non internal users did we add over the last 90 days?",
        "How many non Convictional, non internal orgs do we have that have had at least one activity (activity within the last 90 days)?",
        "Break down the usage of the meetings feature by org name",
        "Give me a table showing org names down the rows and feature names across the columns with total usage as table values. I want to see how each org's total usage of meetings, threads, discussions, decisions, goals, etc compare.",
        "Provide a time series showing thread usage by org over time",
        "What was our total spend with the vendor Paradise Point in USD?",
        "What was our total vendor spend so far in 2025?",
        "Show us the top 5 vendors by spend so far in 2025 with monthly totals",
        "Show me all of the Convictional user bios",
        "What companies have a billing contract with us?",
        "Which companies are active (using some features in the last 30 days) but have not used threads? We want to target these companies for enablement so they should not be Convictional or internal",
        "What's the moving average of feature usage over the last 90 days? Focus on meetings, discussions, goals, decisions, tasks and threads. I want to see the moving average of each feature by org name.",
        "Who is the power user of each feature? A power user is the user who most uses each feature at each org. I want to see the power user for each feature by org name.",
        "How many users have used the meetings feature in the last 30 days?",
        "How many users have used the threads feature in the last 30 days?",
        "Show me any vendor with whom we've had at least one month of spend greater than $5000USD in the last 120 days",
        "Forecast our total vendor spend for the next 3 months using moving averages",
        "Show me the monthly average count of vendors with over $1000 spend during that month for the last 12 months",
        "What percent of time is each feature used first by an external org?",
    ]

    print(f"\n{BOLD}{YELLOW}================================================{RESET}")
    print(f"{BOLD}{YELLOW}           STARTING DEMO QUERIES               {RESET}")
    print(f"{BOLD}{YELLOW}================================================{RESET}")

    # Process each question
    for i, question in enumerate(example_questions, 1):
        print(f"\n{BOLD}{YELLOW}QUERY {i}/{len(example_questions)}: {question}{RESET}\n")

        try:
            # Process the query but don't automatically save
            response = await process_natural_language_query(question, auto_save=False)
            print(f"\n{GREEN}QUERY {i}/{len(example_questions)} PROCESSED{RESET}")

            # Add human verification step
            print(f"\n{YELLOW}=============== STEP 6: HUMAN VERIFICATION ==============={RESET}")

            # Final data to log - start with the current response
            final_log_data = response.copy()
            was_verified = False

            # Show the SQL and results for verification
            print(f"\n{PURPLE}SQL to verify:{RESET}\n{WHITE}{response['sql']}{RESET}")
            print(
                f"\n{PURPLE}Results sample:{RESET}\n{WHITE}{response['results'][:500]}{'...' if len(response['results']) > 500 else ''}{RESET}"
            )

            verify_response = input(f"\n{BOLD}{CYAN}Is this query and answer correct? (y/n): {RESET}")

            if verify_response.lower() == "y":
                # User verified it's correct
                final_log_data["human_verified"] = True
                final_log_data["human_correction"] = ""
                was_verified = True

                # Save the query
                print(f"{CYAN}> Saving verified query to database...{RESET}")
                logger.info("💾 Saving verified query to database...")
                query_id = await save_verified_query(
                    question=question, sql=response["sql"], result_sample=response["results"], verified=True
                )
                print(f"\n{GREEN}Saved verified query with ID: {query_id}{RESET}")
                logger.info(f"✅ Saved verified query with ID: {query_id}")
            else:
                # User indicated it's not correct
                final_log_data["human_verified"] = False

                # Ask for correction feedback
                correction_feedback = input(
                    f"\n{BOLD}{CYAN}Please provide feedback on what's incorrect (press Enter to skip): {RESET}"
                )
                if correction_feedback:
                    final_log_data["human_correction"] = correction_feedback

                # Use the verification handler
                print(f"{YELLOW}Query not saved automatically. Starting verification process...{RESET}")
                was_verified, handler_data = await handle_human_verification(response)

                # If verification was completed in handler, update the final data
                if was_verified:
                    final_log_data["human_verified"] = handler_data["human_verified"]
                    final_log_data["human_correction"] += handler_data["human_correction"]
                    final_log_data["sql"] = handler_data["final_sql"]
                    final_log_data["results"] = handler_data["final_results"]

            # Log the final data once with all verification information
            log_to_csv(final_log_data)

            print(f"\n{GREEN}QUERY {i}/{len(example_questions)} COMPLETED{RESET}")

            # If not the last query, prompt to continue
            if i < len(example_questions):
                continue_prompt = input(f"\n{BOLD}{CYAN}Continue to next query? (y/n): {RESET}")
                if continue_prompt.lower() != "y":
                    print(f"{YELLOW}Demo stopped by user.{RESET}")
                    break

        except Exception as e:
            print(f"\n{RED}ERROR processing question {i}: {e}{RESET}")
            print(f"{YELLOW}Moving to next question...{RESET}\n")
            continue

    # Clean up
    await Tortoise.close_connections()


async def interactive_mode():
    """Run the system in interactive mode, allowing the user to enter questions."""

    print(f"\n{BOLD}{YELLOW}================================================{RESET}")
    print(f"{BOLD}{YELLOW}       AGENTIC SQL INTERACTIVE MODE             {RESET}")
    print(f"{BOLD}{YELLOW}================================================{RESET}")
    print(f"{CYAN}This will process your natural language questions and generate SQL queries{RESET}")

    # Set up the database
    print(f"\n{YELLOW}Setting up database connection...{RESET}")
    await setup_database()
    print(f"{GREEN}Database connected successfully{RESET}")

    # Check BigQuery schema cache status
    print(f"\n{YELLOW}Checking BigQuery schema cache status...{RESET}")

    is_cache_fresh = await check_schema_cache_freshness()

    if is_cache_fresh:
        print(f"{GREEN}BigQuery schema cache is fresh and ready to use{RESET}")
    else:
        print(f"{YELLOW}BigQuery schema cache is not available or stale{RESET}")
        print(f"{YELLOW}It will be automatically updated on first query{RESET}")

    # Initialize CSV logging
    initialize_csv_log()
    print(f"{CYAN}Logging data to benchmark file: {CSV_LOG_PATH}{RESET}")

    # Configure LLM clients
    print(f"\n{YELLOW}Configuring LLM client...{RESET}")
    try:
        set_async_instructor_client(
            CLAUDE_SONNET,
            api_key=settings.anthropic_api_key,
        )
        print(f"{GREEN}Using Anthropic API with model: {CLAUDE_SONNET}{RESET}")
    except Exception as e:
        print(f"\n{RED}WARNING: Error configuring LLM client: {e}{RESET}")
        print(f"{RED}SQL generation and feedback may not work properly.{RESET}")
        print(f"{RED}You can still explore database schema and run queries directly.{RESET}")

    print(f"\n{BOLD}{YELLOW}==== Agentic SQL Interactive Mode (BigQuery) ===={RESET}")
    print(f"{CYAN}Enter '{YELLOW}q{CYAN}' or '{YELLOW}quit{CYAN}' to exit.{RESET}")
    print(f"{CYAN}Enter '{YELLOW}schema{CYAN}' to view BigQuery schema information.{RESET}")
    print(
        f"{CYAN}Enter '{YELLOW}sql:YOUR_QUERY{CYAN}' or '{YELLOW}bq:YOUR_QUERY{CYAN}' to run BigQuery SQL directly.{RESET}"
    )

    query_count = 0

    while True:
        question = input(f"\n{BOLD}{CYAN}Enter your question: {RESET}")

        if question.lower() in ["q", "quit", "exit"]:
            print(f"{YELLOW}Exiting interactive mode...{RESET}")
            break

        if question.lower() == "schema":
            # Show database schema
            print(f"\n{YELLOW}=============== DATABASE SCHEMA ==============={RESET}")
            schemas = await get_table_schemas()
            formatted_schemas = await format_schemas_for_prompt(schemas)
            print(f"{CYAN}{formatted_schemas}{RESET}")
            continue

        if question.lower().startswith("sql:") or question.lower().startswith("bq:"):
            # Run SQL directly on BigQuery
            # Both prefixes now run on BigQuery for consistency
            sql_query = question[4:].strip() if question.lower().startswith("sql:") else question[3:].strip()

            print(f"\n{YELLOW}=============== EXECUTING DIRECT BIGQUERY SQL ==============={RESET}")
            print(f"{PURPLE}SQL Query:{RESET}\n{WHITE}{sql_query}{RESET}")
            try:
                results = await run_query(sql_query, "BigQuery")
                print(f"\n{PURPLE}Results (BigQuery):{RESET}\n{WHITE}{results}{RESET}")
            except Exception as e:
                print(f"{RED}Error executing BigQuery SQL: {str(e)}{RESET}")
            continue

        query_count += 1
        print(f"\n{YELLOW}=============== PROCESSING QUERY #{query_count} ==============={RESET}")

        try:
            # Process the query but don't automatically save
            response = await process_natural_language_query(question, auto_save=False)

            # Add human verification step
            print(f"\n{YELLOW}=============== STEP 6: HUMAN VERIFICATION ==============={RESET}")

            # Final data to log - start with the current response
            final_log_data = response.copy()
            was_verified = False

            # Show the SQL and results for verification
            print(f"\n{PURPLE}SQL to verify:{RESET}\n{WHITE}{response['sql']}{RESET}")
            print(
                f"\n{PURPLE}Results sample:{RESET}\n{WHITE}{response['results'][:500]}{'...' if len(response['results']) > 500 else ''}{RESET}"
            )

            verify_response = input(f"\n{BOLD}{CYAN}Is this query and answer correct? (y/n): {RESET}")

            if verify_response.lower() == "y":
                # User verified it's correct
                final_log_data["human_verified"] = True
                final_log_data["human_correction"] = ""
                was_verified = True

                # Save the query
                print(f"{CYAN}> Saving verified query to database...{RESET}")
                logger.info("💾 Saving verified query to database...")
                query_id = await save_verified_query(
                    question=question, sql=response["sql"], result_sample=response["results"], verified=True
                )
                print(f"\n{GREEN}Saved verified query with ID: {query_id}{RESET}")
                logger.info(f"✅ Saved verified query with ID: {query_id}")
            else:
                # User indicated it's not correct
                final_log_data["human_verified"] = False

                # Ask for correction feedback
                correction_feedback = input(
                    f"\n{BOLD}{CYAN}Please provide feedback on what's incorrect (press Enter to skip): {RESET}"
                )
                if correction_feedback:
                    final_log_data["human_correction"] = correction_feedback

                # Use the verification handler
                print(f"{YELLOW}Query not saved automatically. Starting verification process...{RESET}")
                was_verified, handler_data = await handle_human_verification(response)

                # If verification was completed in handler, update the final data
                if was_verified:
                    final_log_data["human_verified"] = handler_data["human_verified"]
                    final_log_data["human_correction"] = handler_data["human_correction"]
                    final_log_data["sql"] = handler_data["final_sql"]
                    final_log_data["results"] = handler_data["final_results"]

            # Log the final data once with all verification information
            log_to_csv(final_log_data)

        except Exception as e:
            print(f"{RED}Error processing question: {str(e)}{RESET}")
            logger.error(f"Error processing question: {str(e)}", exc_info=True)

    print(f"\n{GREEN}Thank you for using Agentic SQL!{RESET}")
    print(f"{CYAN}All queries have been logged to the benchmark file: {CSV_LOG_PATH}{RESET}")

    # Clean up
    await Tortoise.close_connections()


async def main():
    """Main entry point for the application."""

    print(f"\n{BOLD}{GREEN}================================================{RESET}")
    print(f"{BOLD}{GREEN}          AGENTIC SQL STARTED                   {RESET}")
    print(f"{BOLD}{GREEN}================================================{RESET}")

    # Check for --clear-queries argument
    clear_queries = "--clear-queries" in sys.argv
    if clear_queries:
        print(f"{YELLOW}Clearing existing SQL query pairs...{RESET}")
        # Setup database connection
        await setup_database()
        # Import and call the clear function
        from .postgres_context import clear_sql_query_pairs

        count = await clear_sql_query_pairs()
        print(f"{GREEN}Cleared {count} existing SQL query pairs{RESET}")
        await Tortoise.close_connections()
        # If only --clear-queries was provided, exit
        if len(sys.argv) == 2 and sys.argv[1] == "--clear-queries":
            print(f"{GREEN}Query clear operation completed{RESET}")
            return

    # Check for --clear-bq-cache argument
    clear_bq_cache = "--clear-bq-cache" in sys.argv
    if clear_bq_cache:
        print(f"{YELLOW}Clearing BigQuery schema cache...{RESET}")
        # Setup database connection
        await setup_database()
        # Import and call the clear function
        from .postgres_context import clear_bigquery_schema_cache

        count = await clear_bigquery_schema_cache()
        print(f"{GREEN}Cleared {count} BigQuery schema cache entries{RESET}")
        await Tortoise.close_connections()
        # If only --clear-bq-cache was provided, exit
        if len(sys.argv) == 2 and sys.argv[1] == "--clear-bq-cache":
            print(f"{GREEN}BigQuery cache clear operation completed{RESET}")
            return

    # Check for demo mode
    if "--demo" in sys.argv:
        print(f"{YELLOW}Starting in demo mode...{RESET}")
        # Run in demo mode
        await run_demo()
    else:
        print(f"{YELLOW}Starting in interactive mode...{RESET}")
        # Run in interactive mode
        await interactive_mode()

    print(f"\n{BOLD}{GREEN}================================================{RESET}")
    print(f"{BOLD}{GREEN}          AGENTIC SQL COMPLETED                {RESET}")
    print(f"{BOLD}{GREEN}================================================{RESET}")
