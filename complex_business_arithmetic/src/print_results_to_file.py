import pandas as pd
from pathlib import Path

from .models import TestCase, LLMCalculationRequest, AuditorResponse
from .settings import settings


def print_results_to_file(
    test_case: TestCase,
    messages: list[dict],
    calculation_steps: list[LLMCalculationRequest],
    non_calculation_request_assistant_texts: list[str],
    audit_analysis: AuditorResponse,
):
    """
    Print the results to the output file
    """
    print("Printing results to file...")

    output_path_root = settings.output_path / test_case.results_csv_output_file_name

    print_calculation_steps(calculation_steps, test_case, output_path_root)
    print_messages(messages, test_case, output_path_root)
    print_all_results_to_csv(
        test_case,
        messages,
        calculation_steps,
        non_calculation_request_assistant_texts,
        audit_analysis,
        output_path_root,
    )


def print_calculation_steps(
    calculation_steps: list[LLMCalculationRequest], test_case: TestCase, output_path_root: Path
):
    """
    Print the calculation steps
    """
    print("Printing calculation steps...")

    calculation_steps_string = "".join(
        [
            f"CALCULATION STEP {i+1}\n----------------------------\ntool_use_id: {step.tool_use_id}\nID: {step.id}\nContext: {step.context}\nOperation: {step.operation}\nOperands: {step.operands}\nResult: {step.result}\n\n\n"
            for i, step in enumerate(calculation_steps)
        ]
    )

    with open(str(output_path_root) + "_calculation_steps.md", "w") as f:
        f.write(calculation_steps_string)


def print_messages(messages: list[dict], test_case: TestCase, output_path_root: Path):
    """
    Print the messages
    """
    print("Printing messages...")

    messages_string = "".join(
        [
            f"MESSAGE {i+1}\n----------------------------\nRole: {message['role'].upper()}\nContent:\n{message['content']}\n\n\n"
            for i, message in enumerate(messages)
        ]
    )

    with open(str(output_path_root) + "_messages.md", "w") as f:
        f.write(messages_string)


def print_all_results_to_csv(
    test_case: TestCase,
    messages: list[dict],
    calculation_steps: list[LLMCalculationRequest],
    non_calculation_request_assistant_texts: list[str],
    audit_analysis: AuditorResponse,
    output_path_root: Path,
):
    """
    Print all results to a CSV file
    """
    print("Printing all results to CSV...")

    data = {
        "test_case_id": [test_case.id],
        "test_case_name": [test_case.name],
        "test_case_user_query": [test_case.user_query],
        "test_case_attachments": [
            "".join(
                [
                    f"ATTACHMENT {i+1}\n------------------------\nFile name: {attachment.file_name}\nDescription:\n{attachment.description}\n\n\n"
                    for i, attachment in enumerate(test_case.attachments)
                ]
            )
        ],
        "test_case_run_duration_s": [test_case.run_duration_s],
        "test_case_num_calculation_steps": [test_case.num_calculation_steps],
        "test_case_num_llm_tool_use_calls": [len(set([step.tool_use_id for step in calculation_steps]))],
        "test_case_num_messages": [test_case.num_messages],
        "test_case_num_llm_responses": [test_case.num_llm_responses],
        "test_case_cumulative_num_input_tokens": [test_case.cumulative_num_input_tokens],
        "test_case_cumulative_num_output_tokens": [test_case.cumulative_num_output_tokens],
        "test_case_last_llm_response_num_input_tokens": [test_case.last_llm_response_num_input_tokens],
        "calculation_steps": [
            "".join(
                [
                    f"CALCULATION STEP {i+1}\n----------------------------\ntool_use_id: {step.tool_use_id}\nID: {step.id}\nContext: {step.context}\nOperation: {step.operation}\nOperands: {step.operands}\nResult: {step.result}\n\n\n"
                    for i, step in enumerate(calculation_steps)
                ]
            )
        ],
        "messages": [
            "".join(
                [
                    f"MESSAGE {i+1}\n----------------------------\nRole: {message['role'].upper()}\nContent:\n{message['content']}\n\n\n"
                    for i, message in enumerate(messages)
                ]
            )
        ],
        "non_calculation_request_assistant_texts": [
            "".join(
                [
                    f"TEXT {i+1}\n----------------\n{text}\n\n\n"
                    for i, text in enumerate(non_calculation_request_assistant_texts)
                ]
            )
        ],
        "llm_last_message_content": [
            next((message["content"] for message in reversed(messages) if message["role"] == "assistant"), None)
        ],
        "llm_audit_analysis": [audit_analysis.analysis],
        "llm_audit_request_duration_s": [audit_analysis.request_duration_s],
    }

    df = pd.DataFrame(data)
    df.to_csv(str(output_path_root) + "_all_results.csv", index=False)
