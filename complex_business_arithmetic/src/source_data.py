import yaml

from .settings import settings
from .models import TestCaseFilter, TestCase


source_data_file = settings.input_path / "test_cases.yaml"


def load_test_case_from_file(test_case_filter: TestCaseFilter) -> TestCase:
    """
    This function loads a specific test case from a file

    To do this:
    1. Load the raw test cases data from the file
    2. Find the specific test case from the raw data
    3. Convert the raw test case to a test case object
    """
    print("Loading test case from file...")

    # Load all raw test cases
    print("Loading raw yaml test cases data...")

    with open(source_data_file, "r") as f:
        file_data = yaml.safe_load(f)
    raw_test_cases_data: list[dict] = file_data["test_cases"]

    # Get the specific test case
    print("Getting specific test case...")

    filter_key = test_case_filter.key
    filter_value = test_case_filter.value

    for raw_test_case in raw_test_cases_data:
        if raw_test_case[filter_key] == filter_value:
            test_case: TestCase = TestCase(**raw_test_case)
            break
    else:
        raise ValueError(f"Test case with {filter_key}: '{filter_value}' not found in raw data.")

    print(f"Loaded test case id: {test_case.id}")
    print(f"Loaded test case name: {test_case.name}")
    print(f"With results output file root: {test_case.results_csv_output_file_name}")

    return test_case
