from src.main import main

if __name__ == "__main__":
    """
    This is the main entrypoint for the humans_and_llms experiment.
    Run this using: `make run_experiment ARGS="humans_and_llms"`
    """
    # Call main directly since it's not an async function
    main()
