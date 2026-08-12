import argparse
import src.forecast as src_main


def main():
    parser = argparse.ArgumentParser(description="Forecasting time series data")
    parser.add_argument("command", nargs="?", help="Command to execute")
    args = parser.parse_args()

    if args.command is None:
        src_main.main()
    else:
        command_to_execute = getattr(src_main, args.command, None)
        if command_to_execute:
            command_to_execute()
        else:
            print(f"Command '{args.command}' not found in src_main.")


if __name__ == "__main__":
    main()
