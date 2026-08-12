import csv
import os
import datetime


def save_results_to_csv(results, rankings, feedback, file_path):
    fieldnames = ["model", "response", "ranking", "feedback"]

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header if the file is empty
        if file.tell() == 0:
            writer.writeheader()

        # Write the results, rankings, and feedback to the CSV file
        for model, response in results.items():
            writer.writerow(
                {
                    "model": model,
                    "response": response["response"],
                    "ranking": rankings.get(model, ""),
                    "feedback": feedback.get(model, ""),
                }
            )


def save_elo_ratings(elo_ratings, file_path):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fieldnames = ["timestamp"] + list(elo_ratings.keys())

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Read the existing fieldnames from the CSV file
    existing_fieldnames = []
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            existing_fieldnames = next(reader, [])

    # Add new fieldnames if they don't exist
    new_fieldnames = [field for field in fieldnames if field not in existing_fieldnames]
    fieldnames = existing_fieldnames + new_fieldnames

    with open(file_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header if the file is empty or new fieldnames are added
        if file.tell() == 0 or new_fieldnames:
            writer.writeheader()

        # Write the timestamp and ELO ratings to the CSV file
        row = {"timestamp": timestamp}
        row.update(elo_ratings)
        writer.writerow(row)


def load_elo_ratings(file_path):
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        # Get the last row (most recent ELO ratings)
        last_row = None
        for row in reader:
            last_row = row

        if last_row:
            # Remove the timestamp column and convert values to floats
            return {
                key: float(value[0]) if isinstance(value, list) else float(value)
                for key, value in last_row.items()
                if key != "timestamp"
            }

    return None
