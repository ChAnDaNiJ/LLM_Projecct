import os
import pandas as pd

# Define the directory containing the CSV files
directory = r"C:\Users\dell\Downloads\LLM-Project-master\LLM-Project-master\CASE\Physiological"  # Change this to your actual path

def check_csv_rows(directory):
    """Checks and prints the number of rows in each CSV file in the directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            num_rows = len(df)  # Excludes the header
            print(f"{filename}: {num_rows} rows")

# Run the function
check_csv_rows(directory)
