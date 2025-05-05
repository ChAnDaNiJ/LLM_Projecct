#Creation of Final_CASE.csv with 400 rows from each user and creating Sample to test the codes

import pandas as pd
import os
import random

# Constants
SOURCE_FOLDER = "New_Folder"
TARGET_FILE = "../Final_CASE.csv"
SAMPLE_SIZE = 400

# Collect all .csv files
csv_files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(".csv")]

sampled_dfs = []

for file_name in csv_files:
    file_path = os.path.join(SOURCE_FOLDER, file_name)
    df = pd.read_csv(file_path)

    # Randomly sample 400 rows
    sampled_df = df.sample(n=SAMPLE_SIZE, random_state=42)

    # Add subject_id column for traceability
    subject_id = os.path.splitext(file_name)[0]  # sub_1 from sub_1.csv
    sampled_df["subject_id"] = subject_id

    sampled_dfs.append(sampled_df)

# Combine all sampled data
final_df = pd.concat(sampled_dfs, ignore_index=True)

# Save final result
final_df.to_csv(TARGET_FILE, index=False)

print(f"Saved Final_CASE.csv with shape: {final_df.shape}")
