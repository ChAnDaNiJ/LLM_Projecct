import os
import pandas as pd
import numpy as np

# Define the directory containing the CSV files
directory = "C:\\Users\dell\Downloads\\LLM-Project-master\\LLM-Project-master\\CASE\\Physiological"  # Change this to your actual path

# Target number of rows (excluding the header)
TARGET_ROWS = 49031

def interpolate_csv(file_path):
    """Interpolates a CSV file if it has fewer than TARGET_ROWS rows."""
    
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check the number of rows (excluding the header)
    if len(df) >= TARGET_ROWS:
        print(f"Skipping {file_path}, already has {len(df)} rows.")
        return
    
    print(f"Interpolating {file_path}...")

    # Generate missing row indices
    existing_indices = np.linspace(0, TARGET_ROWS - 1, num=len(df))
    full_indices = np.arange(TARGET_ROWS)

    # Interpolate missing values
    df_interpolated = pd.DataFrame(index=full_indices)
    for col in df.columns:
        df_interpolated[col] = np.interp(full_indices, existing_indices, df[col])

    # Save the interpolated CSV file with a new prefix
    new_file_path = os.path.join(os.path.dirname(file_path), "new_" + os.path.basename(file_path))
    df_interpolated.to_csv(new_file_path, index=False)

    print(f"Saved interpolated file as: {new_file_path}")

def process_directory(directory):
    """Processes all CSV files in the given directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            interpolate_csv(file_path)

# Run the script
process_directory(directory)
