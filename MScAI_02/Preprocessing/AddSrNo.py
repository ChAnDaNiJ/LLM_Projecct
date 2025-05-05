#Adds Sr Number to Input files and Binary Annotations

import os
import pandas as pd
from pathlib import Path

# Define input and output directories
input_folder = Path("../Binary_Annotations")
output_folder = Path("../Binary_Annotations_With_Serials")

# Create the output folder if it doesn't exist
output_folder.mkdir(exist_ok=True)

# Loop through all CSV files in the input folder
for csv_file in input_folder.glob("*.csv"):
    df = pd.read_csv(csv_file)
    df.insert(0, "Sr. No.", range(1, len(df) + 1))

    output_file = output_folder / csv_file.name
    df.to_csv(output_file, index=False)

print("Serial numbers added and files saved in 'Binary_Annotations_With_Serials'")