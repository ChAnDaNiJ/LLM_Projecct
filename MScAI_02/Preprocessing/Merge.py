#Merges both "Binary_Annotations_With_Serials" and "Input_With_Serials" folders

import os
import pandas as pd

folder_1 = 'Binary_Annotations_With_Serials'
folder_2 = 'Input_With_Serials'
new_folder = 'new_folder'
os.makedirs(new_folder, exist_ok=True)

for file2 in os.listdir(folder_2):
    if not file2.endswith('.csv'):
        continue

    participant_id = os.path.splitext(file2)[0]  # sub_1
    file1 = f"a_{participant_id}.csv"           # a_sub_1.csv

    path1 = os.path.join(folder_1, file1)
    path2 = os.path.join(folder_2, file2)

    if not os.path.exists(path1):
        print(f"Missing file: {path1}, skipping.")
        continue

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    if 'jstime' in df1.columns:
        df1 = df1.drop(columns=['jstime'])

    merged_df = pd.concat([df2, df1], axis=1)

    # Move 'Participant_ID' to first column
    if 'Participant_ID' in merged_df.columns:
        cols = merged_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('Participant_ID')))
        merged_df = merged_df[cols]

    output_path = os.path.join(new_folder, file2)
    merged_df.to_csv(output_path, index=False)

print("✅ Merging complete. Files saved in 'New_folder'.")
