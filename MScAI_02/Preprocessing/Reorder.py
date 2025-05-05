#Reorder columns in all files in 'New_folder'

import os
import pandas as pd

folder = 'New_folder'
final_columns = [
    'Sr. No.', 'Participant_ID', 'daqtime', 'ecg', 'bvp', 'gsr', 'rsp', 'skt',
    'emg_zygo', 'emg_coru', 'emg_trap', 'video',
    'valence', 'arousal', 'Valence_class', 'Arousal_class'
]

for file in os.listdir(folder):
    if not file.endswith('.csv'):
        continue

    path = os.path.join(folder, file)
    df = pd.read_csv(path)

    df = df.loc[:, ~df.columns.duplicated()]

    # Remove 'Sr. No.' if present
    if 'Sr. No.' in df.columns:
        df = df.drop(columns=['Sr. No.'])

    # Reorder columns if all expected columns are present
    present = [col for col in final_columns if col in df.columns]
    df = df[present]

    df.to_csv(path, index=False)

print("✅ Cleaned and reordered columns in all files in 'New_folder'.")
