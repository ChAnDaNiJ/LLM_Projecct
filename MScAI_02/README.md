# Emotion_Labeling-GPT4
Automated annotation framework using GPT-4 to reduce labeling overhead in emotion datasets through Few-Shot and Zero-Shot learning. 
Classifies valence and arousal levels from physiological signals (GSR, Heart Rate).

## Project Overview
This project automates the classification of valence and arousal from physiological sensor data using OpenAI’s GPT-4 API. 
It leverages Few-Shot and Zero-Shot prompting to minimize manual labeling.

## Dataset
- **CASE Dataset**: Annotated physiological responses using joystick-based continuous labels.
- **UserStudy Dataset**: Collected physiological responses from participants watching emotional videos, with annotations based on self-reported valence and arousal ratings using a post-trial questionnaire.

---
### CASE Dataset
The CASE Dataset is a public collection where participants wore sensors while watching emotional video clips. 
It includes their physiological signals like heart rate and skin conductance, along with their self-reported feelings of valence (pleasantness) and arousal (intensity). 
This dataset helps researchers train and test models that can recognize emotions from bodily signals across a wide and diverse group of people.

- **Participants**: 30 subjects (15 male, 15 female), aged 22–37.
- **Sensors**: ECG, BVP, EMG (3x), GSR, respiration, and skin temperature.
- **Sampling_Rate**: 1000 Hz (physiological), 20 Hz (annotations).
- **Stimuli**: 8 emotion-tagged videos (amusing, boring, relaxing, scary) validated in a
pre-study.
- **Annotations**: Continuous, 2D valence-arousal ratings based on the Circumplex Model
using JERI.

This part of the project processes the raw continuously annotated physiological signals of emotion from the [CASE dataset](https://gitlab.com/karan-shr/case_dataset).

| **Column Name**  | **Description**                                            |
| ---------------- |------------------------------------------------------------|
| `subject_id`     | Unique subject identifier (e.g., sub\_1).                  |
| `Participant_ID` | Numeric ID of participant.                                 |
| `daqtime`        | Time in milliseconds at which the data point was recorded. |
| `ecg`            | Electrocardiograph: records heart activity (e.g., heartbeat pattern) |
| `bvp`            | Blood Volume Pulse: tracks changes in blood flow           |
| `gsr`            | Galvanic Skin Response: measures skin conductivity due to sweating |
| `rsp`            | Respiration: captures breathing pattern or rhythm          |
| `skt`            | Skin Temperature: records skin surface temperature in Celsius |
| `emg_zygo`       | EMG of Zygomaticus Major: detects smiling-related facial muscle activity |
| `emg_coru`       | EMG of Corrugator Supercilii: detects frowning or brow tension |
| `emg_trap`       | EMG of Trapezius muscle: tracks shoulder/neck tension      |
| `video`          | Video ID associated with the emotional stimulus.           |
| `valence`        | Continuous emotional valence rating (pleasantness of emotion). |
| `arousal`        | Continuous emotional arousal rating (intensity of emotion). |
| `Valence_class`  | Discrete class label for valence (e.g., high vs. low).     |
| `Arousal_class`  | Discrete class label for arousal (e.g., high vs. low).     |

### Folder Structure ([CASE](CASE)):
Main directory for the CASE dataset and related outputs.

- **[CASE_Code](CASE/CASE_Code)** 
  - Contains Code for classifying emotions using GPT-4.
    - [FewShot](CASE/CASE_Code/FewShot) (Contains the script for few-shot classification.)\
      `API(FewShot).py`: Uses GPT-4 with few-shot examples to classify valence/arousal.
    - [ZeroShot](CASE/CASE_Code/ZeroShot) (Contains the script for zero-shot classification.)\
      `API(ZeroShot).py`: Uses GPT-4 without examples to classify valence/arousal.


- **[CASE_Data](CASE/CASE_Data)**
  - Raw data used for labeling.
    - `Final_CASE.csv`: Contains the sensor data and emotional annotations.


- **[CASE_Plots](CASE/CASE_Plots)** – Contains plots to discribe the CASE Dataset.
  - Visualizations of dataset distributions and model performance.
    * Multiple PNG files: show valence/arousal class distributions, ROC-AUC, combined plots, etc.
    * `Plots_Info [CASE].txt`: Describes what each plot represents.


- **[CASE_Results](CASE/CASE_Results)**
  - Results from running the classifiers.
    - [FewShot](CASE/CASE_Results/FewShot) (Contains the script for few-shot classification.)\
      `FewShot_CASE_labeled.csv`: Labeled data output from few-shot model.
    - [ZeroShot](CASE/CASE_Results/ZeroShot) (Contains the script for zero-shot classification.)\
      `ZeroShot_CASE_labeled.csv`: Labeled data output from zero-shot model.

### Scripts:

#### `API(ZeroShot).py`[[API(ZeroShot).py](CASE/CASE_Code/ZeroShot/API%28ZeroShot%29.py)]
  - Reads sensor data from a CSV file.
  - Sends each row to the OpenAI API to classify valence and arousal levels (as 1 or 2).
  - Saves the results to a new CSV.
  - Runs efficiently with parallel requests and error handling.

#### `API(FewShot).py`[[API(FewShot).py](CASE/CASE_Code/FewShot/API%28FewShot%29.py)]
  - Reads sensor data from a CSV file.
  - Builds a prompt using few-shot examples and each row's sensor values.
  - Sends each row to the OpenAI GPT-4 API to classify valence and arousal (as 1 or 2).
  - Appends results to a new CSV file.
  - Processes rows in parallel using async for faster speed.


### Outputs:
- `ZeroShot_CASE_labeled.csv`[[FewShot_CASE_labeled.csv](CASE/CASE_Results/FewShot/FewShot_CASE_labeled.csv)]
  - Output of `API(ZeroShot).py`


- `FewShot_CASE_labeled.csv`[[ZeroShot_CASE_labeled.csv](CASE/CASE_Results/ZeroShot/ZeroShot_CASE_labeled.csv)]
  - Output of `API(FewShot).py`

---
### UserStudy Dataset
The **UserStudy Dataset** is a small, custom-made collection of data where people watched emotional videos and their body responses (like heart rate and skin signals) were recorded. After each video, they rated how they felt in terms of **valence** (happy to sad) and **arousal** (excited to calm). This dataset helps us see if an AI model can understand emotions from physical signals, even with just a few examples.

- Participants: 36 users (equal gender distribution), aged 20–40.
- Sensor Setup: GSR (Grove V1.2) and HR (HW-827), sampled at 10 Hz.
- Annotation Interface: Emotion ratings based on keyboard arrow key movements
across the Circumplex model.
- Experiment Duration: Approximately 12 hours total across all users.
- Stimuli: 8 validated videos (2 per emotion quadrant).
- Annotation Approach: Continuous rating of emotion during playback.

This part of the project processes the post-video self-reported emotional responses collected through questionnaires, along with corresponding physiological signals recorded during video viewing.

| **Column Name**  | **Description**                                                  |
| ---------------- | ---------------------------------------------------------------- |
| `P_id`           | Participant ID – unique identifier for each participant.         |
| `start_time`     | Start time of the video segment (in seconds).                    |
| `end_time`       | End time of the video segment (in seconds).                      |
| `mean_GSR`       | Mean value of Galvanic Skin Response over the segment.           |
| `std_GSR`        | Standard deviation of GSR, indicating variability.               |
| `mean_HR`        | Mean Heart Rate over the segment.                                |
| `std_HR`         | Standard deviation of Heart Rate.                                |
| `mean_Valence`   | Average perceived valence for the segment (pleasantness).        |
| `std_Valence`    | Standard deviation of valence scores.                            |
| `median_Valence` | Median valence rating.                                           |
| `mean_Arousal`   | Average perceived arousal for the segment (emotional intensity). |
| `std_Arousal`    | Standard deviation of arousal scores.                            |
| `median_Arousal` | Median arousal rating.                                           |
| `videoID`        | Identifier of the video clip shown.                              |
| `Score`          | Overall emotion score derived or annotated.                      |
| `Valence`        | Final discrete valence label (e.g., high/low).                   |
| `Arousal`        | Final discrete arousal label (e.g., high/low).                   |
| `Index`          | Row index or data sample number.                                 |

### Folder Structure [UserStudy](UserStudy):
Main directory for the UserStudy dataset and its results.

- **[UserStudy_Code](UserStudy/UserStudy_Code)** 
  - Contains Scripts to run Few-Shot and Zero-Shot classification.
    - [FewShot](UserStudy/UserStudy_Code/FewShot) (Contains the script for few-shot classification.)\
      `FewShot_User.py`: Runs few-shot classification per user.  
      `FewShot_User_Looped.py`: Runs few-shot classification multiple times and logs average metrics.
    - [ZeroShot](UserStudy/UserStudy_Code/ZeroShot) (Contains the script for zero-shot classification.)\
      `ZeroShot_User.py`: Runs zero-shot classification on the whole dataset.\
      `ZeroShot_User_Looped.py`: Repeats zero-shot classification runs to compute average stats.


- **[UserStudy_Data](UserStudy/UserStudy_Data)**
  - Preprocessed physiological and annotation data.
    `Final_UserStudy.csv`: Final version of the dataset used for labeling.


- **[UserStudy_Plots](UserStudy/UserStudy_Plots)**
  - Contains plots summarizing data distribution and model results.
    * Includes performance metrics, class distributions, and plot documentation.
    * `Plots_Info [UserStudy].txt`: Describes each visualization.


- **[UserStudy_Results](UserStudy/UserStudy_Results)**
  - Outputs from model runs.
    - [FewShot](UserStudy/UserStudy_Results/FewShot) (Contains the script for few-shot classification.)\
      `fewshot_labeled_*.csv`: Labeled data for each run.\
      `fewshot_avg_metrics_*.csv`: Accuracy, precision, recall, etc.\
      `fewshot_log_*.txt`: Logs of few-shot runs.
    - [ZeroShot](UserStudy/UserStudy_Results/ZeroShot) (Contains the script for zero-shot classification.)\
      `zero_shot_labeled.csv`: Zero-shot labeling results.\
      `ZeroShot_User_Log_*.txt`: Log file with run metrics.

### Scripts:

#### `FewShot_User.py`[[FewShot_User.py](UserStudy/UserStudy_Code/FewShot/FewShot_User.py)]
  - Reads user-specific sensor data from CSV.
  - Splits each user’s data into train/test sets.
  - Builds few-shot prompts from training data.
  - Sends test data to GPT-4 for classification (Valence, Arousal).
  - Handles responses with async API calls and retries.
  - Saves predictions and true labels to a CSV.
  - Logs per-user and overall accuracy.

#### `FewShot_User_Looped.py`[[FewShot_User_Looped.py](UserStudy/UserStudy_Code/FewShot/FewShot_User_Looped.py)]
- Reads user-specific sensor data from a combined CSV.
- Loops through multiple train/test splits (10% to 90%) per user.
- Builds few-shot prompts using each user's training data.
- Sends test samples to GPT-4 via async API calls with retries.
- Parses model predictions for Valence and Arousal.
- Logs per-user accuracy and aggregates metrics across splits.
- Saves predictions, logs, and evaluation metrics to CSV files.

#### `ZeroShot_User.py`[[ZeroShot_User.py](UserStudy/UserStudy_Code/ZeroShot/ZeroShot_User.py)]
  - Loads full sensor dataset from CSV.
  - Splits data into large chunks (e.g., 1500 rows).
  - Builds zero-shot GPT-4 prompts for each chunk with only feature descriptions (no examples).
  - Classifies Valence and Arousal per row using async GPT-4 API requests.
  - Writes predictions and true labels to zero_shot_labeled.csv.
  - Computes and prints full classification metrics: Accuracy, Precision, Recall, F1, and ROC AUC.


#### `ZeroShot_User_Loop.py`[[ZeroShot_User_Looped.py](UserStudy/UserStudy_Code/ZeroShot/ZeroShot_User_Looped.py)]
  - Runs the zero-shot labeling process 5 times.
  - Collects classification metrics from each run.
  - Logs results to a timestamped log file.
  - Computes and logs the average and standard deviation of all metrics.


### Outputs:

- `zero_shot_labeled.csv`[[zero_shot_labeled.csv](UserStudy/UserStudy_Results/ZeroShot/zero_shot_labeled.csv)] and `ZeroShot_User_Log_20250415_020103.txt`[[ZeroShot_User_Log_20250415_020103.txt](UserStudy/UserStudy_Results/ZeroShot/ZeroShot_User_Log_20250415_020103.txt)]
  - Output of `ZeroShot_User_Loop.py`


- `fewshot_avg_metrics_*.csv`[[fewshot_avg_metrics_20250415-094316.csv](UserStudy/UserStudy_Results/FewShot/fewshot_avg_metrics_20250415-094316.csv)] , 
   `fewshot_labeled_*.csv`[[fewshot_labeled_20250415-094316.csv](UserStudy/UserStudy_Results/FewShot/fewshot_labeled_20250415-094316.csv)] and 
   `fewshot_log_*.txt`[[fewshot_log_20250415-094316.txt](UserStudy/UserStudy_Results/FewShot/fewshot_log_20250415-094316.txt)]
  - Output of `FewShot_User_Looped.py`[[FewShot_User_Looped.py](UserStudy/UserStudy_Code/FewShot/FewShot_User_Looped.py)]

---
## Setup Instructions

### Preprocessing for CASE Dataset
1. Download the CASE dataset (30 participants).
2. Run the preprocessing scripts in the following order (located in [Preprocessing](Preprocessing)):
- `Annotator.py` – Reads and interpolates CSV files to fill missing data.
- `Counter.py` – Verifies row counts; expected total: 49,031 rows (excluding headers).
- `AddSrNo.py` – Adds serial numbers to each row.
- `Merge.py` – Combines sensor data with binary annotations.
- `Reorder.py` – Cleans and reorders columns into a consistent structure.
- `Sample.py` – Samples 400 rows per user and compiles into `Final_CASE.csv` for model testing.

This completes the preprocessing. You'll now have a ready-to-use dataset for further learning.

### Running the Few-Shot & Zero-Shot Models
After preprocessing, you can use the models provided in the [FewShot]([FewShot](CASE/CASE_Code/FewShot)) and [ZeroShot]([ZeroShot](CASE/CASE_Code/ZeroShot)) folders:

#### Few-Shot Inference:
- API(FewShot).py:
  - Uses GPT-4 with training examples from each user. 
  - Builds prompts, gets predictions. 
  - Logs results and saves labeled output.

#### Zero-Shot Inference:
- API(ZeroShot).py:
  - Uses GPT-4 with only general instructions (no training samples).
  - Classifies each row based on features like GSR, HR, etc.

### Preprocessing Scripts:
`Annotator.py`
  - Reads all CSV files in the given directory.
  - Check the row count (excluding the header).
  - Interpolates missing rows using linear interpolation.
  - Saves a new version with "new_" as the prefix.

`Counter.py`
- Counts the number of rows in each data set and outputs
- Scans all CSV files in the directory.
- Counts the number of data rows (excluding the header).
- Prints the filename and row count.

`AddSrNo.py`
- Adds a "Sr. No." (serial number) column to the beginning of each CSV file.
- Reads all .csv files from the `../Binary_Annotations folder`.
- Inserts serial numbers starting from 1 for each row.
- Saves the modified files to the `../Binary_Annotations_With_Serials` directory.
- Automatically creates the output folder if it doesn’t exist.

`Merge.py`
- Merges paired CSV files from `Binary_Annotations_With_Serials` and `Input_With_Serials` folders.
- Matches files by extracting the _Participant ID_ and locating the corresponding annotation file prefixed with a_.
- Drops the _jstime_ column from the annotation file if present.
- Concatenates the input and annotation files side by side.
- Moves the _Participant_ID_ column to the first position if it exists.
- Saves the merged files into a new folder named `new_folder`, creating it if necessary.
- Skips and reports any missing matching annotation files.

`Reorder.py`
- Reorders and cleans columns in all CSV files inside the `New_folder` directory.
- Defines a standard column order to follow (final_columns).
- For each file:
  - Removes any duplicate columns.
  - Drops the _Sr. No._ column if present.
  - Reorders columns based on the predefined list, using only the columns that exist in the file.
- Overwrites each file with the cleaned and reordered version.

`Sample.py`
- Generates `Final_CASE.csv` by sampling 400 rows from each participant's CSV in the `New_Folder` directory.
- For each CSV file:
  - Randomly samples 400 rows.
  - Appends a _subject_id_ column based on the file name for traceability.
- Combines all sampled data into one DataFrame.
- Saves the final combined dataset to `Final_CASE.csv`.

---
### Preprocessing for User Study Dataset
Preprocessing steps are confidential and not publicly shared. 
However, the model inference follows the same logic and uses the same FewShot and ZeroShot pipelines.

---
### Related Links
[CASE Dataset GitLab](https://gitlab.com/karan-shr/case_dataset)\
[OpenAI GPT-4 API](https://platform.openai.com/docs)

### Contact
Feel free to reach out if you want to collaborate or have questions!\
Author: [Chandani Jha]\
Email: [mscai2023.chandani@unigoa.ac.in]