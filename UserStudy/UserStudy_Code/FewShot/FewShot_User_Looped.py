import openai
import pandas as pd
import asyncio
import aiohttp
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()  # load variables from .env

# Set OpenAI API Key
API_KEY = os.getenv("API_KEY")
openai.api_key = API_KEY

SEMAPHORE = asyncio.Semaphore(1)
MAX_RETRIES = 3
INPUT_FILE = "Final_UserStudy.csv"

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FILE = f"fewshot_log_{timestamp}.txt"
METRICS_CSV = f"fewshot_avg_metrics_{timestamp}.csv"
OUTPUT_FILE = f"fewshot_labeled_{timestamp}.csv"
TRAIN_SPLITS = [i / 10 for i in range(1, 10)]

# Prompt builder
def build_fewshot_prompt(train_rows, test_rows):
    header = (
        "You are an expert affective computing model trained to estimate valence and arousal"
        "based on physiological signal features.\n"
        " - Valence: 0 = Negative/Unpleasant, 1 = Positive/Pleasant\n"
        " - Arousal: 0 = Low Excitement/Calm, 1 = High Excitement/Alert\n\n"
        "Each sample represents a 5-second window of physiological sensor data from a user and contains:\n"
        " - mean_GSR: Mean Galvanic Skin Response\n"
        " - std_GSR: Standard Deviation of GSR\n"
        " - mean_HR: Mean Heart Rate\n"
        " - std_HR: Standard Deviation of HR\n"
        " - change_score: Measures how abruptly the user's physiological state has changed.\n"
        "   It score is computed using RuLSIF (Relative unconstrained Least Squares Importance Fitting),\n"
        "   which estimates the relative density ratio between two consecutive segments of data,\n"
        "   to quantify distributional change (i.e., sudden emotional transitions).\n\n"
        "Label each instance with Valence and Arousal based on the training examples below:\n"
    )

    examples = ""
    for row in train_rows.itertuples(index=False):
        examples += (
            f"mean_GSR: {getattr(row, 'mean_GSR', 0.0):.2f}, std_GSR: {getattr(row, 'std_GSR', 0.0):.2f}, "
            f"mean_HR: {getattr(row, 'mean_HR', 0.0):.2f}, std_HR: {getattr(row, 'std_HR', 0.0):.2f}, change_score: {getattr(row, 'Score', 0.0):.3f}"
            f" → Valence: {row.Valence}, Arousal: {row.Arousal}\n"
        )

    instructions = (
        "\nNow classify the following test samples based on the above training examples.\n"
        "Respond only in the following format (no extra text):\n"
        "1 → Valence,Arousal\n"
        "2 → Valence,Arousal\n"
        "...\n\n"
    )
    body = ""
    for i, row in enumerate(test_rows.itertuples(index=False), start=1):
        body += (
            f"{i}. mean_GSR: {row.mean_GSR:.2f}, std_GSR: {row.std_GSR:.2f}, "
            f"mean_HR: {row.mean_HR:.2f}, std_HR: {row.std_HR:.2f}, change_score: {row.Score:.3f}\n"
        )

    return header + examples + instructions + body.strip()

# Parse response
def parse_response(response_text, num_rows):
    lines = response_text.strip().splitlines()
    parsed = []
    for i in range(1, num_rows + 1):
        line = next((l for l in lines if l.startswith(f"{i} →")), None)
        if line:
            parts = line.split(" → ")[-1].split(",")
            if len(parts) == 2:
                parsed.append([p.strip() for p in parts])
            else:
                parsed.append(["3", "3"])
        else:
            parsed.append(["3", "3"])
    return parsed

# Async classifier
async def classify_user_split(session, train_rows, test_rows, attempt=1):
    prompt = build_fewshot_prompt(train_rows, test_rows)
    #print(f"#Prompt: {prompt}")
    print(f"Prompt tokens estimate: ~{len(prompt.split())}")

    async with SEMAPHORE:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": "gpt-4o-mini-2024-07-18",
                        "messages": [
                            {"role": "system", "content": "You are a machine learning model specialized in Affective Computing."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0,
                        "max_tokens": 2500
                    }
                ) as response:
                    data = await response.json()
                    parsed = parse_response(data["choices"][0]["message"]["content"], len(test_rows))
                    if any(p == ["3", "3"] for p in parsed):
                        if attempt < MAX_RETRIES:
                            print(f"Fallback '3,3' detected. Retrying attempt {attempt + 1}...")
                            return await classify_user_split(session, train_rows, test_rows, attempt + 1)
                        else:
                            print("Max retries exceeded. Returning fallback predictions.")
                    return parsed
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(2 ** attempt)
        return [["3", "3"]] * len(test_rows)

# Metric computation
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_pred)
    }

# Main function
async def main():
    df = pd.read_csv(INPUT_FILE)
    df.index.name = "Index"
    #print("#Dataframe with index: \n",df)

    os.makedirs("User_Study_Results", exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as log, open(METRICS_CSV, "w", newline='') as metric_csv:
        writer = csv.writer(metric_csv)
        writer.writerow(["Split", "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_F1", "Val_ROC_AUC",
                         "Aro_Accuracy", "Aro_Precision", "Aro_Recall", "Aro_F1", "Aro_ROC_AUC"])

        for split_frac in TRAIN_SPLITS:
            print(f"\n📊 Running for split {int(split_frac * 100)}%")
            all_val_preds, all_val_true = [], []
            all_aro_preds, all_aro_true = [], []

            async with aiohttp.ClientSession() as session:
                for user_id in df["P_id"].unique():
                    try:
                        print("----------------------------")
                        print(f"User ID: {user_id}")
                        user_data = df[df["P_id"] == user_id].reset_index(drop=True)
                        train_df = user_data.groupby("videoID", group_keys=False).apply(
                            lambda x: x.iloc[:int(len(x) * split_frac)]).reset_index(drop=True)
                        test_df = user_data.drop(index=train_df.index).reset_index()

                        preds = await classify_user_split(session, train_df, test_df, attempt=1)

                        out_df = pd.DataFrame(preds, columns=["Valence", "Arousal"], index=test_df["Index"])
                        out_df["True_Valence"] = test_df["Valence"].values
                        out_df["True_Arousal"] = test_df["Arousal"].values
                        out_df["User"] = user_id
                        out_df["Split"] = int(split_frac*100)

                        out_df.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE))

                        val_acc = accuracy_score(out_df["True_Valence"], out_df["Valence"].astype(int))
                        aro_acc = accuracy_score(out_df["True_Arousal"], out_df["Arousal"].astype(int))
                        line = f"User {user_id}, Split {int(split_frac * 100)}% - Valence Acc: {val_acc:.3f}, Arousal Acc: {aro_acc:.3f}"
                        print(line)
                        log.write(line + "\n")
                        log.flush()

                        all_val_preds.extend(out_df["Valence"].astype(int))
                        all_val_true.extend(out_df["True_Valence"])
                        all_aro_preds.extend(out_df["Arousal"].astype(int))
                        all_aro_true.extend(out_df["True_Arousal"])
                    except Exception as e:
                        print(f"Error processing user {user_id}: {e}")
                        log.write(f"Error processing user {user_id}: {e}\n")
                        log.flush()

            val_metrics = compute_metrics(all_val_true, all_val_preds)
            aro_metrics = compute_metrics(all_aro_true, all_aro_preds)
            to_write = f"\nSplit {split_frac * 100}% Summary:\nVal: {val_metrics}\nAro: {aro_metrics}\n"
            print(to_write)
            log.write(to_write)
            log.flush()
            writer.writerow([int(split_frac * 100)] + [
                val_metrics[k] for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
            ] + [
                aro_metrics[k] for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
            ])
            metric_csv.flush()
            print(f"✅ Logged metrics for split {int(split_frac * 100)}%")

    print("🏁 Completed all few-shot evaluations.")

if __name__ == "__main__":
    asyncio.run(main())
