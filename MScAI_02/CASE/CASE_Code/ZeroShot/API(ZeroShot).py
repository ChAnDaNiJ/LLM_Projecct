import os
import openai
import asyncio
import aiohttp
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

load_dotenv()
API_KEY = os.getenv("API_KEY")
openai.api_key = API_KEY

SEMAPHORE = asyncio.Semaphore(1)
MAX_RETRIES = 5
OUTPUT_FILE = "ZeroShot_CASE_labeled.csv"
INPUT_FILE = "Final_CASE.csv"
ROWS_PER_PROMPT = 1500

def build_multi_row_prompt(rows: pd.DataFrame):
    header = (
        "You are a model that classifies valence and arousal (0 = Low, 1 = High) "
        "from physiological data. Classify each of the following rows based on the features below.\n\n"
    )
    instructions = (
        "Format your answer as:\n"
        "1 → Valence,Arousal\n"
        "2 → Valence,Arousal\n"
        "...\n\n"
    )

    body = ""
    for i, row in enumerate(rows.itertuples(index=False), start=1):
        body += (
            f"{i}. ECG: {row.ecg:.5f}\n"
            f"BVP: {row.bvp:.5f}\n"
            f"GSR: {row.gsr:.5f}\n"
            f"RSP: {row.rsp:.5f}\n"
            f"SKT: {row.skt:.5f}\n"
            f"EMG_zygo: {row.emg_zygo:.5f}\n"
            f"EMG_coru: {row.emg_coru:.5f}\n"
            f"EMG_trap: {row.emg_trap:.5f}\n"
        )
    return header + instructions + body.strip()

def parse_response(response_text, num_rows):
    lines = response_text.strip().splitlines()
    parsed = []
    for i in range(1, num_rows + 1):
        line = next((l for l in lines if l.startswith(f"{i} →")), None)
        if line:
            parts = line.split(" → ")[-1].split(',')
            if len(parts) == 2:
                try:
                    parsed.append([int(p.strip()) for p in parts])
                except ValueError:
                    parsed.append([3, 3])
            else:
                parsed.append([4, 4])
        else:
            parsed.append([5, 5])
    return parsed

def chunk_rows(df, size):
    for i in range(0, len(df), size):
        print(f"Processing rows from {i} to {min(i+size-1, len(df)-1)}")
        yield df.iloc[i:i+size]

async def classify_batch_async(session, rows):
    prompt = build_multi_row_prompt(rows)
    print(f"🔎 Estimated prompt size: {len(prompt.split())} tokens (approx)")

    async with SEMAPHORE:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": "gpt-4o-mini-2024-07-18",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a machine learning model specialized in Affective Computing."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0,
                        "max_tokens": 11000
                    }
                ) as response:

                    if response.status == 429:
                        wait_time = int(response.headers.get("Retry-After", 2 ** attempt))
                        print(f"❌ Rate limit hit (429). Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status != 200:
                        print(f"❌ API error {response.status}.")
                        raise RuntimeError("API error occurred!")

                    data = await response.json()
                    parsed = parse_response(data["choices"][0]["message"]["content"], len(rows))

                    if any(p == [3, 3] for p in parsed):
                        print("❌ Error: Received fallback '3,3'. Exiting.")
                        exit()

                    return parsed

            except Exception as e:
                print(f"⚠️ Error: {e}. Retrying in {2 ** attempt} seconds...")
                await asyncio.sleep(2 ** attempt)

        print("❌ Max retries exceeded. Exiting.")
        exit()

async def process_chunk_async(session, chunk, write_header=False):
    predictions = await classify_batch_async(session, chunk)
    df_pred = pd.DataFrame(predictions, columns=["Pred_Valence", "Pred_Arousal"], index=chunk.index)
    df_pred["True_Valence"] = chunk["Valence"].values
    df_pred["True_Arousal"] = chunk["Arousal"].values
    df_pred = df_pred[["True_Valence", "True_Arousal", "Pred_Valence", "Pred_Arousal"]]
    df_pred.to_csv(OUTPUT_FILE, index=True, header=write_header, mode='a')
    return df_pred

async def main():
    df = pd.read_csv(INPUT_FILE)
    print("📌 Columns loaded:", df.columns.tolist())  # Add this lin
    df = df.rename(columns={"Valence_class": "Valence", "Arousal_class": "Arousal"})
    df["mean_GSR"] = df["gsr"]
    df["std_GSR"] = 0.0
    df["mean_HR"] = df[["ecg", "bvp"]].mean(axis=1)
    df["std_HR"] = 0.0
    df["Score"] = df["valence"] + df["arousal"]
    df = df.reset_index(drop=True)
    df.index.name = "Index"

    with open(OUTPUT_FILE, "w") as f:
        f.write("Index,True_Valence,True_Arousal,Pred_Valence,Pred_Arousal\n")

    all_results = []
    async with aiohttp.ClientSession() as session:
        first_chunk = True
        for chunk in chunk_rows(df, ROWS_PER_PROMPT):
            df_chunk_result = await process_chunk_async(session, chunk, write_header=first_chunk)
            first_chunk = False
            all_results.append(df_chunk_result)

    final_df = pd.concat(all_results)
    print("✅ Zero-shot labels generated successfully!")

    true_valence = final_df["True_Valence"].astype(int)
    pred_valence = final_df["Pred_Valence"].astype(int)
    true_arousal = final_df["True_Arousal"].astype(int)
    pred_arousal = final_df["Pred_Arousal"].astype(int)

    metrics = {
        "Valence Accuracy": accuracy_score(true_valence, pred_valence),
        "Valence Precision": precision_score(true_valence, pred_valence, average='macro', zero_division=0),
        "Valence Recall": recall_score(true_valence, pred_valence, average='macro', zero_division=0),
        "Valence F1": f1_score(true_valence, pred_valence, average='macro', zero_division=0),
        "Valence ROC AUC": roc_auc_score(true_valence, pred_valence),
        "Arousal Accuracy": accuracy_score(true_arousal, pred_arousal),
        "Arousal Precision": precision_score(true_arousal, pred_arousal, average='macro', zero_division=0),
        "Arousal Recall": recall_score(true_arousal, pred_arousal, average='macro', zero_division=0),
        "Arousal F1": f1_score(true_arousal, pred_arousal, average='macro', zero_division=0),
        "Arousal ROC AUC": roc_auc_score(true_arousal, pred_arousal),
    }

    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    return metrics

if __name__ == "__main__":
    metrics = asyncio.run(main())
