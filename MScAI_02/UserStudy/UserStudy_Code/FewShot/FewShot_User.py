import openai
import pandas as pd
import asyncio
import aiohttp
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os

load_dotenv()  # load variables from .env

# Set OpenAI API Key
API_KEY = os.getenv("API_KEY")
openai.api_key = API_KEY

SEMAPHORE = asyncio.Semaphore(1)
MAX_RETRIES = 3
OUTPUT_FILE = "userwise_fewshot_labeled_401.csv"
INPUT_FILE = "Final_UserStudy.csv"
LOG_FILE = "few_shot_accuracy_log_401.txt"

TRAIN_SPLITS = [0.4]  # Run only 10% first

# Prompt builder
def build_fewshot_prompt(train_rows, test_rows):
    header = (
        "You are a model that classifies valence and arousal (0 = Low, 1 = High)\n"
        "Given physiological features, label Valence and Arousal.\n"
        "Use the training examples below:\n"
    )
    examples = ""
    for row in train_rows.itertuples(index=False):
        examples += (
            f"mean_GSR: {getattr(row, 'mean_GSR', 0.0):.2f}, std_GSR: {getattr(row, 'std_GSR', 0.0):.2f}, "
            f"mean_HR: {getattr(row, 'mean_HR', 0.0):.2f}, std_HR: {getattr(row, 'std_HR', 0.0):.2f}, change_score: {getattr(row, 'change_score', 0.0):.3f}"
            f" → Valence: {row.Valence}, Arousal: {row.Arousal}\n"
        )

    instructions = ("Now classify the following:\n"
                   "Format your answer as:\n"
                    "1 → Valence,Arousal\n"
                    "2 → Valence,Arousal\n"
                    "...\n\n")
    body = ""
    for i, row in enumerate(test_rows.itertuples(index=False), start=1):
        row_vals = {
            'mean_GSR': getattr(row, 'mean_GSR', 0.0),
            'std_GSR': getattr(row, 'std_GSR', 0.0),
            'mean_HR': getattr(row, 'mean_HR', 0.0),
            'std_HR': getattr(row, 'std_HR', 0.0),
            'change_score': getattr(row, 'change_score', 0.0)
        }
        body += (
            f"{i}. mean_GSR: {row_vals['mean_GSR']:.2f}, std_GSR: {row_vals['std_GSR']:.2f}, "
            f"mean_HR: {row_vals['mean_HR']:.2f}, std_HR: {row_vals['std_HR']:.2f}, change_score: {row_vals['change_score']:.3f}\n"
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

# Classify async
async def classify_user_split(session, train_rows, test_rows, attempt=1):
    prompt = build_fewshot_prompt(train_rows, test_rows)
    print(f"🧠 Prompt tokens: ~{len(prompt.split())}")

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
                        "max_tokens": 11000
                    }
                ) as response:
                    data = await response.json()
                    parsed = parse_response(data["choices"][0]["message"]["content"], len(test_rows))
                    if any(p == ["3", "3"] for p in parsed):
                        if attempt < MAX_RETRIES:
                            print(f"⚠️ Received fallback. Retrying attempt {attempt + 1}...")
                            return await classify_user_split(session, train_rows, test_rows, attempt+1)
                        print("Incorrect response:\n", parsed)
                        print("❌ Error: Received fallback '3,3'. Exiting.")
                        exit()
                    return parsed

            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(2 ** attempt)
        return [["3", "3"]] * len(test_rows)

# Main driver
async def main():
    df = pd.read_csv(INPUT_FILE)
    df.index.name = "Index"
    all_val_preds, all_val_true = [], []
    all_aro_preds, all_aro_true = [], []

    with open(OUTPUT_FILE, "w") as f:
        f.write("Index,Valence,Arousal,True_Valence,True_Arousal,User,Split\n")

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        async with aiohttp.ClientSession() as session:
            for user_id in df["P_id"].unique():
                user_data = df[df["P_id"] == user_id].reset_index(drop=True)

                for split_frac in TRAIN_SPLITS:
                    train_df = user_data.groupby("videoID", group_keys=False).apply(lambda x: x.iloc[:int(len(x) * split_frac)], include_groups=False).reset_index(drop=True)
                    test_df = user_data.drop(index=train_df.index).reset_index()

                    preds = await classify_user_split(session, train_df, test_df, attempt=1)

                    out_df = pd.DataFrame(preds, columns=["Valence", "Arousal"], index=test_df["Index"])
                    out_df["True_Valence"] = test_df["Valence"].values
                    out_df["True_Arousal"] = test_df["Arousal"].values
                    out_df["User"] = user_id
                    out_df["Split"] = int(split_frac * 100)

                    out_df.to_csv(OUTPUT_FILE, mode='a', header=False)

                    acc_val = accuracy_score(out_df["True_Valence"], out_df["Valence"].astype(int))
                    acc_aro = accuracy_score(out_df["True_Arousal"], out_df["Arousal"].astype(int))
                    line = f"User {user_id}, Split {int(split_frac*100)}% → Valence Acc: {acc_val:.3f}, Arousal Acc: {acc_aro:.3f}\n"
                    print("📊 " + line.strip())
                    log.write(line)

                    all_val_preds.extend(out_df["Valence"].astype(int).tolist())
                    all_val_true.extend(out_df["True_Valence"].tolist())
                    all_aro_preds.extend(out_df["Arousal"].astype(int).tolist())
                    all_aro_true.extend(out_df["True_Arousal"].tolist())

        overall_val_acc = accuracy_score(all_val_true, all_val_preds)
        overall_aro_acc = accuracy_score(all_aro_true, all_aro_preds)
        final_line = f"\nOverall Accuracy → Valence: {overall_val_acc:.3f}, Arousal: {overall_aro_acc:.3f}\n"
        print(final_line)
        log.write(final_line)

    print("✅ User-wise few-shot labels generated!")

asyncio.run(main())
