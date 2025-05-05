import os
import numpy as np
import logging
import asyncio
from ZeroShot_User import main as zero_shot_main
from datetime import datetime

# Ensure the logging directory exists
os.makedirs("User_Study_Results", exist_ok=True)

# Timestamped log filename to avoid overwrites
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"ZeroShot_User_Log_{timestamp}.txt"                                                         #log_path = f"UserStudy_Results/ZeroShot_User_Log_{timestamp}.txt"

# Setup logging
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
print(f"✅ Logging setup complete. Log file: {log_path}")

all_metrics = []

for run in range(5):
    print(f"\n🚀 Starting Run {run + 1}...\n")
    logging.info(f"========== Run {run + 1} ==========")

    try:
        metrics = asyncio.run(zero_shot_main())  # Get metrics directly ✅

        all_metrics.append(metrics)

        logging.info("✅ Metrics:")
        for k, v in metrics.items():
            #print(f"   {k}: {v:.3f}")
            logging.info(f"{k}: {v:.3f}")

    except Exception as e:
        print(f"❌ Exception occurred in Run {run + 1}: {e}")
        logging.error(f"Exception in Run {run + 1}: {e}")

# Compute and log mean + std
print("\n🧮 Computing average and standard deviation of metrics...")
if all_metrics:
    keys = all_metrics[0].keys()
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in keys}
    std = {k: np.std([m[k] for m in all_metrics]) for k in keys}

    summary = "\n\n====== 📊 Final Summary over 5 Runs ======\n"
    summary += "\nAverage Metrics:\n"
    summary += "\n".join(f"{k}: {v:.3f}" for k, v in avg.items())
    summary += "\n\nStandard Deviation:\n"
    summary += "\n".join(f"{k}: {v:.3f}" for k, v in std.items())

    print(summary)
    logging.info(summary)
else:
    print("❌ No metrics collected in any run.")
    logging.error("❌ No metrics collected in any run.")
