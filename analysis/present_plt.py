import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# 读取汇总表
df = pd.read_csv(BASE_DIR / "present_14round_min_score_table.csv")

df = df.sort_values("diff_int").reset_index(drop=True)

plt.figure(figsize=(14, 6))

plt.plot(df["min_score"], linewidth=1)

plt.axhline(64, linestyle="--")
plt.ylim(60, 130)

plt.xlabel("Input difference index")
plt.ylabel("Minimum total score (-log2 P)")
plt.title("PRESENT 14-round Differential Search Results")

plt.tight_layout()
plt.show()
plt.savefig(BASE_DIR / "result.png", dpi=300)
