import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_top1_scores(directory, speck, r):
    results = []

    filename_pattern = re.compile(
        rf"speck{speck}_result_ga_{r}round_(\d+).txt"
    )

    score_pattern = re.compile(
        r"----\s*路径\s*1\s*\(总分:\s*([0-9.]+)\)"
    )

    for filename in os.listdir(directory):
        file_match = filename_pattern.match(filename)
        if file_match:
            a = int(file_match.group(1))
            file_path = os.path.join(directory, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            score_match = score_pattern.search(content)
            if score_match:
                score = float(score_match.group(1))
                results.append(((a, 0x0), score))

    results.sort(key=lambda x: (x[0][0], x[0][1]))
    return results


speck = 96
r = 14
directory_path = str(PROJECT_ROOT / "saved_ea_txt" / f"speck{speck}")
top1_scores = extract_top1_scores(directory_path, speck=speck, r=r)

print(top1_scores)
