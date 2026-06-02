import os
import re
from collections import defaultdict
from math import pow, log2
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def collect_output_diff_probabilities(diff_list, round_count, base_path=None):
    if base_path is None:
        base_path = str(PROJECT_ROOT / "saved_ea_txt" / "present")

    results = {}

    for diff in tqdm(diff_list, desc="Processing input differences", unit="diff"):
        diff_hex = f"0x{diff:x}"
        filepath = os.path.join(base_path, f"present_result_{round_count}round_{diff_hex}.txt")
        if not os.path.exists(filepath):
            print(f"[WARN] 文件不存在: present_result_{round_count}round_{diff_hex}.txt")
            continue

        out_prob = defaultdict(float)

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        path_blocks = content.split("---- 路径")[1:]

        for block in path_blocks:
            score_match = re.search(r"总分:\s*([\d.]+)", block)
            if not score_match:
                continue
            score = float(score_match.group(1))

            layers = re.findall(r"层\d+:\s*(0x[0-9a-fA-F]+)", block)
            if not layers:
                continue
            out_diff = layers[-1]
            out_prob[out_diff] += pow(2, -score)

        results[hex(diff)] = dict(out_prob)

    return results


diffs = [0x1001]
ROUND = 15
BASE_PATH = str(PROJECT_ROOT / "saved_ea_txt" / "present")

res = collect_output_diff_probabilities(diffs, ROUND, base_path=BASE_PATH)

valid_diffs = []
for diff_hex, out_map in res.items():
    for out_diff, prob in out_map.items():
        if prob <= 0:
            continue
        neg_log = -log2(prob)
        if neg_log < 64:
            valid_diffs.append(diff_hex)
            break

seen = set()
valid_unique = []
for d in valid_diffs:
    if d not in seen:
        seen.add(d)
        valid_unique.append(d)

valid_129 = valid_unique[:129]
valid_output_path = os.path.join(BASE_PATH, f"valid_diffs_{ROUND}round_129.txt")
os.makedirs(BASE_PATH, exist_ok=True)
with open(valid_output_path, "w", encoding="utf-8") as vf:
    vf.write("[" + ", ".join(valid_129) + "]\n")

print(f"Saved {len(valid_129)} valid diffs to {valid_output_path}")

output_path = os.path.join(BASE_PATH, f"present_count_diff_{ROUND}round_results.txt")
os.makedirs(BASE_PATH, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as out_f:
    for diff_hex, out_map in res.items():
        filtered = []
        for out_diff, prob in out_map.items():
            if prob <= 0:
                continue
            neg_log = -log2(prob)
            if neg_log < 64:
                filtered.append((out_diff, prob, neg_log))

        if not filtered:
            continue

        filtered.sort(key=lambda x: x[1], reverse=True)
        line1 = f"\nInput diff = {diff_hex}"
        line2 = f"Number of output diffs = {len(out_map)}"
        print(line1)
        print(line2)
        out_f.write(line1 + "\n")
        out_f.write(line2 + "\n")

        for out_diff, prob, neg_log in filtered:
            s = f"  {out_diff:>18s}  P = 2^(-{neg_log:.4f})"
            print(s)
            out_f.write(s + "\n")
