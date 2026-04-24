import os
import re
from collections import defaultdict
from math import pow, log2
from tqdm import tqdm

def collect_output_diff_probabilities(diff_list, ROUND, base_path="./saved_ea_txt/present"):
    results = {}

    for diff in tqdm(diff_list, desc="Processing input differences", unit="diff"):
        diff_hex = f"0x{diff:x}"
        filename = f"present_result_{ROUND}round_{diff_hex}.txt"
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            print(f"[WARN] 文件不存在: {filename}")
            continue
        diff_hex = hex(diff)
        filename = f"{base_path}/present_15round_0x1001_to_0x50000000500.txt"

        out_prob = defaultdict(float)

        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # 以“---- 路径”切分
        path_blocks = content.split("---- 路径")[1:]

        for block in path_blocks:
            # 1️⃣ 提取路径总分
            m_match = re.search(r"总分:\s*([\d.]+)", block)
            if not m_match:
                continue
            m = float(m_match.group(1))

            # 2️⃣ 提取最后一层的输出差分
            layers = re.findall(r"层\d+:\s*(0x[0-9a-fA-F]+)", block)
            if not layers:
                continue
            out_diff = layers[-1]   # 最后一层

            # 3️⃣ 概率累加
            out_prob[out_diff] += pow(2, -m)

        results[diff_hex] = dict(out_prob)

    return results


# diffs = [
#     0x1001, 0x2002, 0x2004, 0x4002, 0x4004,
#     0x10010000, 0x20020000, 0x20040000, 0x40020000, 0x40040000,
#     0x100100000000, 0x200200000000, 0x200400000000,
#     0x400200000000, 0x400400000000
# ]
# w1 = [1 << i for i in range(64)]
# w2 = [(1 << i) | (1 << j) for i in range(64) for j in range(i + 1, 64)]
# diffs = [0x3, 0x5, 0x9, 0x11, 0x21, 0x41, 0x81, 0x101, 0x201, 0x401, 0x801, 0x1001, 0x2001, 0x4001, 0x8001, 0x12, 0x22, 0x42, 0x102, 0x202, 0x402, 0x1002, 0x2002, 0x4002, 0x8002, 0x14, 0x24, 0x44, 0x104, 0x204, 0x404, 0x1004, 0x2004, 0x4004, 0x8004, 0x18, 0x108, 0x1008, 0x2008, 0x4008, 0x8008, 0x110, 0x1010, 0x220, 0x420, 0x2020, 0x4020, 0x240, 0x440, 0x2040, 0x4040, 0x300, 0x900, 0x1100, 0x2200, 0x4200, 0x2400, 0x4400, 0x3000, 0x5000, 0x9000, 0x30000, 0x50000, 0x110000, 0x810000, 0x1010000, 0x8010000, 0x10010000, 0x20010000, 0x40010000, 0x80010000, 0x220000, 0x420000, 0x2020000, 0x4020000, 0x10020000, 0x20020000, 0x40020000, 0x240000, 0x440000, 0x2040000, 0x4040000, 0x10040000, 0x20040000, 0x40040000, 0x180000, 0x1080000, 0x10080000, 0x80080000, 0x1100000, 0x10100000, 0x3000000, 0x11000000, 0x30000000, 0x1100000000, 0x8100000000, 0x10100000000, 0x80100000000, 0x100100000000, 0x200100000000, 0x400100000000, 0x800100000000, 0x2200000000, 0x4200000000, 0x20200000000, 0x40200000000, 0x100200000000, 0x200200000000, 0x400200000000, 0x2400000000, 0x4400000000, 0x20400000000, 0x40400000000, 0x100400000000, 0x200400000000, 0x400400000000, 0x1800000000, 0x10800000000, 0x100800000000, 0x800800000000, 0x11000000000, 0x101000000000, 0x110000000000, 0x9000000000000, 0x1001000000000000, 0x2002000000000000, 0x4002000000000000, 0x2004000000000000, 0x4004000000000000]
diffs = [0x1001]

ROUND = 15

# 输出保存目录（与 collect_output_diff_probabilities 的默认 base_path 保持一致）
BASE_PATH = "./saved_ea_txt/present"

res = collect_output_diff_probabilities(diffs, ROUND, base_path=BASE_PATH)

# 收集满足条件的合法差分（存在输出且有 prob>0 且 -log2(prob) < 60），并保存前 129 个
valid_diffs = []
for diff_hex, out_map in res.items():
    for out_diff, prob in out_map.items():
        if prob <= 0:
            continue
        neg_log = -log2(prob)
        if neg_log < 64:
            valid_diffs.append(diff_hex)
            break

# 保持出现顺序并去重
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

# for diff_hex, mp in res.items():
#     total_prob = sum(mp.values())
#     print(diff_hex, len(mp), -log2(total_prob))


output_path = os.path.join(BASE_PATH, f"present_count_diff_{ROUND}round_results.txt")
os.makedirs(BASE_PATH, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as out_f:
    # n = 0
    for diff_hex, out_map in res.items():
        # 先过滤出满足条件的输出差分（prob>0 且 neg_log < 60）
        filtered = []
        for out_diff, prob in out_map.items():
            if prob <= 0:
                continue
            neg_log = -log2(prob)
            if neg_log < 64:
                filtered.append((out_diff, prob, neg_log))

        # 如果没有符合条件的差分，则跳过该输入差分，不输出 header 行
        if not filtered:
            continue
        # n = n + 1
        # 按概率降序排序并写 header
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
        # print(n)

