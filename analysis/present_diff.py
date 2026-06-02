import os
import re
from pathlib import Path


# ===== 1. 生成所有输入差分 =====
w1 = [1 << i for i in range(64)]
w2 = [(1 << i) | (1 << j) for i in range(64) for j in range(i + 1, 64)]
all_diffs = w1 + w2

# ===== 2. 结果目录 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = str(PROJECT_ROOT / "saved_ea_txt" / "present")
ROUND = 14

# ===== 3. 正则：匹配路径1的总分 =====
path1_pattern = re.compile(
    r"---- 路径\s+1\s+\|\s+激活S盒:\d+\s+\|\s+总分:([\d\.]+)\s+----"
)

# ===== 4. 汇总表 =====
results = []

for diff in all_diffs:
    diff_hex = f"0x{diff:x}"
    filename = f"present_result_{ROUND}round_{diff_hex}.txt"
    filepath = os.path.join(RESULT_DIR, filename)

    if not os.path.exists(filepath):
        print(f"[WARN] 文件不存在: {filename}")
        continue

    min_score = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            m = path1_pattern.search(line)
            if m:
                min_score = float(m.group(1))
                break

    if min_score is None:
        print(f"[WARN] 未找到路径1总分: {filename}")
        continue

    results.append({
        "diff_hex": diff_hex,
        "diff_int": diff,
        "min_score": min_score
    })

# ===== 5. 按 diff 排序 =====
results.sort(key=lambda x: x["diff_int"])

# ===== 6. 输出为表格（CSV）=====
output_csv = PROJECT_ROOT / "present_14round_min_score_table.csv"
with open(output_csv, "w", encoding="utf-8") as f:
    f.write("diff_hex,diff_int,min_score\n")
    for r in results:
        f.write(f"{r['diff_hex']},{r['diff_int']},{r['min_score']}\n")

print(f"已完成，共收集 {len(results)} 条结果")
print(f"输出文件: {output_csv}")
