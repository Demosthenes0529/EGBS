import os
import re

def extract_top1_scores(directory, speck, r):
    results = []

    # 正则匹配文件名中的 (a,b)
    filename_pattern = re.compile(
        rf"speck{speck}_result_ga_{r}round_(\d+).txt"
    )

    # filename_pattern = re.compile(
    #     rf"speck{speck}_result_ea_{r}round_\((\d+),(\d+)\)\.txt"
    # )

    # filename_pattern = re.compile(
    #     rf"speck{speck}_result_sub_ea_{r}round_\((\d+),(\d+)\)\.txt"
    # )

    # filename_pattern = re.compile(
    #     rf"speck{speck}_result_egbs_{r}round_\((\d+),(\d+)\)\.txt"
    # )

    # 正则匹配 路径1 总分
    score_pattern = re.compile(
        r"----\s*路径\s*1\s*\(总分:\s*([0-9.]+)\)"
    )
    # for filename in os.listdir(directory):
    #     file_match = filename_pattern.match(filename)
    #     if file_match:
    #         a = int(file_match.group(1))
    #         b = int(file_match.group(2))

    #         file_path = os.path.join(directory, filename)

    #         with open(file_path, 'r', encoding='utf-8') as f:
    #             content = f.read()

    #         score_match = score_pattern.search(content)
    #         if score_match:
    #             score = float(score_match.group(1))
    #             results.append(((a, b), score))

    # results.sort(key=lambda x: (x[0][0] != 0, x[0][0], x[0][1]))

    # return results
    for filename in os.listdir(directory):
        file_match = filename_pattern.match(filename)
        if file_match:
            a = int(file_match.group(1))


            file_path = os.path.join(directory, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            score_match = score_pattern.search(content)
            if score_match:
                score = float(score_match.group(1))
                results.append(((a, 0x0), score))

    # 按 (a,b) 排序，保证顺序一致
    results.sort(key=lambda x: (x[0][0], x[0][1]))

    return results

speck = 96
r = 14
# 使用示例
directory_path = f"./saved_ea_txt/speck{speck}"
top1_scores = extract_top1_scores(directory_path, speck=speck, r=r)

print(top1_scores)
