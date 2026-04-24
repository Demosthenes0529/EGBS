import math
from itertools import product, combinations
from functools import lru_cache
import numpy as np
from operator import itemgetter
import pickle
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
import time

raw_P = [0,  16, 32, 48, 1,  17, 33, 49, 2,  18, 34, 50, 3,  19, 35, 51,
     4,  20, 36, 52, 5,  21, 37, 53, 6,  22, 38, 54, 7,  23, 39, 55,
     8,  24, 40, 56, 9,  25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
     12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]
raw_P_inv = [0] * 64
for i, p in enumerate(raw_P):
    raw_P_inv[p] = i

# PRESENT S-box
S = [0xC,0x5,0x6,0xB,0x9,0x0,0xA,0xD,0x3,0xE,0xF,0x8,0x4,0x7,0x1,0x2]

# build DDT counts and probs
DDT_counts = [[0]*16 for _ in range(16)]
for a in range(16):
    for x in range(16):
        beta = S[x] ^ S[x ^ a]
        DDT_counts[a][beta] += 1
DDT_prob = [[c/16.0 for c in row] for row in DDT_counts]

def present_p_layer(x: int) -> int:
    y = 0
    for i in range(64):
        y |= ((x >> i) & 1) << raw_P[i]
    return y

def present_p_layer_inv(x: int) -> int:
    y = 0
    for i in range(64):
        y |= ((x >> i) & 1) << raw_P_inv[i]
    return y

def count_active_sboxes(x: int) -> int:
    """
    Count how many PRESENT S-boxes are active in the 64-bit input x.
    S-box is active if the 4-bit nibble != 0.
    """
    cnt = 0
    for i in range(16):           # 16 nibbles for 16 S-boxes
        if (x >> (i*4)) & 0xF:    # nibble != 0 means active
            cnt += 1
    return cnt

def int_to_nibbles_int(x: int):
    """convert 64-bit integer diff to 16 nibbles list"""
    hex_16 = f"{x:016X}"
    return [int(ch, 16) for ch in hex_16]

def nibbles_to_int(nibbles):
    return int("".join(f"{x:X}" for x in nibbles), 16)

def compute_neglog2_transition(p_diff:int, child_p_afterP:int) -> float:
    """
    计算从 p_diff (即S盒层输入的64-bit) 到 child（P之后的64-bit）这一单步的 -log2 概率。
    注意：delta_optimal_output_diff 给出的 child 是 P-layer 后的差分，我们先取逆 P 得到 S-box 层的输出，
    然后逐 nibble 用 DDT_prob 计算 -log2 概率的和。
    返回 float（若有不可能项返回 math.inf）。
    """
    # 把 child 逆 P 得到 S-box 层输出（每 nibble 的输出差分 b_i）
    child_preS = present_p_layer_inv(child_p_afterP)
    a_nibbles = int_to_nibbles_int(p_diff)        # 输入差分的每 nibble a_i
    b_nibbles = int_to_nibbles_int(child_preS)    # S-盒输出差分 b_i
    total_neglog2 = 0.0
    for a, b in zip(a_nibbles, b_nibbles):
        # DDT_prob stored as counts/16 or probabilities? In our earlier code DDT_prob[a][b] = count/16.0
        prob = DDT_prob[a][b]
        if prob <= 0:
            return math.inf
        total_neglog2 += -math.log2(prob)
    return total_neglog2

def best_output_diff(p_diff):
    """
    输入：p_diff（整数）
    输出：按激活 S 盒数量升序排序后的 best_cipher（整数列表）
    """
    p_nibbles = int_to_nibbles_int(p_diff)
    # 1. 获取每个 nibble 的最佳输出差分
    best_list_per_pos = []
    for a in p_nibbles:
        row = DDT_prob[a]
        max_prob = max(row)
        best_outputs = [b for b in range(16) if row[b] == max_prob]
        best_list_per_pos.append(best_outputs)
    # 2. 根据所有最佳 nibble 组合生成所有 c_diff
    best_results = [nibbles_to_int(combo) for combo in product(*best_list_per_pos)]
    # 3. 对每个 c_diff 进行 P-layer
    best_cipher = [present_p_layer(i) for i in best_results]
    # 4. 根据激活 S 盒数量排序（升序）
    best_cipher_sorted = sorted(best_cipher, key=count_active_sboxes)
    return best_cipher_sorted

def delta_optimal_output_diff(p_diff: int, delta: int):
    """
    输入：
        p_diff : 输入差分 (整数)
        delta  : 允许增加的 -log2 P 范围（整数）
    输出：
        best_cipher_sorted : 满足条件的所有输出差分（整数数组）
    """
    p_nibbles = int_to_nibbles_int(p_diff)
    # 对每个 nibble 获取：
    # best_outputs: 最高概率输出
    # second_outputs: 第二高概率输出
    best_list = []
    second_list = []
    best_log_list = []
    diff_log_list = []
    for a in p_nibbles:
        row = DDT_prob[a]
        # 找最高 & 次高概率
        max_p = max(row)
        best_out = [b for b in range(16) if row[b] == max_p]
        # 次高概率（排除 max）
        remain = [v for v in row if v != max_p]
        second_p = max(remain) if remain else 0.0
        second_out = [b for b in range(16) if row[b] == second_p]
        best_list.append(best_out)
        second_list.append(second_out)
        best_log = 0 if max_p == 1 else -math.log(max_p, 2)
        second_log = 0 if second_p == 1 else -math.log(second_p, 2) if second_p > 0 else float("inf")
        diff_log = second_log - best_log
        best_log_list.append(best_log)
        diff_log_list.append(diff_log)
    # 最佳总 log
    best_total_log = sum(best_log_list)
    # 允许的最大 log
    limit_log = best_total_log + delta
    # === 构造所有近似输出差分 ===
    results = set()
    # 枚举要使用次佳的 nibbles 数量：最多 16 个
    for k in range(17):  # 从 0 到 16
        for idxs in combinations(range(16), k):
            # 如果这些 idx 的 diff_log 加起来超过 delta，则跳过
            cost = sum(diff_log_list[i] for i in idxs)
            if cost > delta:
                continue
            # 构造每个位置的取值集合
            choices = []
            for i in range(16):
                if i in idxs:
                    choices.append(second_list[i])    # 改为次高
                else:
                    choices.append(best_list[i])      # 使用最高
            # 多重组合生成所有 nibble 输出
            for combo in product(*choices):
                out_int = nibbles_to_int(combo)
                results.add(out_int)
    # === P-layer 并排序 ===
    ciphers = [(present_p_layer(r), r) for r in results]
    # 排序：激活 S 盒数 → 最小 -log2P
    ciphers_sorted = sorted(ciphers, key=lambda x: (count_active_sboxes(x[0]),  ))
    # 最终输出仅 ciphertext（P-layer 后的整数）
    return [c[0] for c in ciphers_sorted]

def evaluate_path_present(path):
    """
    类似 SPECK 的 evaluate_path_add：路径内每一步的 score = -log2( transition_prob )
    path 是一条按层排列的差分序列：例如 [d0, d1, d2, ...], 其中 d_i 为该层的 64-bit 差分（P-layer 后）
    我们将对每对相邻节点 path[i] -> path[i+1] 计算 -log2P（把 path[i] 当作 S-box 层输入，
    path[i+1] 是下层 P-layer 后的值），累加返回总 score（越小越好）。
    """
    fitness_score = 0.0
    total_active = 0
    # 累加每一层激活 S 盒数
    for node in path:
        total_active += count_active_sboxes(node)
    # 概率累积
    for i in range(len(path) - 1):
        parent = path[i]
        child_afterP = path[i+1]
        score = compute_neglog2_transition(parent, child_afterP)
        fitness_score += score
    return fitness_score, total_active

def first_best_search_present(root_diff:int, nr:int, tree:dict, top1=20, top2=10, delta=2, t1=10, t2=22, expand_threshold=None, population_size=2000, verbose=True):
    root = root_diff
    population = [[root]]
    current_paths = [[root]]
    if verbose:
        print(f"=== 开始 PRESENT 分层束搜索 (总层数: {nr}) ===")
    for layer in range(nr):
        if verbose:
            print(f"\n--- 正在处理第 {layer+1} 层 (当前路径数: {len(current_paths)}) ---")
        next_paths = []
        expanded_nodes = 0
        for path_idx, path in enumerate(current_paths):
            last_node = path[-1]
            if verbose and (path_idx % 10 == 0):
                print(f"  处理进度: {path_idx}/{len(current_paths)} 条路径", end="\r")
            # 未展开则生成
            if last_node not in tree or len(tree[last_node]) == 0:
                cur_active = count_active_sboxes(last_node)
                if cur_active <= t1:
                    candidates = delta_optimal_output_diff(last_node, delta)
                    cand_scores = []
                    for c in candidates:
                        neglog2 = compute_neglog2_transition(last_node, c)
                        child_active = count_active_sboxes(c)
                        if (cur_active + child_active) > t2:
                            continue
                        cand_scores.append((c, child_active, neglog2))
                    cand_scores.sort(key=lambda x: (x[1], x[2]))
                    # print(cand_scores)
                    top_children = [c for (c, _, _) in cand_scores[:top1]]
                    tree.setdefault(last_node, [])
                    for child in top_children:
                        tree[last_node].append(child)
                        expanded_nodes += 1
            # 扩展 top2 子节点
            valid_children = tree.get(last_node, [])
            top = min(len(valid_children), top2)
            for child in valid_children[:top]:
                new_path = path + [child]
                next_paths.append(new_path)
        if verbose:
            print(f"\n  本层扩展了 {expanded_nodes} 个新节点")
        population.append(next_paths)
        current_paths = next_paths
        if layer >= 3 and len(current_paths) > population_size:
            evaluated = []
            for p in current_paths:
                score, active = evaluate_path_present(p)
                evaluated.append(((active, score), p))
            evaluated.sort(key=lambda x: x[0])  # (active, score) 排序
            current_paths = [path for (_, path) in evaluated[:population_size]]
            if verbose:
                best_active, best_score = evaluated[0][0]
                print(f"  选择完成: 保留前{population_size} 条最优路径 | 激活S盒:{best_active}, 分数:{best_score:.4f}")
    # ============ 最终排序 ============
    final_evaluated = []
    for p in current_paths:
        score, active = evaluate_path_present(p)
        final_evaluated.append(((active, score), p))
    final_evaluated.sort(key=lambda x: x[0])  # (激活S盒, 概率)
    sorted_paths = [p for (_, p) in final_evaluated]
    # ============ 保存到文件 ============
    save_path = f"./saved_ea_txt/present/present_result_{nr}round_{hex(root_diff)}.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== PRESENT 差分束搜索结果 ===\n")
        f.write(f"总轮数: {nr}\n")
        f.write(f"最终路径数: {len(final_evaluated)}\n\n")
        for idx, (info, path) in enumerate(final_evaluated):
            active_total, total_score = info
            f.write(f"\n---- 路径 {idx+1} | 激活S盒总数: {active_total} | 总分: {total_score:.4f} ----\n")
            for layer in range(len(path)):
                node = path[layer]
                active = count_active_sboxes(node)
                line = f"  层{layer}: {hex(node)} | 激活S盒:{active}"
                if layer > 0:
                    prev_node = path[layer - 1]
                    step_score = compute_neglog2_transition(prev_node, node)
                    line += f" | 层分: {step_score:.4f}"
                f.write(line + "\n")
    return sorted_paths, tree

def worker_expand_present(args):
    last_node, delta, t1, t2, expand_threshold = args
    cur_active = count_active_sboxes(last_node)
    if cur_active > t1:
        return (last_node, [])
    candidates = delta_optimal_output_diff(last_node, delta)
    cand_scores = []
    for c in candidates:
        neglog2 = compute_neglog2_transition(last_node, c)
        if expand_threshold is not None:
            prob = 2 ** (-neglog2) if neglog2 != float("inf") else 0.0
            if prob < expand_threshold:
                continue
        child_active = count_active_sboxes(c)
        if cur_active + child_active > t2:
            continue
        cand_scores.append((c, child_active, neglog2))
    cand_scores.sort(key=lambda x: (x[1], x[2]))
    top_children = [c for (c, _, _) in cand_scores]
    return (last_node, top_children)

def worker_eval_present(path):
    score, active = evaluate_path_present(path)
    return ((active, score), path)

def first_best_search_present_parallel(root_diff:int, nr:int, tree:dict, top1=20, top2=10, delta=2, t1=10, t2=22, expand_threshold=None, population_size=2000, verbose=True, n_jobs=32):
    root = root_diff
    current_paths = [[root]]
    if verbose:
        print(f"=== 开始 PRESENT 分层束搜索 (总层数: {nr}) ===")
    for layer in range(nr):
        if verbose:
            print(f"\n--- 第 {layer+1} 层 ---")
        next_paths = []
        expanded_nodes = 0
        # ========= 1) 找出需要展开的节点 =========
        need_expand = []
        for path in current_paths:
            last_node = path[-1]
            if last_node not in tree or len(tree[last_node]) == 0:
                need_expand.append(last_node)
        # ========= 2) 仅保留一个进度条：Expand =========
        if need_expand:
            tasks = [(node, delta, t1, t2, expand_threshold) for node in need_expand]
            with mp.Pool(processes=n_jobs) as pool:
                results = list(tqdm(pool.imap(worker_expand_present, tasks), total=len(tasks), desc=f"[Layer {layer+1}] Expand"))
            for last_node, children in results:
                if last_node not in tree:
                    tree[last_node] = []
                # 取 top1
                children = children[:top1]
                for c in children:
                    tree[last_node].append(c)
                    expanded_nodes += 1
        if verbose:
            print(f"  已扩展节点：{expanded_nodes}")
        # ========= 3) 扩展路径（不使用 tqdm） =========
        for path in current_paths:
            last_node = path[-1]
            valid_children = tree.get(last_node, [])
            top = min(len(valid_children), top2)
            for c in valid_children[:top]:
                next_paths.append(path + [c])
        current_paths = next_paths
        # ========= 4) 剪枝（不用 tqdm） =========
        if layer >= 3 and len(current_paths) > population_size:
            if verbose:
                print(f"  路径数 {len(current_paths)} > {population_size}，进行剪枝…")
            with mp.Pool(processes=n_jobs) as pool:
                evaluated = pool.map(worker_eval_present, current_paths)
            evaluated.sort(key=lambda x: x[0])
            current_paths = [p for (_, p) in evaluated[:population_size]]
            if verbose:
                best_active, best_score = evaluated[0][0]
                print(f"  剪枝后最优路径：激活S盒:{best_active} | 分数:{best_score:.4f}")
    # ========= 最终排序 =========
    with mp.Pool(processes=n_jobs) as pool:
        final_evaluated = pool.map(worker_eval_present, current_paths)
    final_evaluated.sort(key=lambda x: x[0])
    sorted_paths = [p for (_, p) in final_evaluated]
    # ========= 保存输出 =========
    save_path = f"./saved_ea_txt/present/present_result_{nr}round_{hex(root_diff)}.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== PRESENT 差分束搜索结果 ===\n")
        f.write(f"总轮数: {nr}\n")
        f.write(f"最终路径数: {len(final_evaluated)}\n\n")
        for idx, (info, path) in enumerate(final_evaluated):
            active_total, total_score = info
            f.write(f"\n---- 路径 {idx+1} | 激活S盒:{active_total} | 总分:{total_score:.4f} ----\n")
            for layer in range(len(path)):
                node = path[layer]
                active = count_active_sboxes(node)
                line = f"  层{layer}: {hex(node)} | 激活S盒:{active}"
                if layer > 0:
                    prev_node = path[layer - 1]
                    step_score = compute_neglog2_transition(prev_node, node)
                    line += f" | 层分:{step_score:.4f}"
                f.write(line + "\n")
    return sorted_paths, tree

# w1 = [0x3, 0x5, 0x9, 0x11, 0x21, 0x41, 0x81, 0x101, 0x201, 0x401, 0x801, 0x1001, 0x2001, 0x4001, 0x8001, 0x12, 0x22, 0x42, 0x102, 0x202, 0x402, 0x1002, 0x2002, 0x4002, 0x8002, 0x14, 0x24, 0x44, 0x104, 0x204, 0x404, 0x1004, 0x2004, 0x4004, 0x8004, 0x18, 0x108, 0x1008, 0x2008, 0x4008, 0x8008, 0x110, 0x1010, 0x220, 0x420, 0x2020, 0x4020, 0x240, 0x440, 0x2040, 0x4040, 0x300, 0x900, 0x1100, 0x2200, 0x4200, 0x2400, 0x4400, 0x3000, 0x5000, 0x9000, 0x30000, 0x50000, 0x110000, 0x810000, 0x1010000, 0x8010000, 0x10010000, 0x20010000, 0x40010000, 0x80010000, 0x220000, 0x420000, 0x2020000, 0x4020000, 0x10020000, 0x20020000, 0x40020000, 0x240000, 0x440000, 0x2040000, 0x4040000, 0x10040000, 0x20040000, 0x40040000, 0x180000, 0x1080000, 0x10080000, 0x80080000, 0x1100000, 0x10100000, 0x3000000, 0x11000000, 0x30000000, 0x1100000000, 0x8100000000, 0x10100000000, 0x80100000000, 0x100100000000, 0x200100000000, 0x400100000000, 0x800100000000, 0x2200000000, 0x4200000000, 0x20200000000, 0x40200000000, 0x100200000000, 0x200200000000, 0x400200000000, 0x2400000000, 0x4400000000, 0x20400000000, 0x40400000000, 0x100400000000, 0x200400000000, 0x400400000000, 0x1800000000, 0x10800000000, 0x100800000000, 0x800800000000, 0x11000000000, 0x101000000000, 0x110000000000, 0x9000000000000, 0x1001000000000000, 0x2002000000000000, 0x4002000000000000, 0x2004000000000000, 0x4004000000000000]
w1 = [0x1001]
for diff in w1:
    print(diff)
    tree = defaultdict(list)
    tree[(diff)] = []  
    t1 = time.time()
    current_paths, tree = first_best_search_present_parallel(diff, nr=14, tree=tree, top1=10, top2=10, delta=2, t1=6, t2=22, population_size=2000, verbose=True, n_jobs=16)
    t2 = time.time()
    print(f"搜索完成！总耗时: {t2 - t1:.2f} 秒")







# best_results = best_output_diff(diff)
# print([hex(i) for i in best_results])
# print([count_active_sboxes(i) for i in best_results])
# best_results = delta_optimal_output_diff(diff, 4)
# print([hex(i) for i in best_results])
# print([count_active_sboxes(i) for i in best_results])
# print([compute_neglog2_transition(diff, i) for i in best_results])




















# def check_diff(p_diff, c_diff):
#     """
#     For each nibble i:
#       - use count = DDT_counts[b][a] 
#       - prob = count/16
#       - -log2P = inf if count==0 else -log2(prob)
#     Returns (possible(bool), total_-log2P
#     """
#     p_nibbles = int_to_nibbles_int(p_diff)
#     c_nibbles = int_to_nibbles_int(c_diff)
#     details = []
#     total_log2 = 0.0
#     possible = True
#     for i, (a, b) in enumerate(zip(p_nibbles, c_nibbles)):
#         count = DDT_counts[a][b]
#         prob = DDT_prob[a][b]
#         if count == 0:
#             neg_log2 = math.inf
#             possible = False
#         else:
#             neg_log2 = -math.log2(prob)
#             total_log2 += neg_log2
#         details.append({"index": i, "p_nibble": f"{a:X}", "c_nibble": f"{b:X}",
#                         "count": count, "prob": prob, "-log2P": neg_log2})
#         # print(f"Nibble {i}: P_nibble={a:X}, C_nibble={b:X}, Count={count}, Prob={prob:.4f}, -log2P={neg_log2 if neg_log2!=math.inf else 'inf'}")
#     return possible, total_log2
