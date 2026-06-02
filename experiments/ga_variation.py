import sys
sys.stdout.reconfigure(line_buffering=True)
from math import *
import itertools
from collections import defaultdict, OrderedDict, Counter
import copy
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import random
import os
from functools import lru_cache
from math import log2
import ea_initial_population as init
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

word_size = 48
MASK_VAL = 2 ** word_size - 1

# 变异阶段性能开关（可通过环境变量覆盖）
GA_MAX_POSITION_COMBOS = int(os.getenv("GA_MAX_POSITION_COMBOS", "3000"))
GA_MAX_MUTATION_POOL = int(os.getenv("GA_MAX_MUTATION_POOL", "160"))
GA_TIME_BUDGET_MIN = float(os.getenv("GA_TIME_BUDGET_MIN", "25"))
GA_MIN_GENERATIONS = int(os.getenv("GA_MIN_GENERATIONS", "2"))
GA_FAST_COMPARE_MODE = os.getenv("GA_FAST_COMPARE_MODE", "1") == "1"
GA_BREED_MAX_ATTEMPTS_FACTOR = int(os.getenv("GA_BREED_MAX_ATTEMPTS_FACTOR", "12"))


def ALPHA():
    if word_size == 16:return(7)
    else:return(8)

def BETA():
    if word_size == 16:return(2)
    else:return(3)

#左移
def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (word_size - k)));
#右移
def ror(x,k):
    return((x >> k) | ((x << (word_size - k)) & MASK_VAL));
#差分以16进制表示，转为二进制列表,并补0
def hex_2_bin(value):
    ft = '{:0'+str(word_size)+'b}'
    return ([int(ft.format(value)[i]) for i in range(word_size)])

# 十六进制取反,negation
def ne(value):
    return value & MASK_VAL ^ MASK_VAL

# print(bin(ne(0b1110)))

# x,y,z里有哪些位是一样的
def eq(x,y,z):
    return (ne(x) ^ y) & (ne(x) ^ z) & MASK_VAL

# 计算汉明重量 
def ham_w(i):
    w = 0
    if i == 0:
        return 0
    while i > 0:
        i = i & (i - 1); w += 1
    return w

# Lipmaa-Moriai Alg2
def judge_valid(x, y, z):
    tmp1 = eq(x << 1, y << 1, z << 1)
    tmp2 = x ^ y ^ z ^ ((y<<1) & MASK_VAL)
    if (tmp1 & tmp2) == 0:
        return True
    else:
        return False


def xdp_add(x, y, z):
    # 运算这个函数的前提是已经判断了有效性
    mask = 2 ** (word_size - 1) - 1
    return 2 ** (-ham_w(ne(eq(x, y, z))&mask))


# all-one parity function
# 该位以前的连续1长度的奇偶性
def aop(x):
    log2n = int(log2(word_size))
    rx = [0] * log2n
    ry = [0] * log2n
    rx[0] = x & (x >> 1)
    for i in range(1,log2n-1):
        rx[i] = rx[i-1] & (rx[i-1] >> (1<<i))
    ry[0] = x & (ne(rx[0]))
    for i in range(1,log2n):
        ry[i] = ry[i-1] | ((ry[i-1] >> (1<<i)) & rx[i-1])
    return ry[-1]
# print(bin(aop(0b101110110111)))


# common alternation parity function
def cac(x,y):
    tmp = ne(x ^ y)
    #ne(x ^ y)检查是否相等
    p_list = hex_2_bin(aop(tmp & (tmp >> 1) & (x ^ (x >> 1))))
    #tmp & (tmp >> 1) & (x ^ (x >> 1))))检查是否交替

    # return p_list
    # 不知道为什么即使没有使用修正算法，也能得到概率为0.25的gamma
    # Emanuele Bellini对common alternation parity算法的修正
    x_list = hex_2_bin(x)
    y_list = hex_2_bin(y)
    for i in range(word_size-1,0,-1):
        j = word_size - i
        # 注意密码的高低位与列表的高低位是相反的
        if (x_list[j-1] == y_list[j-1]) and (x_list[j] == y_list[j]) and (x_list[j-1] != x_list[j]):
            p_list[j-1] = 0
        else:
            break
    return p_list

# print(cac(0b1011101101010000, 0b1011101100010001))

class Node:
    def __init__(self): #必须要有一个self参数，
        self.lsb = -1
        #i=1时的取值
        self.successors = [[False,False],[False,False]]
        #Gamma_i取0或1时能不能合法转移至下一比特
    def __repr__(self):
        return f"Node(lsb={self.lsb}, successors={self.successors})"

def delta_optimal(x, y, delta):
    graphs = []
    graph_positions = []
    # index_list = []
    x_list = hex_2_bin(x)
    y_list = hex_2_bin(y)
    p_list = cac(x, y)
    # print("x_list=", x_list)
    # print("y_list=", y_list)
    possibleCPositions = []
    for i in range(1, word_size):
        index = word_size - 1 - i
        if x_list[index] == y_list[index]:
            possibleCPositions.append(index)
    # print("possibleCPositions=", possibleCPositions)
    positionsLists = []
    for i in range(delta + 1):
        itera = itertools.combinations(possibleCPositions, i)
        for comb in itera:
            positionsLists.append(list(comb))
    # print("positionsLists = ", positionsLists)
    # print("positionsLists.shape = ", len(positionsLists))
    for positions in positionsLists:
        # print("positions = ", positions)
        graph=[Node() for i in range(word_size)]
        for i in range(word_size):
            graph[i].lsb = x_list[word_size - 1] ^ y_list[word_size - 1]
        for i in range(1, word_size):
            index = word_size - 1 - i
            # print("index = ", index)
            for j in [0, 1]:
                # print(graph[index].lsb)
                if (index == word_size - 2 and graph[index].lsb == j) or (index <= word_size - 3 and (graph[index + 2].successors[0][j] or graph[index + 2].successors[1][j])):
                        if (x_list[index + 1] == y_list[index + 1]) and (x_list[index + 1] == j):
                            graph[index + 1].successors[j][x_list[index] ^ y_list[index] ^ y_list[index + 1]] = True
                        elif ((index == 0) or (x_list[index] != y_list[index]) or (p_list[index] == 1)):
                            graph[index + 1].successors[j] = [True, True]
                        else:                   
                            if index in positions:
                                # print(index)
                                # print("drop time")
                                # index_list.append(index)
                                graph[index + 1].successors[j][1 - x_list[index]] = True
                            else:
                                # print("select time")
                                graph[index + 1].successors[j][x_list[index]] = True
        # print("next ", graph[15].successors[0][1])
        graphs.append(graph)
        graph_positions.append(positions)  # 记录扰动位置
    # print("graph.shape = ", len(graphs))

    return graphs, graph_positions


def wt(path):
    count = 0
    for p in path:
        if p == True:
            count +=1
    return count

# find paths in graph
def output_all_possible_gamma(graphs, t1):
    a = 1
    if graphs[0][word_size - 1].successors[0][0] == True or graphs[0][word_size - 1].successors[0][1] == True:
        a = 0
    def find_paths_dfs(graph, step, path):
        # 在这里就限制汉明重量，进一步加速
        if step == word_size - 1:
            if wt(path[::-1]) <= t1:
                return [path[::-1]]
            else:
                return []
        paths = []
        for i in [0, 1]:
            if graph[word_size - 1 - step].successors[path[-1]][i] == True:
                paths = paths + find_paths_dfs(graph, step + 1, path + [i])
        # print(paths)
        if wt(paths) <= t1:        
            return paths
        else:
            return []
    def find_paths(graph):
        paths = []
        # for i in [0, 1]:
        #     paths = paths + find_paths_dfs(graph, 0, [i])
        paths = paths + find_paths_dfs(graph, 0, [a])
        return paths
    z = []
    for graph in graphs:
        paths = find_paths(graph)
        for path in paths:
            z.append(path)
    return z


def convert_list_gamma_to_int(all_gamma):
    # 从向量转成数字保存，为了方便调用计算模加概率函数xdp_add
    z = set([])
    for var in all_gamma:
        if var!=[]:
            tmp = 0
            for i in range(word_size):
                tmp = tmp + (var[i] << (word_size-1-i))
            # 将有效的gamma存入
            # if judge_valid(x,y,tmp):
            z.add(tmp)
    z = list(z)
    return z

def trace_gamma_origin(graphs, graph_positions, t1):
    def wt(path):
        return sum(1 for p in path if p)

    def find_paths_dfs(graph, step, path):
        if step == word_size - 1:
            if wt(path[::-1]) <= t1:
                return [path[::-1]]
            else:
                return []
        paths = []
        for i in [0, 1]:
            if graph[word_size - 1 - step].successors[path[-1]][i]:
                paths += find_paths_dfs(graph, step + 1, path + [i])
        return paths

    def find_paths(graph, initial_bit):
        return find_paths_dfs(graph, 0, [initial_bit])

    gamma_origin_map = {}
    a = 1
    if graphs[0][word_size - 1].successors[0][0] or graphs[0][word_size - 1].successors[0][1]:
        a = 0

    for i, graph in enumerate(graphs):
        paths = find_paths(graph, a)
        for path in paths:
            gamma_int = sum(path[j] << (word_size - 1 - j) for j in range(word_size))
            if gamma_int not in gamma_origin_map:
                gamma_origin_map[gamma_int] = []
            gamma_origin_map[gamma_int].append(graph_positions[i])

    return gamma_origin_map


def construct_best_gamma_add_1round(x, y):
    x_list = hex_2_bin(ror(x, ALPHA()))
    y_list = hex_2_bin(y)
    p_list = cac(ror(x, ALPHA()), y)
    gamma = [0] * word_size
    gamma[word_size - 1] = x_list[word_size - 1] ^ y_list[word_size - 1]
    for i in range(word_size - 2):
        if x_list[word_size - 1 - i] == y_list[word_size - 1 - i] == gamma[word_size - 1 - i]:
            gamma[word_size - 2 - i] = x_list[word_size - 1 - i] ^ x_list[word_size - 2 - i] ^ y_list[word_size - 2 - i]  # rule (a)
        elif x_list[word_size - 2 - i] != y_list[word_size - 2 - i]:
            gamma[word_size - 2 - i] = 0  # rule (b) 自由选择取0
        elif x_list[word_size - 2 - i] == y_list[word_size - 2 - i] and p_list[word_size - 2 - i] == 0:
            gamma[word_size - 2 - i] = x_list[word_size - 2 - i]  # rule (c)
        else:
            gamma[word_size - 2 - i] = 0  # fallback：选0
    gamma1 = 0
    for i in range(word_size):
        gamma1 |= (gamma[i] << (word_size - 1 - i))
    z = -log2(xdp_add(ror(x, ALPHA()), y, gamma1))
    return gamma1, z

def construct_best_gamma_add(x, y):
    #获取每个子节点的最优孙节点权重，作为排序依据
    r = 3
    total_z = 0
    for _ in range(r):
        gamma, z = construct_best_gamma_add_1round(x, y)
        total_z += z
        x = gamma
        y = gamma ^ rol(y, BETA())
    return total_z

def construct_best_gamma_sub_1round(x, y):
    x_list = hex_2_bin(x)
    y_new = ror(x ^ y, BETA())
    y_list = hex_2_bin(y_new)
    p_list = cac(x, y_new)
    gamma = [0] * word_size
    gamma[word_size - 1] = x_list[word_size - 1] ^ y_list[word_size - 1]
    for i in range(word_size - 2):
        if x_list[word_size - 1 - i] == y_list[word_size - 1 - i] == gamma[word_size - 1 - i]:
            gamma[word_size - 2 - i] = x_list[word_size - 1 - i] ^ x_list[word_size - 2 - i] ^ y_list[word_size - 2 - i]  # rule (a)
        elif x_list[word_size - 2 - i] != y_list[word_size - 2 - i]:
            gamma[word_size - 2 - i] = 0  # rule (b) 自由选择取0
        elif x_list[word_size - 2 - i] == y_list[word_size - 2 - i] and p_list[word_size - 2 - i] == 0:
            gamma[word_size - 2 - i] = x_list[word_size - 2 - i]  # rule (c)
        else:
            gamma[word_size - 2 - i] = 0  # fallback：选0
    gamma1 = 0
    for i in range(word_size):
        gamma1 |= (gamma[i] << (word_size - 1 - i))
    gamma1 = rol(gamma1, ALPHA())
    z = -log2(xdp_add(x, y_new, gamma1))
    return gamma1, z

def construct_best_gamma_sub(x, y):
    #获取每个子节点的最优孙节点权重，作为排序依据
    r = 3 
    total_z = 0
    for _ in range(r):
        gamma, z = construct_best_gamma_sub_1round(x, y)
        total_z += z
        y = ror(x ^ y, BETA())
        y = ror(gamma ^ y, BETA())
        x = gamma
    return total_z

def get_children_score_rank_add(diff_l, diff_r, possible_children):
    scored_children = []
    for child in possible_children:
        score1 = -log2(xdp_add(ror(diff_l, ALPHA()), diff_r, child))
        l_new = child
        r_new = child ^ rol(diff_r, BETA())
        score2 = construct_best_gamma_add(l_new, r_new)
        scored_children.append((child, score1 + score2))
    scored_children.sort(key=lambda x: x[1])
    sorted_children = [child for child, _ in scored_children]
    sorted_scores = [score for _, score in scored_children]
    return sorted_children, sorted_scores

def generate_possible_children_add(diff_l, diff_r, delta, t1):
    graphs, graph_positions = delta_optimal(ror(diff_l, ALPHA()), diff_r, delta)  
    all_gamma = output_all_possible_gamma(graphs, t1)
    possible_children = convert_list_gamma_to_int(all_gamma)
    sorted_children, sorted_scores = get_children_score_rank_add(diff_l, diff_r, possible_children)
    return sorted_children, sorted_scores

def get_children_score_rank_sub(diff_l, diff_r, possible_children):
    scored_children = []
    for child in possible_children:
        score1 = -log2(xdp_add(diff_l, ror(diff_l ^ diff_r, BETA()), child))
        l_new = rol(child, ALPHA())
        r_new = ror(l_new ^ ror(diff_l ^ diff_r, BETA()), BETA())
        score2 = construct_best_gamma_sub(l_new, r_new)
        scored_children.append((child, score1 + score2))
    scored_children.sort(key=lambda x: x[1])
    sorted_children = [child for child, _ in scored_children]
    sorted_scores = [score for _, score in scored_children]
    return sorted_children, sorted_scores

def generate_possible_children_sub(diff_l, diff_r, delta, t1):
    graphs, graph_positions = delta_optimal(diff_l, ror(diff_l ^ diff_r, BETA()), delta)  
    all_gamma = output_all_possible_gamma(graphs, t1)
    possible_children = convert_list_gamma_to_int(all_gamma)
    sorted_children, sorted_scores = get_children_score_rank_sub(diff_l, diff_r, possible_children)
    return sorted_children, sorted_scores

def _collect_position_combos(eq_positions, delta, max_combos):
    """收集扰动位置组合，并限制数量以避免组合爆炸。"""
    positions = []
    for i in range(delta + 1):
        for comb in itertools.combinations(eq_positions, i):
            positions.append(comb)
            if i > 0 and comb[0] > i:
                positions.append(tuple(range(comb[0] - i + 1, comb[0] + 1)))
            if len(positions) >= max_combos:
                return positions
    return positions


@lru_cache(maxsize=8192)
def _vary_difference_cached(alpha, beta, gamma, delta, num_samples):
    # 变异差分（缓存 + 限制组合规模）
    alpha_bits = hex_2_bin(alpha)
    beta_bits = hex_2_bin(beta)
    gamma_bits = hex_2_bin(gamma)
    eq_positions = [i for i in range(word_size - 1) if alpha_bits[i] == beta_bits[i] == gamma_bits[i]]

    if not eq_positions:
        return tuple(), tuple()

    positions_lists = _collect_position_combos(eq_positions, delta, GA_MAX_POSITION_COMBOS)
    max_candidate_pool = max(num_samples * 8, GA_MAX_MUTATION_POOL)

    candidates = []
    tried = set()
    orig_prob = xdp_add(alpha, beta, gamma)
    if orig_prob <= 0:
        return tuple(), tuple()

    for pos_combo in positions_lists:
        gamma_prime = gamma
        for i in pos_combo:
            gamma_prime ^= (1 << (word_size - 1 - i))
        if gamma_prime in tried:
            continue
        tried.add(gamma_prime)
        if not judge_valid(alpha, beta, gamma_prime):
            continue
        prob = xdp_add(alpha, beta, gamma_prime)
        if prob <= 0:
            continue
        log_ratio = -log2(prob / orig_prob)
        if log_ratio > 0:
            candidates.append(gamma_prime)
            if len(candidates) >= max_candidate_pool:
                break

    if not candidates:
        return tuple(), tuple()

    ranked_children, ranked_scores = get_children_score_rank_add(alpha, beta, candidates)
    return tuple(ranked_children[:num_samples]), tuple(ranked_scores[:num_samples])


def vary_difference(alpha, beta, gamma, delta, num_samples=10):
    candidates, sorted_scores = _vary_difference_cached(alpha, beta, gamma, delta, num_samples)
    return list(candidates), list(sorted_scores)

def count_difference(current_paths, n):
    # 假设要统计第n层的差分情况
    layer_diffs = [path[n] for path in current_paths]
    # 统计出现次数
    counter = Counter(layer_diffs)
    sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    diff_nodes = [item[0] for item in sorted_counts]  # 差分节点
    counts = [item[1] for item in sorted_counts]      # 出现次数
    # print("差分节点数组：", [(hex(node[0]), hex(node[1])) for node in diff_nodes])
    # print("出现次数数组：", counts)
    return diff_nodes, counts

def rank_children(sorted_children, sorted_scores):
    ranks = []
    rank = 1
    prev_score = None
    for s in sorted_scores:
        if s != prev_score:
            ranks.append(rank)
            rank += 1
            prev_score = s
        else:
            ranks.append(rank - 1)
    return list(zip(sorted_children, sorted_scores, ranks))

def expand_population_with_rank(sorted_children, alpha, beta, delta, num_samples, require_multi_parent=False, top_n=None, output_as_list=False):
    """
    扩展种群并记录排名与来源关系，可筛选满足条件的节点（按排名前 top_n）。
    参数：
        sorted_children: list
            初始节点
        alpha, beta: 用于 vary_difference 的差分参数
        delta, num_samples: 变异控制参数
        require_multi_parent: bool
            若为 True，则只保留至少由两个不同初始节点变异得到的节点
        top_n: int 或 None
            若指定，则按分数升序取前 n 名节点
        output_as_list: bool
            若 True，返回满足筛选条件的节点数组，否则返回排序字典
    """
    all_nodes = set(sorted_children)
    origin_map = defaultdict(set)   # new_node -> {(parent, local_rank)}
    node_score_map = {}             # node -> score（包括变异节点）
    # --- 对每个父节点进行变异 ---
    for parent in sorted_children:
        candidates, scores = vary_difference(alpha, beta, parent, delta, num_samples)
        ranked_candidates = rank_children(candidates, scores)
        for cand, cand_score, cand_rank in ranked_candidates:
            origin_map[cand].add((parent, cand_rank))
            all_nodes.add(cand)
            node_score_map[cand] = cand_score
    # --- 多来源筛选 ---
    if require_multi_parent:
        filtered_nodes = {node for node, parents in origin_map.items() if len({p for (p, _) in parents}) >= 2}
    else:
        filtered_nodes = all_nodes
    # --- 按 score 升序排名，并取前 top_n ---
    if top_n is not None:
        # 若 node 没有 score，视为 +∞
        ranked_list = sorted(filtered_nodes, key=lambda n: node_score_map.get(n, float('inf')))
        filtered_nodes = set(ranked_list[:top_n])
    # --- 输出模式：节点数组 ---
    if output_as_list:
        return sorted(filtered_nodes, key=lambda n: node_score_map.get(n, float('inf')))
    # --- 字典模式 ---
    node_info_unsorted = {node: {"score": node_score_map.get(node, None),"parents": origin_map.get(node, set())} for node in filtered_nodes}
    node_info_sorted = OrderedDict(sorted(node_info_unsorted.items(), key=lambda x: (x[1]["score"] is None, x[1]["score"])))
    return node_info_sorted, origin_map


def find_initial_paths(first_path, start_layer, sorted_children, tree, top1=20, t1=14, t2=35, expand_threshold=2**(-24), top_n=12, vary_times=2, require_multi_parent=True):
    """根据变异结果生成初始路径"""
    diff_l, diff_r = first_path[start_layer - 1]
    # print(first_path)
    diff_l_new = ror(diff_l, ALPHA())
    candidates = sorted_children
    for i in range(vary_times):
        candidates = expand_population_with_rank(candidates, diff_l_new, diff_r, delta=4, num_samples=100, require_multi_parent=require_multi_parent, top_n=top_n**(i+1), output_as_list=True)
        # print(len(candidates))
        sorted_children.extend(candidates)
        # print(sorted_children)
    initial_paths = []
    prefix = first_path[:start_layer]
    for cand in tqdm(sorted_children, desc="生成初始路径进度", unit="节点", ncols=100):
        new_path = prefix.copy()
        new_node = (cand, cand ^ rol(diff_r, BETA()))
        # print(new_node)
        new_path.append(new_node)
        # -------- Step 1: 生成子节点 --------
        if len(tree[new_node]) == 0:
            x_ham = ham_w(new_node[0])
            y_ham = ham_w(new_node[1])
            if x_ham <= t1 and y_ham <= t1:
                possible_children, sorted_scores = generate_possible_children_add(new_node[0], new_node[1], delta=2, t1=14)
                top_children = possible_children[:top1]
                for child in top_children:
                    z_ham = ham_w(child)
                    if (x_ham + y_ham + z_ham <= t2) and (xdp_add(ror(new_node[0], ALPHA()), new_node[1], child) >= expand_threshold):
                        l_new = child
                        r_new = child ^ rol(new_node[1], BETA())
                        child_node = (l_new, r_new)
                        tree[new_node].append(child_node)
        # -------- Step 2: 若该节点无子节点，则跳过 --------
        if len(tree[new_node]) == 0:
            # 可选打印提示（不建议太频繁）
            # print(f"节点 {new_node} 无可扩展子节点，跳过")
            continue
        # -------- Step 3: 处理第一个子节点 --------
        new_son_node = tree[new_node][0]
        new_vary_son, sorted_scores = vary_difference(ror(new_node[0], ALPHA()), new_node[1], new_son_node[0],delta=4, num_samples=10)
        for son in new_vary_son:
            new_son_path = new_path.copy()
            son_vary_node = (son, son ^ rol(new_node[1], BETA()))
            new_son_path.append(son_vary_node)
            initial_paths.append(new_son_path)
        new_path.append(new_son_node)
        initial_paths.append(new_path)
    print(f"生成初始路径 {len(initial_paths)} 条，每条长度为 {len(initial_paths[0]) if initial_paths else 0}")
    # === 保存初始路径 ===
    save_path = project_path("saved_trail", f"speck64_{start_layer+2}round_{diff_l}_initial_paths.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(initial_paths, f)

    print(f"\n📁 结果已保存至:\n  {save_path}")
    print("🎯 搜索任务完成！")    
    return initial_paths

def process_candidate(args):
    cand, diff_r, prefix, t1, t2, top1, expand_threshold = args
    local_tree = {}
    local_paths = []
    new_path = prefix.copy()
    new_node = (cand, cand ^ rol(diff_r, BETA()))
    new_path.append(new_node)
    local_tree[new_node] = []
    # Step 1: 生成子节点
    x_ham = ham_w(new_node[0])
    y_ham = ham_w(new_node[1])
    if x_ham <= t1 and y_ham <= t1:
        possible_children, sorted_scores = generate_possible_children_add(new_node[0], new_node[1], delta=2, t1=t1)
        top_children = possible_children[:top1]
        for child in top_children:
            z_ham = ham_w(child)
            if (x_ham + y_ham + z_ham <= t2) and (xdp_add(ror(new_node[0], ALPHA()), new_node[1], child) >= expand_threshold):
                l_new = child
                r_new = child ^ rol(new_node[1], BETA())
                child_node = (l_new, r_new)
                local_tree[new_node].append(child_node)
    # Step 2: 若该节点无子节点，则跳过
    if len(local_tree[new_node]) == 0:
        return local_tree, local_paths
    # Step 3: 处理前两个子节点
    for new_son_node in local_tree[new_node][:2]:
        new_vary_son, sorted_scores = vary_difference(ror(new_node[0], ALPHA()), new_node[1], new_son_node[0], delta=4, num_samples=10)
        # 扩展路径
        for son in new_vary_son:
            new_son_path = new_path.copy()
            son_vary_node = (son, son ^ rol(new_node[1], BETA()))
            new_son_path.append(son_vary_node)
            local_paths.append(new_son_path)
        new_path.append(new_son_node)
    local_paths.append(new_path)
    return local_tree, local_paths

def find_initial_paths_parallel(first_path, start_layer, sorted_children, tree, top1=20, t1=14, t2=35, expand_threshold=2**(-24), top_n=12, vary_times=2, n_jobs=8, require_multi_parent=True):
    """根据变异结果生成初始路径（并行版本）"""
    diff_l, diff_r = first_path[start_layer - 1]
    # print(first_path[start_layer - 1])
    diff_l_new = ror(diff_l, ALPHA())
    candidates = sorted_children
    # print(candidates)
    for i in range(vary_times):
        candidates = expand_population_with_rank(candidates, diff_l_new, diff_r, delta=4, num_samples=100, require_multi_parent=require_multi_parent, top_n=top_n**(i+1), output_as_list=True)
        sorted_children.extend(candidates)
        # print(sorted_children)
    prefix = first_path[:start_layer]
    # --- 并行执行 ---
    tasks = [(cand, diff_r, prefix, t1, t2, top1, expand_threshold) for cand in sorted_children]
    initial_paths = []
    local_trees = []
    with Pool(processes=n_jobs) as pool:  # 可根据CPU核心数修改
        for local_tree, local_paths in tqdm(pool.imap_unordered(process_candidate, tasks), total=len(tasks), desc="生成初始路径进度", unit="节点", ncols=100):
            local_trees.append(local_tree)
            initial_paths.extend(local_paths)
    # --- 汇总 tree ---
    for local_tree in local_trees:
        for k, v in local_tree.items():
            if k not in tree:
                tree[k] = []
            tree[k].extend(v)
    print(f"生成初始路径 {len(initial_paths)} 条，每条长度为 {len(initial_paths[0]) if initial_paths else 0}")
    return initial_paths

def paths_search_add(diff_l, diff_r, tree, nr, top1=20, top2=10, delta=2, t1=14, t2=35, expand_threshold=2**(-24), speck=64, population_size=2000, initial_paths=None):
    """
    改进版：支持从部分初始路径继续搜索的版本。
    参数：
        diff_l, diff_r: 起始差分（用于命名输出文件）
        initial_paths: list[list[tuple]]，每条为部分初始路径（长度相同）
    """
    # === 初始化路径 ===
    if initial_paths is not None and len(initial_paths) > 0:
        current_paths = initial_paths.copy()
        start_layer = len(initial_paths[0]) - 1
        print(f"=== 从 {len(initial_paths)} 条部分初始路径继续搜索 (起始层: {start_layer + 1}/{nr}) ===")
    else:
        root = (diff_l, diff_r)
        current_paths = [[root]]
        start_layer = 0
        print(f"=== 从根节点开始分层搜索 (总层数: {nr}) ===")
    population = [current_paths]
    # === 主搜索循环 ===
    for layer in range(start_layer, nr):
        print(f"\n--- 正在处理第 {layer + 1} 层 (当前路径数: {len(current_paths)}) ---")
        next_paths = []
        expanded_nodes = 0
        for path_idx, path in enumerate(current_paths):
            last_node = path[-1]
            print(f"  处理进度: {path_idx}/{len(current_paths)} 条路径", end='\r')
            # 如果该节点无子节点，则生成
            if len(tree[last_node]) == 0:
                x_ham = ham_w(last_node[0])
                y_ham = ham_w(last_node[1])
                if x_ham <= t1 and y_ham <= t1:
                    possible_children, sorted_scores = generate_possible_children_add(last_node[0], last_node[1], delta, t1)
                    top_children = possible_children[:top1]
                    for child in top_children:
                        z_ham = ham_w(child)
                        if (x_ham + y_ham + z_ham <= t2) and (xdp_add(ror(last_node[0], ALPHA()), last_node[1], child) >= expand_threshold):
                            l_new = child
                            r_new = child ^ rol(last_node[1], BETA())
                            child_node = (l_new, r_new)
                            tree[last_node].append(child_node)
                            expanded_nodes += 1
            # 添加有效子节点
            valid_children = tree[last_node]
            for child_node in valid_children[:top2]:
                new_path = path.copy()
                new_path.append(child_node)
                next_paths.append(new_path)

        print(f"\n  本层扩展了 {expanded_nodes} 个新节点")
        population.append(next_paths)
        current_paths = next_paths
        # 路径筛选逻辑
        if layer >= 3 and len(current_paths) > population_size:
            print(f"  正在进行路径选择 (当前路径数: {len(current_paths)})...")
            evaluated_paths = [(evaluate_path_add(path, nr), path) for path in current_paths]
            evaluated_paths.sort(key=lambda x: x[0])  # 按分数升序
            current_paths = [path for (score, path) in evaluated_paths[:population_size]]
            print(f"  ✅ 选择完成: 保留前{population_size}条最优路径 (最低分数: {evaluated_paths[0][0]:.6f})")
        # 若没有可扩展路径，则提前结束
        if not current_paths:
            print("⚠️ 无可扩展路径，搜索提前结束。")
            break
    # === 输出结果 ===
    root_diff_l = diff_l
    save_path1 = project_path("saved_ea_txt", f"speck{speck}", f"speck{speck}_result_ea_{nr}round_{diff_l}.txt")
    os.makedirs(os.path.dirname(save_path1), exist_ok=True)
    with open(save_path1, 'w') as f:
        f.write("=== 差分路径搜索结果 ===\n")
        f.write(f"Speck版本: {speck}\n")
        f.write(f"总层数: {nr}\n")
        f.write(f"起始层: {start_layer + 1}\n")
        f.write(f"最终保留路径数: {len(current_paths)}\n\n")
        for i, path in enumerate(current_paths):
            score = evaluate_path_add(path, nr)
            f.write(f"\n---- 路径 {i + 1} (总分: {score:.4f}) ----\n")
            for j in range(len(path) - 1):
                diff_l, diff_r = path[j]
                child = path[j + 1][0]
                node_score = -log2(xdp_add(ror(diff_l, ALPHA()), diff_r, child))
                f.write(f"层 {j}: ({hex(diff_l)}, {hex(diff_r)}) | 分数: {node_score:.4f}\n")
            f.write(f"层 {len(path) - 1}: ({hex(path[-1][0])}, {hex(path[-1][1])})\n")
    # === 保存结构数据 ===
    save_path2 = project_path("saved_trail", f"speck{speck}_{nr}round_{root_diff_l}_evolution_data.pkl")
    os.makedirs(os.path.dirname(save_path2), exist_ok=True)
    with open(save_path2, 'wb') as f:
        pickle.dump((current_paths, tree), f)
    print(f"\n📁 结果已保存至:\n  {save_path1}\n  {save_path2}")
    print("🎯 搜索任务完成！")
    return current_paths, tree

def evaluate_path_add(path, nr):
    """评估路径的适应度分数"""
    fitness_score = 0.0
    for i in range(len(path) - 1):
        diff_l, diff_r = path[i]
        child = path[i + 1][0]
        fitness_score += -log2(xdp_add(ror(diff_l, ALPHA()), diff_r, child))
    # 若路径长度足够（接近完整轮数），仅返回实际得分
    if len(path) > nr - 3:
        return fitness_score
    # 否则加入预估得分（基于最后节点的左右差分）
    else:
        diff_l_last, diff_r_last = path[-1]
        estimated_score = construct_best_gamma_add(diff_l_last, diff_r_last)
        return fitness_score + estimated_score

def expand_single_path_add(args):
    """单条路径的扩展逻辑"""
    path, tree, top1, top2, delta, t1, t2, expand_threshold = args
    next_paths = []
    expanded_nodes = 0
    last_node = path[-1]
    if len(tree[last_node]) == 0:
        x_ham = ham_w(last_node[0])
        y_ham = ham_w(last_node[1])
        if x_ham <= t1 and y_ham <= t1:
            possible_children, sorted_scores = generate_possible_children_add(last_node[0], last_node[1], delta, t1)
            top_children = possible_children[:top1]
            for child in top_children:
                z_ham = ham_w(child)
                if (x_ham + y_ham + z_ham <= t2) and (xdp_add(ror(last_node[0], ALPHA()), last_node[1], child) >= expand_threshold):
                    l_new = child
                    r_new = child ^ rol(last_node[1], BETA())
                    child_node = (l_new, r_new)
                    tree[last_node].append(child_node)
                    expanded_nodes += 1
    valid_children = tree[last_node]
    for child_node in valid_children[:top2]:
        new_path = path.copy()
        new_path.append(child_node)
        next_paths.append(new_path)
    return next_paths, expanded_nodes

def paths_search_add_parallel(diff_l, diff_r, tree, nr, top1=10, top2=10, delta=2, t1=14, t2=30, expand_threshold=2**(-24), speck=64, population_size=2000, initial_paths=None, n_jobs=8):
    """
    多进程加速版路径搜索
    """
    # === 初始化路径 ===
    if initial_paths is not None and len(initial_paths) > 0:
        current_paths = initial_paths.copy()
        start_layer = len(initial_paths[0]) - 1
        print(f"=== 从 {len(initial_paths)} 条部分初始路径继续搜索 (起始层: {start_layer + 1}/{nr}) ===")
    else:
        root = (diff_l, diff_r)
        current_paths = [[root]]
        start_layer = 0
        print(f"=== 从根节点开始分层搜索 (总层数: {nr}) ===")
    population = [current_paths]
    total_start_time = time.time()
    # === 主搜索循环 ===
    for layer in range(start_layer, nr):
        layer_start = time.time()
        print(f"\n--- 正在处理第 {layer + 1} 层 (当前路径数: {len(current_paths)}) ---")
        next_paths = []
        expanded_nodes_total = 0
        task_args = [(path, tree, top1, top2, delta, t1, t2, expand_threshold)
                     for path in current_paths]
        # === 使用多进程 + tqdm ===
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(expand_single_path_add, args) for args in task_args]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"第 {layer+1} 层处理中", ncols=90):
                expanded, expanded_nodes = future.result()
                next_paths.extend(expanded)
                expanded_nodes_total += expanded_nodes
        elapsed = time.time() - layer_start
        print(f"  ✅ 本层扩展 {expanded_nodes_total} 个新节点，共生成 {len(next_paths)} 条路径，用时 {elapsed:.1f}s")
        population.append(next_paths)
        current_paths = next_paths
        # === 路径筛选逻辑 ===
        if layer >= 3 and len(current_paths) > population_size:
            print(f"  正在进行路径选择 (当前路径数: {len(current_paths)})...")
            evaluated_paths = [(evaluate_path_add(path, nr), path) for path in current_paths]
            evaluated_paths.sort(key=lambda x: x[0])  # 按分数升序
            current_paths = [path for (score, path) in evaluated_paths[:population_size]]
            print(f"  ✅ 选择完成: 保留前{population_size}条最优路径 (最低分: {evaluated_paths[0][0]:.6f})")
        # === 若无可扩展路径则提前结束 ===
        if not current_paths:
            print("⚠️ 无可扩展路径，搜索提前结束。")
            break
    total_elapsed = time.time() - total_start_time
    print(f"\n🎯 搜索完成，总耗时 {total_elapsed/60:.2f} 分钟")
    # === 输出结果 ===
    root_diff_l = diff_l
    save_path1 = project_path("saved_ea_txt", f"speck{speck}", f"speck{speck}_result_ga_{nr}round_{diff_l}.txt")
    os.makedirs(os.path.dirname(save_path1), exist_ok=True)
    with open(save_path1, 'w') as f:
        f.write("=== 差分路径搜索结果 ===\n")
        f.write(f"Speck版本: {speck}\n")
        f.write(f"总层数: {nr}\n")
        f.write(f"起始层: {start_layer + 1}\n")
        f.write(f"最终保留路径数: {len(current_paths)}\n")
        f.write(f"总耗时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)\n\n")
        for i, path in enumerate(current_paths):
            score = evaluate_path_add(path, nr)
            f.write(f"\n---- 路径 {i + 1} (总分: {score:.4f}) ----\n")
            for j in range(len(path) - 1):
                diff_l, diff_r = path[j]
                child = path[j + 1][0]
                node_score = -log2(xdp_add(ror(diff_l, ALPHA()), diff_r, child))
                f.write(f"层 {j}: ({hex(diff_l)}, {hex(diff_r)}) | 分数: {node_score:.4f}\n")
            f.write(f"层 {len(path) - 1}: ({hex(path[-1][0])}, {hex(path[-1][1])})\n")
    # === 保存结构数据 ===
    save_path2 = project_path("saved_trail", f"speck{speck}_{nr}round_{root_diff_l}_ga.pkl")
    os.makedirs(os.path.dirname(save_path2), exist_ok=True)
    with open(save_path2, 'wb') as f:
        pickle.dump((current_paths, tree), f)
    print(f"\n📁 结果已保存至:\n  {save_path1}\n  {save_path2}")
    return current_paths, tree

def paths_search_sub(diff_l, diff_r, tree, nr, top1=20, top2=10, delta=2, t1=14, t2=35, expand_threshold=2**(-24), speck=64, population_size=2000, initial_paths=None):
    """
    Sub 版本：支持从部分初始路径继续搜索。
    参数与 paths_search_add 相同，但内部使用 sub 模式的候选/构造/评分逻辑。
    """
    # === 初始化路径 ===
    if initial_paths is not None and len(initial_paths) > 0:
        current_paths = initial_paths.copy()
        start_layer = len(initial_paths[0]) - 1
        print(f"=== 从 {len(initial_paths)} 条部分初始路径继续搜索 (起始层: {start_layer + 1}/{nr}) ===")
    else:
        root = (diff_l, diff_r)
        current_paths = [[root]]
        start_layer = 0
        print(f"=== 从根节点开始分层搜索 (总层数: {nr}) ===")
    population = [current_paths]
    # === 主搜索循环 ===
    for layer in range(start_layer, nr):
        print(f"\n--- 正在处理第 {layer + 1} 层 (当前路径数: {len(current_paths)}) ---")
        next_paths = []
        expanded_nodes = 0
        for path_idx, path in enumerate(current_paths):
            last_node = path[-1]
            print(f"  处理进度: {path_idx}/{len(current_paths)} 条路径", end='\r')
            # 如果该节点无子节点，则生成（sub 模式）
            if len(tree[last_node]) == 0:
                x_ham = ham_w(last_node[0])
                y_ham = ham_w(last_node[1])
                if x_ham <= t1 and y_ham <= t1:
                    possible_children, sorted_scores = generate_possible_children_sub(last_node[0], last_node[1], delta, t1)
                    top_children = possible_children[:top1]
                    for child in top_children:
                        z_ham = ham_w(child)
                        # sub 模式：计算 diff_r_new（逆向/减法关系）
                        diff_r_new = ror(last_node[0] ^ last_node[1], BETA())
                        # 判定扩展阈值（用 xdp_add(last_l, diff_r_new, child)）
                        if (x_ham + y_ham + z_ham <= t2) and (xdp_add(last_node[0], diff_r_new, child) >= expand_threshold):
                            l_new = rol(child, ALPHA())
                            r_new = diff_r_new
                            child_node = (l_new, r_new)
                            tree[last_node].append(child_node)
                            expanded_nodes += 1
            # 添加有效子节点（最多 top2 个）
            valid_children = tree[last_node]
            for child_node in valid_children[:top2]:
                new_path = path.copy()
                new_path.append(child_node)
                next_paths.append(new_path)
        print(f"\n  本层扩展了 {expanded_nodes} 个新节点")
        population.append(next_paths)
        current_paths = next_paths
        # 路径筛选逻辑（使用 evaluate_path_sub）
        if layer >= 3 and len(current_paths) > population_size:
            print(f"  正在进行路径选择 (当前路径数: {len(current_paths)})...")
            evaluated_paths = [(evaluate_path_sub(path, nr), path) for path in current_paths]
            evaluated_paths.sort(key=lambda x: x[0])  # 按分数升序（越小越好）
            current_paths = [path for (score, path) in evaluated_paths[:population_size]]
            print(f"  ✅ 选择完成: 保留前{population_size}条最优路径 (最低分数: {evaluated_paths[0][0]:.6f})")
        # 若没有可扩展路径，则提前结束
        if not current_paths:
            print("⚠️ 无可扩展路径，搜索提前结束。")
            break
    # === 输出结果 ===
    root_diff_l = diff_l
    save_path1 = project_path("saved_ea_txt", f"speck{speck}", f"speck{speck}_result_ea_{nr}round_{diff_l}.txt")
    os.makedirs(os.path.dirname(save_path1), exist_ok=True)
    with open(save_path1, 'w') as f:
        f.write("=== 差分路径搜索结果 ===\n")
        f.write(f"Speck版本: {speck}\n")
        f.write(f"总层数: {nr}\n")
        f.write(f"起始层: {start_layer + 1}\n")
        f.write(f"最终保留路径数: {len(current_paths)}\n\n")
        for i, path in enumerate(current_paths):
            score = evaluate_path_sub(path, nr)
            f.write(f"\n---- 路径 {i + 1} (总分: {score:.4f}) ----\n")
            for j in range(len(path) - 1):
                diff_l, diff_r = path[j]
                child = path[j + 1][0]
                # sub 模式的节点分数公式（与先前 first_best_search_sub 保持一致）
                node_score = -log2(xdp_add(diff_l, ror(diff_l ^ diff_r, BETA()), ror(child, ALPHA())))
                f.write(f"层 {j}: ({hex(diff_l)}, {hex(diff_r)}) | 分数: {node_score:.4f}\n")
            f.write(f"层 {len(path) - 1}: ({hex(path[-1][0])}, {hex(path[-1][1])})\n")
    # === 保存结构数据 ===
    save_path2 = project_path("saved_trail", f"speck{speck}_{nr}round_{root_diff_l}_evolution_data.pkl")
    os.makedirs(os.path.dirname(save_path2), exist_ok=True)
    with open(save_path2, 'wb') as f:
        pickle.dump((current_paths, tree), f)
    print(f"\n📁 结果已保存至:\n  {save_path1}\n  {save_path2}")
    print("🎯 搜索任务完成！")
    return current_paths, tree

def evaluate_path_sub(path, nr):
    """评估路径的适应度分数"""
    fitness_score = 0
    for i in range(len(path) - 1):
        diff_l, diff_r = path[i]
        child = path[i + 1][0]
        fitness_score += -log2(xdp_add(diff_l, ror(diff_l ^ diff_r, BETA()), ror(child, ALPHA())))
    # 若路径长度足够（接近完整轮数），仅返回实际得分
    if len(path) > nr - 3:
        return fitness_score
    # 否则加入预估得分（基于最后节点的左右差分）
    else:
        diff_l_last, diff_r_last = path[-1]
        estimated_score = construct_best_gamma_sub(diff_l_last, diff_r_last)
        return fitness_score + estimated_score

def expand_single_path_sub(args):
    """单条路径的扩展逻辑（sub 模式）"""
    path, tree, top1, top2, delta, t1, t2, expand_threshold = args
    next_paths = []
    expanded_nodes = 0
    last_node = path[-1]
    if len(tree[last_node]) == 0:
        x_ham = ham_w(last_node[0])
        y_ham = ham_w(last_node[1])
        if x_ham <= t1 and y_ham <= t1:
            possible_children, sorted_scores = generate_possible_children_sub(last_node[0], last_node[1], delta, t1)
            top_children = possible_children[:top1]
            for child in top_children:
                z_ham = ham_w(child)
                diff_r_new = ror(last_node[0] ^ last_node[1], BETA())
                if (x_ham + y_ham + z_ham <= t2) and (xdp_add(last_node[0], diff_r_new, child) >= expand_threshold):
                    l_new = rol(child, ALPHA())
                    r_new = diff_r_new
                    child_node = (l_new, r_new)
                    tree[last_node].append(child_node)
                    expanded_nodes += 1
    valid_children = tree[last_node]
    for child_node in valid_children[:top2]:
        new_path = path.copy()
        new_path.append(child_node)
        next_paths.append(new_path)
    return next_paths, expanded_nodes

def paths_search_sub_parallel(diff_l, diff_r, tree, nr, top1=20, top2=10, delta=2, t1=14, t2=30, expand_threshold=2**(-24), speck=64, population_size=2000, initial_paths=None, n_jobs=8):
    """
    多进程并行的 sub 版本路径搜索（对应 paths_search_add_parallel）。
    依赖 expand_single_path_sub 做单条路径扩展。
    """
    # === 初始化路径 ===
    if initial_paths is not None and len(initial_paths) > 0:
        current_paths = initial_paths.copy()
        start_layer = len(initial_paths[0]) - 1
        print(f"=== 从 {len(initial_paths)} 条部分初始路径继续搜索 (起始层: {start_layer + 1}/{nr}) ===")
    else:
        root = (diff_l, diff_r)
        current_paths = [[root]]
        start_layer = 0
        print(f"=== 从根节点开始分层搜索 (总层数: {nr}) ===")
    population = [current_paths]
    total_start_time = time.time()
    # === 主搜索循环 ===
    for layer in range(start_layer, nr):
        layer_start = time.time()
        print(f"\n--- 正在处理第 {layer + 1} 层 (当前路径数: {len(current_paths)}) ---")
        next_paths = []
        expanded_nodes_total = 0
        task_args = [(path, tree, top1, top2, delta, t1, t2, expand_threshold) for path in current_paths]
        # 多进程并行调用 expand_single_path_sub
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(expand_single_path_sub, args) for args in task_args]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"第 {layer+1} 层处理中", ncols=90):
                expanded, expanded_nodes = future.result()
                next_paths.extend(expanded)
                expanded_nodes_total += expanded_nodes
        elapsed = time.time() - layer_start
        print(f"  ✅ 本层扩展 {expanded_nodes_total} 个新节点，共生成 {len(next_paths)} 条路径，用时 {elapsed:.1f}s")
        population.append(next_paths)
        current_paths = next_paths
        # 路径筛选（使用 evaluate_path_sub）
        if layer >= 3 and len(current_paths) > population_size:
            print(f"  正在进行路径选择 (当前路径数: {len(current_paths)})...")
            evaluated_paths = [(evaluate_path_sub(path, nr), path) for path in current_paths]
            evaluated_paths.sort(key=lambda x: x[0])  # 升序p
            current_paths = [path for (score, path) in evaluated_paths[:population_size]]
            print(f"  ✅ 选择完成: 保留前{population_size}条最优路径 (最低分: {evaluated_paths[0][0]:.6f})")
        # 若无可扩展路径则提前结束
        if not current_paths:
            print("⚠️ 无可扩展路径，搜索提前结束。")
            break
    total_elapsed = time.time() - total_start_time
    print(f"\n🎯 搜索完成，总耗时 {total_elapsed/60:.2f} 分钟")
    # === 输出结果 ===
    root_diff_l = diff_l
    save_path1 = project_path("saved_ea_txt", f"speck{speck}", f"speck{speck}_result_ea_{nr}round_{diff_l}.txt")
    os.makedirs(os.path.dirname(save_path1), exist_ok=True)
    with open(save_path1, 'w') as f:
        f.write("=== 差分路径搜索结果 ===\n")
        f.write(f"Speck版本: {speck}\n")
        f.write(f"总层数: {nr}\n")
        f.write(f"起始层: {start_layer + 1}\n")
        f.write(f"最终保留路径数: {len(current_paths)}\n\n")
        for i, path in enumerate(current_paths):
            score = evaluate_path_sub(path, nr)
            f.write(f"\n---- 路径 {i + 1} (总分: {score:.4f}) ----\n")
            for j in range(len(path) - 1):
                diff_l, diff_r = path[j]
                child = path[j + 1][0]
                node_score = -log2(xdp_add(diff_l, ror(diff_l ^ diff_r, BETA()), ror(child, ALPHA())))
                f.write(f"层 {j}: ({hex(diff_l)}, {hex(diff_r)}) | 分数: {node_score:.4f}\n")
            f.write(f"层 {len(path) - 1}: ({hex(path[-1][0])}, {hex(path[-1][1])})\n")
    # 保存结构数据
    save_path2 = project_path("saved_trail", f"speck{speck}_{nr}round_{root_diff_l}_evolution_data.pkl")
    os.makedirs(os.path.dirname(save_path2), exist_ok=True)
    with open(save_path2, 'wb') as f:
        pickle.dump((current_paths, tree), f)
    print(f"\n📁 结果已保存至:\n  {save_path1}\n  {save_path2}")
    return current_paths, tree


# =========================
# GA baseline for round-4/5 mutation (parallel version)
# =========================

def make_add_node_from_left(child_left, prev_right):
    """由左支差分恢复当前轮节点 (L, R)。"""
    return (child_left, child_left ^ rol(prev_right, BETA()))


def is_valid_add_transition(prev_node, child_left, expand_threshold=0.0):
    """检查 prev_node -> child_left 是否为有效 add 转移。"""
    alpha = ror(prev_node[0], ALPHA())
    beta = prev_node[1]
    if not judge_valid(alpha, beta, child_left):
        return False
    if expand_threshold > 0 and xdp_add(alpha, beta, child_left) < expand_threshold:
        return False
    return True


def filter_valid_round4_children(prev_node, candidate_children, delta=2, t1=14, expand_threshold=2**(-24), fallback_topk=20):
    """过滤用户提供的 round-4 左支候选，并在不足时补齐可行候选。"""
    valid = []
    seen = set()
    for child in candidate_children:
        if child in seen:
            continue
        if is_valid_add_transition(prev_node, child, expand_threshold=expand_threshold):
            valid.append(child)
            seen.add(child)
    if len(valid) < max(4, min(fallback_topk, len(candidate_children) + 2)):
        generated, _ = generate_possible_children_add(prev_node[0], prev_node[1], delta=delta, t1=t1)
        for child in generated[:fallback_topk]:
            if child in seen:
                continue
            if is_valid_add_transition(prev_node, child, expand_threshold=expand_threshold):
                valid.append(child)
                seen.add(child)
    return valid


def get_round5_candidates_from_round4(node4, delta=2, t1=14, expand_threshold=2**(-24), topk=12):
    """给定 round-4 节点，生成 round-5 合法左支候选。"""
    x_ham = ham_w(node4[0])
    y_ham = ham_w(node4[1])
    if x_ham > t1 or y_ham > t1:
        return []
    children, _ = generate_possible_children_add(node4[0], node4[1], delta=delta, t1=t1)
    out = []
    for child in children:
        z_ham = ham_w(child)
        if (x_ham + y_ham + z_ham <= 35) and is_valid_add_transition(node4, child, expand_threshold=expand_threshold):
            out.append(child)
        if len(out) >= topk:
            break
    return out


def repair_round5_gene(node4, preferred_gene=None, delta=2, t1=14, expand_threshold=2**(-24), topk=12):
    """在给定 round-4 节点下修复/生成合法的 round-5 左支差分。"""
    cands = get_round5_candidates_from_round4(node4, delta=delta, t1=t1, expand_threshold=expand_threshold, topk=topk)
    if not cands:
        return None
    if preferred_gene is not None and preferred_gene in cands:
        return preferred_gene
    return cands[0]


def build_round45_individual(prev_node, g4, g5=None, delta=2, t1=14, expand_threshold=2**(-24)):
    """构建两基因个体，必要时自动修复 round-5。"""
    if not is_valid_add_transition(prev_node, g4, expand_threshold=expand_threshold):
        return None
    node4 = make_add_node_from_left(g4, prev_node[1])
    g5_fixed = repair_round5_gene(node4, preferred_gene=g5, delta=delta, t1=t1, expand_threshold=expand_threshold)
    if g5_fixed is None:
        return None
    return {'g4': g4, 'g5': g5_fixed}


def decode_round45_individual(first_path, start_layer, individual):
    """把 round-4/5 两基因个体解码为部分路径。"""
    prefix = copy.deepcopy(first_path[:start_layer])
    prev_node = prefix[-1]
    node4 = make_add_node_from_left(individual['g4'], prev_node[1])
    node5 = make_add_node_from_left(individual['g5'], node4[1])
    return prefix + [node4, node5]


def prepare_ga_seed_population(first_path, start_layer, sorted_children, population_size=64, delta=2, t1=14, expand_threshold=2**(-24)):
    """根据固定前缀生成 GA 初始种群。"""
    prev_node = first_path[start_layer - 1]
    seed_children = filter_valid_round4_children(prev_node, sorted_children, delta=delta, t1=t1, expand_threshold=expand_threshold, fallback_topk=max(20, population_size))
    population = []
    seen = set()
    for g4 in seed_children:
        node4 = make_add_node_from_left(g4, prev_node[1])
        round5_cands = get_round5_candidates_from_round4(node4, delta=delta, t1=t1, expand_threshold=expand_threshold, topk=4)
        for g5 in round5_cands:
            key = (g4, g5)
            if key in seen:
                continue
            population.append({'g4': g4, 'g5': g5})
            seen.add(key)
            if len(population) >= population_size:
                return population
    # 不足时，通过对已有个体做轻量随机扰动补齐
    if not population:
        raise RuntimeError('GA 初始种群为空：请检查 round-4 候选是否与固定前缀兼容。')
    while len(population) < population_size:
        parent = random.choice(population)
        mutated = mutate_round45_individual(first_path, start_layer, parent, mutation_delta=4, num_samples=12, expand_threshold=expand_threshold)
        if mutated is None:
            mutated = parent.copy()
        key = (mutated['g4'], mutated['g5'])
        if key not in seen:
            population.append(mutated)
            seen.add(key)
    return population


def tournament_select(scored_population, tournament_size=3):
    """锦标赛选择：scored_population 形如 [(score, ind, path), ...]。"""
    candidates = random.sample(scored_population, min(tournament_size, len(scored_population)))
    candidates.sort(key=lambda x: x[0])
    return copy.deepcopy(candidates[0][1])


def crossover_round45_individuals(first_path, start_layer, parent1, parent2, delta=2, t1=14, expand_threshold=2**(-24)):
    """两基因交叉：round-4 来自一方，round-5 尽量继承另一方，否则自动修复。"""
    prev_node = first_path[start_layer - 1]
    g4 = random.choice([parent1['g4'], parent2['g4']])
    node4 = make_add_node_from_left(g4, prev_node[1])
    preferred_g5 = random.choice([parent1['g5'], parent2['g5']])
    child = build_round45_individual(prev_node, g4, preferred_g5, delta=delta, t1=t1, expand_threshold=expand_threshold)
    if child is not None:
        return child
    # 修复失败时退化为最优合法 round-5
    g5 = repair_round5_gene(node4, preferred_gene=None, delta=delta, t1=t1, expand_threshold=expand_threshold)
    if g5 is None:
        return copy.deepcopy(random.choice([parent1, parent2]))
    return {'g4': g4, 'g5': g5}


def mutate_round45_individual(first_path, start_layer, individual, mutation_delta=4, num_samples=12, expand_threshold=2**(-24)):
    """同时对 round-4 和 round-5 两轮做变异。"""
    prev_node = first_path[start_layer - 1]
    alpha4 = ror(prev_node[0], ALPHA())
    beta4 = prev_node[1]
    # 先变 round-4
    g4_base = individual['g4']
    cand4, _ = vary_difference(alpha4, beta4, g4_base, delta=mutation_delta, num_samples=num_samples)
    cand4 = [g for g in cand4 if is_valid_add_transition(prev_node, g, expand_threshold=expand_threshold)]
    g4_new = random.choice(cand4) if cand4 else g4_base
    node4 = make_add_node_from_left(g4_new, prev_node[1])
    # 再在新 round-4 条件下变 round-5
    g5_base = repair_round5_gene(node4, preferred_gene=individual.get('g5'), delta=2, t1=14, expand_threshold=expand_threshold)
    if g5_base is None:
        return None
    alpha5 = ror(node4[0], ALPHA())
    beta5 = node4[1]
    cand5, _ = vary_difference(alpha5, beta5, g5_base, delta=mutation_delta, num_samples=num_samples)
    cand5 = [g for g in cand5 if is_valid_add_transition(node4, g, expand_threshold=expand_threshold)]
    g5_new = random.choice(cand5) if cand5 else g5_base
    return {'g4': g4_new, 'g5': g5_new}


def mutate_round45_individual_fast(first_path, start_layer, individual, mutation_delta=2, num_samples=6, expand_threshold=2**(-24)):
    """轻量变异：仅变异一个基因，适合对比实验快速跑通。"""
    prev_node = first_path[start_layer - 1]
    out = {'g4': individual['g4'], 'g5': individual['g5']}
    mutate_g4 = random.random() < 0.5

    if mutate_g4:
        alpha4 = ror(prev_node[0], ALPHA())
        beta4 = prev_node[1]
        cand4, _ = vary_difference(alpha4, beta4, out['g4'], delta=mutation_delta, num_samples=num_samples)
        cand4 = [g for g in cand4 if is_valid_add_transition(prev_node, g, expand_threshold=expand_threshold)]
        if cand4:
            out['g4'] = random.choice(cand4)
        node4 = make_add_node_from_left(out['g4'], prev_node[1])
        fixed_g5 = repair_round5_gene(node4, preferred_gene=out['g5'], delta=2, t1=14, expand_threshold=expand_threshold)
        if fixed_g5 is None:
            return None
        out['g5'] = fixed_g5
        return out

    node4 = make_add_node_from_left(out['g4'], prev_node[1])
    base_g5 = repair_round5_gene(node4, preferred_gene=out['g5'], delta=2, t1=14, expand_threshold=expand_threshold)
    if base_g5 is None:
        return None
    alpha5 = ror(node4[0], ALPHA())
    beta5 = node4[1]
    cand5, _ = vary_difference(alpha5, beta5, base_g5, delta=mutation_delta, num_samples=num_samples)
    cand5 = [g for g in cand5 if is_valid_add_transition(node4, g, expand_threshold=expand_threshold)]
    out['g5'] = random.choice(cand5) if cand5 else base_g5
    return out


def evaluate_round45_individual_worker(args):
    """多进程评估单个 GA 个体（快速评分，不构造完整路径）。"""
    first_path, start_layer, individual, target_nr, prefix_score = args

    prev_node = first_path[start_layer - 1]
    g4 = individual['g4']
    g5 = individual['g5']

    node4 = make_add_node_from_left(g4, prev_node[1])
    node5 = make_add_node_from_left(g5, node4[1])

    score4 = -log2(xdp_add(ror(prev_node[0], ALPHA()), prev_node[1], g4))
    score5 = -log2(xdp_add(ror(node4[0], ALPHA()), node4[1], g5))
    score = prefix_score + score4 + score5

    decoded_len = start_layer + 2
    if decoded_len <= target_nr - 3:
        score += construct_best_gamma_add(node5[0], node5[1])
    return score, individual


def evaluate_population_round45_parallel(first_path, start_layer, population, target_nr, n_jobs=8, executor=None, prefix_score=0.0, progress_desc='GA适应度评估'):
    """并行评估种群。"""
    tasks = [(first_path, start_layer, ind, target_nr, prefix_score) for ind in population]
    scored = []
    own_executor = executor is None
    if own_executor:
        executor = ProcessPoolExecutor(max_workers=n_jobs)
    try:
        futures = [executor.submit(evaluate_round45_individual_worker, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=progress_desc, ncols=90):
            scored.append(future.result())
    finally:
        if own_executor:
            executor.shutdown(wait=True)
    scored.sort(key=lambda x: x[0])
    return scored


def ga_find_initial_paths_parallel(first_path,
                                   start_layer,
                                   sorted_children,
                                   tree,
                                   target_nr=14,
                                   population_size=64,
                                   elite_size=16,
                                   generations=12,
                                   crossover_rate=0.9,
                                   mutation_rate=0.8,
                                   mutation_delta=4,
                                   num_samples=12,
                                   n_jobs=8,
                                   return_top_k=32,
                                   expand_threshold=2**(-24),
                                   save_tag='ga_round45',
                                   max_ga_minutes=GA_TIME_BUDGET_MIN,
                                   min_generations=GA_MIN_GENERATIONS,
                                   fast_compare_mode=GA_FAST_COMPARE_MODE):
    """
    GA baseline：固定前缀，在第4/5轮做两基因进化，并行评估适应度。
    输出可直接送入 paths_search_add_parallel 的 initial_paths。
    """
    if elite_size <= 0 or elite_size > population_size:
        raise ValueError('elite_size 必须在 (0, population_size] 范围内')

    population = prepare_ga_seed_population(
        first_path=first_path,
        start_layer=start_layer,
        sorted_children=sorted_children,
        population_size=population_size,
        delta=2,
        t1=14,
        expand_threshold=expand_threshold,
    )

    prefix_score = 0.0
    for i in range(start_layer - 1):
        diff_l_i, diff_r_i = first_path[i]
        child_l = first_path[i + 1][0]
        prefix_score += -log2(xdp_add(ror(diff_l_i, ALPHA()), diff_r_i, child_l))

    best_history = []
    generation_stats = []
    ga_start = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for gen in tqdm(range(generations), desc='GA总代进度', ncols=90):
            gen_start = time.time()
            elapsed_minutes = (time.time() - ga_start) / 60.0
            if gen >= min_generations and elapsed_minutes >= max_ga_minutes:
                print(f'  ⏱️ 达到GA时间预算 {max_ga_minutes:.1f} 分钟，提前停止在第 {gen}/{generations} 代')
                break

            print(f'\n===== GA 第 {gen + 1}/{generations} 代 =====')
            eval_start = time.time()
            scored = evaluate_population_round45_parallel(
                first_path, start_layer, population, target_nr,
                n_jobs=n_jobs, executor=executor, prefix_score=prefix_score,
                progress_desc=f'GA评估 G{gen+1}'
            )
            eval_time = time.time() - eval_start
            best_score = scored[0][0]
            mean_score = float(np.mean([s for s, _ in scored]))
            best_history.append(best_score)
            print(f'  最优分数: {best_score:.6f} | 平均分数: {mean_score:.6f}')

            elites = [copy.deepcopy(ind) for _, ind in scored[:elite_size]]
            next_population = elites[:]
            seen = {(ind['g4'], ind['g5']) for ind in next_population}
            offspring_target = population_size - len(next_population)
            cross_count = 0
            mutate_count = 0
            cross_time = 0.0
            mutate_time = 0.0
            breed_start = time.time()
            breed_attempts = 0
            max_breed_attempts = max(offspring_target * GA_BREED_MAX_ATTEMPTS_FACTOR, offspring_target + 8)

            with tqdm(total=offspring_target, desc=f'GA繁殖 G{gen+1}', ncols=90) as pbar_breed:
                while len(next_population) < population_size and breed_attempts < max_breed_attempts:
                    breed_attempts += 1
                    p1 = tournament_select(scored)
                    p2 = tournament_select(scored)
                    if random.random() < crossover_rate:
                        t_cross = time.time()
                        child = crossover_round45_individuals(first_path, start_layer, p1, p2, delta=2, t1=14, expand_threshold=expand_threshold)
                        cross_time += (time.time() - t_cross)
                        cross_count += 1
                    else:
                        child = copy.deepcopy(random.choice([p1, p2]))
                    # 越靠后代，变异规模逐步收敛，降低单代耗时
                    if fast_compare_mode:
                        curr_mutation_delta = 2
                        curr_num_samples = 4
                        curr_mutation_rate = min(0.35, mutation_rate)
                    else:
                        curr_mutation_delta = max(2, mutation_delta - (gen // 3))
                        curr_num_samples = max(6, num_samples - 2 * (gen // 4))
                        curr_mutation_rate = mutation_rate
                    if random.random() < curr_mutation_rate:
                        t_mut = time.time()
                        if fast_compare_mode:
                            mutated = mutate_round45_individual_fast(
                                first_path, start_layer, child,
                                mutation_delta=curr_mutation_delta,
                                num_samples=curr_num_samples,
                                expand_threshold=expand_threshold
                            )
                        else:
                            mutated = mutate_round45_individual(
                                first_path, start_layer, child,
                                mutation_delta=curr_mutation_delta,
                                num_samples=curr_num_samples,
                                expand_threshold=expand_threshold
                            )
                        mutate_time += (time.time() - t_mut)
                        mutate_count += 1
                        if mutated is not None:
                            child = mutated
                    key = (child['g4'], child['g5'])
                    if key in seen:
                        continue
                    next_population.append(child)
                    seen.add(key)
                    pbar_breed.update(1)
            # 若去重导致始终补不满，直接用精英/已生成个体回填，避免长时间卡在繁殖阶段
            while len(next_population) < population_size:
                base = random.choice(elites if elites else next_population)
                next_population.append(copy.deepcopy(base))
                pbar_breed.update(1)
            population = next_population
            breed_time = time.time() - breed_start
            gen_time = time.time() - gen_start
            stat = {
                'gen': gen + 1,
                'eval_s': eval_time,
                'breed_s': breed_time,
                'cross_s': cross_time,
                'mutate_s': mutate_time,
                'cross_n': cross_count,
                'mutate_n': mutate_count,
                'breed_attempts': breed_attempts,
                'total_s': gen_time
            }
            generation_stats.append(stat)
            print(
                f"  ⏱️ G{gen+1}耗时: total={gen_time:.1f}s | eval={eval_time:.1f}s | "
                f"breed={breed_time:.1f}s (cross={cross_time:.1f}s/{cross_count}次, "
                f"mutate={mutate_time:.1f}s/{mutate_count}次, attempts={breed_attempts})"
            )

        final_scored = evaluate_population_round45_parallel(
            first_path, start_layer, population, target_nr,
            n_jobs=n_jobs, executor=executor, prefix_score=prefix_score,
            progress_desc='GA最终评估'
        )
    initial_paths = [decode_round45_individual(first_path, start_layer, ind) for _, ind in final_scored[:return_top_k]]
    print(f'\nGA 结束：返回 {len(initial_paths)} 条 round-4/5 初始路径，最佳估计分数 {final_scored[0][0]:.6f}')
    if generation_stats:
        print('\n=== GA 分阶段耗时汇总(秒) ===')
        for st in generation_stats:
            print(
                f"G{st['gen']:02d}: total={st['total_s']:.1f}, eval={st['eval_s']:.1f}, "
                f"breed={st['breed_s']:.1f}, cross={st['cross_s']:.1f}({st['cross_n']}次), "
                f"mutate={st['mutate_s']:.1f}({st['mutate_n']}次), attempts={st['breed_attempts']}"
            )

    # 记录结果，便于和原 EA 对照
    root_diff_l = first_path[0][0]
    save_path = project_path("saved_trail", f"speck{2*word_size}_{target_nr}round_{root_diff_l}_{save_tag}.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'initial_paths': initial_paths,
                'best_history': best_history,
                'generation_stats': generation_stats,
                'final_population': final_scored[:return_top_k]
            }, f)
        print(f'📁 GA 初始路径已保存至: {save_path}')
    except Exception as e:
        print(f'⚠️ 保存 GA 结果失败: {e}')

    return initial_paths, final_scored


if __name__ == "__main__":
    for i in [12,14,16,18,20,21,23,25,27,29,31,33,36,38]:
        diff_l = 1 << i
        # diff_l = 0x8000000
        diff_r = 0x0
        print(f"diff_l = {diff_l}")
        nr = 8
        def make_tree():
            return defaultdict(list)  
        tree = make_tree()
        tree[(diff_l, diff_r)] = []     
        t1 = time.time()
        # current_paths = init.first_best_search_add_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=nr, top1=10, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-18), speck=2*word_size, population_size=2000, n_jobs=16)
        saved_tree_path1 = project_path("saved_trail", f"speck{2*word_size}_{nr}round_{diff_l}_evolution_data.pkl")
        with open(saved_tree_path1, "rb") as f:
            current_paths, tree = pickle.load(f)
        diff_nodes, counts = count_difference(current_paths, 4) 
        # print(f"第4层差分统计: {hex_2_bin(diff_nodes[0][0]), hex_2_bin(diff_nodes[1][0])}")  # 只打印前两个差分节点及其计数
        # print(f"第4层差分节点计数总和: {counts}")
        # ===== GA baseline 示例：固定前缀，在 round 4/5 两轮做交叉+变异 =====
        sorted_children = [node[0] for node in diff_nodes[:8]]
        initial_paths, ga_scored = ga_find_initial_paths_parallel(
            first_path=current_paths[0],
            start_layer=4,
            sorted_children=sorted_children,
            tree=tree,
            target_nr=11,
            population_size=64,
            elite_size=16,
            generations=12,
            mutation_delta=4,
            num_samples=12,
            n_jobs=32,
            return_top_k=32,
            max_ga_minutes=25,
            min_generations=2,
            save_tag='ga_round45'
        )
        paths_search_add_parallel(diff_l, diff_r, tree, nr=11, speck=2*word_size, initial_paths=initial_paths, n_jobs=32)
        t2 = time.time()
        print(f"总耗时: {(t2 - t1)/60:.2f} 分钟")

    # diff_l = 0x80
    # diff_r = 0x0
    # print(f"diff_l = {diff_l}")
    # nr = 12
    # def make_tree():
    #     return defaultdict(list)  
    # tree = make_tree()
    # tree[(diff_l, diff_r)] = []     
    # current_paths = init.first_best_search_sub_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=nr, top1=20, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-18), speck=2*word_size, population_size=2000, n_jobs=32)
    # saved_tree_path1 = f"./saved_trail/speck{2*word_size}_{nr}round_{diff_l}_evolution_data.pkl"
    # with open(saved_tree_path1, "rb") as f:
    #     current_paths, tree = pickle.load(f)
    # diff_nodes, counts = count_difference(current_paths, 6)  
    # # print(diff_nodes)
    # sorted_children = [diff_nodes[0][0], diff_nodes[1][0]]
    # initial_paths = find_initial_paths_parallel(current_paths[0], start_layer=6, sorted_children=sorted_children, tree=tree, top_n=12, n_jobs=32)
    # # print(initial_paths[0])
    # paths_search_sub_parallel(diff_l, diff_r, tree, nr=16, speck=2*word_size, initial_paths=initial_paths, n_jobs=32)






















        # diff_l = 0x8000000
        # diff_r = 0x0
        # print(f"diff_l = {diff_l}")
        # nr = 8
        # def make_tree():
        #     return defaultdict(list)  
        # tree = make_tree()
        # tree[(diff_l, diff_r)] = []     
        # current_paths = init.first_best_search_add_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=nr, top1=20, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-18), speck=2*word_size, population_size=2000, n_jobs=32)
        # saved_tree_path1 = f"./saved_trail/speck96_{nr}round_{diff_l}_evolution_data.pkl"
        # with open(saved_tree_path1, "rb") as f:
        #     current_paths, tree = pickle.load(f)
        # diff_nodes, counts = count_difference(current_paths, 4) 
        # print(diff_nodes)
        # sorted_children = [diff_nodes[0][0], diff_nodes[1][0]]
        # initial_paths = find_initial_paths_parallel(current_paths[0], start_layer=4, sorted_children=sorted_children, tree=tree, n_jobs=32)
        # print(initial_paths[0])
        # paths_search_add_parallel(diff_l, diff_r, tree, nr=14, speck=96, initial_paths=initial_paths, n_jobs=32)







    # diff_l = 4718600
    # diff_r = 34095112 
    # diff_l_new = ror(4718600, ALPHA())
    # # diff_l = 8800354961416
    # # diff_r = 8800090851400 
    # # diff_l_new = ror(8800354961416, ALPHA())
    # print(f"变异前节点: (L={hex(diff_l)}, R={hex(diff_r)})")
    # # sorted_children, sorted_scores = generate_possible_children_add(diff_l, diff_r, delta=2, t1=14)
    # # print(sorted_children[:10])
    # # print(hex(sorted_children[0]))
    # # print(hex_2_bin(sorted_children[0]))
    # # print(hex_2_bin(0x772400040))
    # # print(hex_2_bin(49161437248))
    # # candidates, sorted_scores = vary_difference(ror(diff_l, ALPHA()), diff_r, 49161437248, delta=4, num_samples=100)
    # # print(candidates)
    # # print(0x8000e080808)
    # # print(0x8001e080808)
    # # print(0x8003e080808)
    # # print(0x800fe080808)
    # sorted_children = [0x80002080808, 0x80006080808]
    # # sorted_children = [0x180006080808, 0x8000e080808, 0x80006081808, 0x80006080818, 0x8001e080808, 0x18000e080808, 0x80006088808, 0x180006080818, 0x180006081808, 0x8000e081808, 0x80006089808]
    # candidates = expand_population_with_rank(sorted_children[:10], diff_l_new, diff_r, delta=4, num_samples=100, require_multi_parent=True, top_n=12, output_as_list=True)
    # print(candidates)
    # node_info, origin_map = expand_population_with_rank(candidates, diff_l_new, diff_r, delta=4, num_samples=100, require_multi_parent=True, top_n=144)
    # print(f"{'节点':<10} {'分数':<8} {'来源(父节点,排名,分数)'}")
    # print("-" * 60)
    # for node, info in node_info.items():
    #     parents = ", ".join([f"({hex(p)},{r})" for (p, r) in info["parents"]]) if info["parents"] else "-"
    #     score = f"{info['score']:.2f}" if info["score"] is not None else "-"
    #     print(f"{hex(node):<10} {score:<8} {parents}")








