import os
import json
import pickle
import time
import math
from itertools import combinations
from multiprocessing import Pool
import sys
sys.stdout.reconfigure(line_buffering=True)
from math import *
import itertools
from collections import defaultdict, OrderedDict, Counter
import copy
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import statistics
from math import log2
import ea_initial_population as init
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


# ============================================================================
#  This module is meant to be pasted into / imported from the same namespace
#  as your SPECK search code. It assumes the following names already exist:
#    ham_w, ALPHA, BETA, rol, ror, xdp_add,
#    generate_possible_children_add, evaluate_path_add,
#    vary_difference, rank_children, expand_population_with_rank.
# ============================================================================



word_size = 48
MASK_VAL = 2 ** word_size - 1


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

def vary_difference(alpha, beta, gamma, delta, num_samples=10, include_free_bits=True):
    #变异差分    
    alpha_bits = hex_2_bin(alpha)
    beta_bits = hex_2_bin(beta)
    gamma_bits = hex_2_bin(gamma)
    eq_positions = []
    for i in range(word_size - 1):
        if alpha_bits[i] == beta_bits[i] == gamma_bits[i]:
            eq_positions.append(i)
    free_positions = []
    if include_free_bits:
        # rule (b) 对应的“自由位”：alpha 与 beta 在该位不同
        for i in range(word_size - 1):
            if alpha_bits[i] != beta_bits[i]:
                free_positions.append(i)
    mutable_positions = sorted(set(eq_positions) | set(free_positions))
    positionsLists = []
    candidates = []
    tried = set()
    for i in range(delta + 1):
        itera = itertools.combinations(mutable_positions, i)
        for comb in itera:
            positionsLists.append(list(comb))
            if i > 0 and comb[0] > i:
                positionsLists.append(list(range(comb[0] - i + 1, comb[0] + 1)))
    # print("生成扰动位置：", positionsLists)
    for pos_combo in positionsLists:
        new_bits = gamma_bits[:]
        for i in pos_combo:
            new_bits[i] ^= 1  # 扰动
        gamma_prime = sum(new_bits[i] << (word_size-1-i) for i in range(word_size))
        if gamma_prime in tried:
            continue
        tried.add(gamma_prime)
        if judge_valid(alpha, beta, gamma_prime):
            prob = xdp_add(alpha, beta, gamma_prime)
            orig_prob = xdp_add(alpha, beta, gamma)
            log_ratio = -log2(prob / orig_prob)
            if (delta - log_ratio) < delta:  
                candidates.append((gamma_prime))
    candidates, sorted_scores = get_children_score_rank_add(alpha, beta, candidates)
    return candidates[:num_samples], sorted_scores[:num_samples]

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

def save_paths_to_txt(paths, txt_path):
    """将路径列表保存为可读文本，每条路径一段。"""
    with open(txt_path, "w") as f:
        f.write("=== Initial Paths ===\n")
        f.write(f"total_paths: {len(paths)}\n")
        f.write(f"path_length: {len(paths[0]) if paths else 0}\n\n")
        for i, path in enumerate(paths, 1):
            nodes = " -> ".join([f"({hex(l)}, {hex(r)})" for (l, r) in path])
            f.write(f"[{i}] {nodes}\n")


def find_initial_paths(first_path, start_layer, sorted_children, tree, top1=20, t1=14, t2=35, expand_threshold=2**(-24), top_n=12, vary_times=2, require_multi_parent=True):
    """根据变异结果生成初始路径"""
    diff_l, diff_r = first_path[start_layer - 1]
    root_diff_l = first_path[0][0]
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
        new_vary_son, sorted_scores = vary_difference(ror(new_node[0], ALPHA()), new_node[1], new_son_node[0],delta=6, num_samples=10)
        for son in new_vary_son:
            new_son_path = new_path.copy()
            son_vary_node = (son, son ^ rol(new_node[1], BETA()))
            new_son_path.append(son_vary_node)
            initial_paths.append(new_son_path)
        new_path.append(new_son_node)
        initial_paths.append(new_path)
    print(f"生成初始路径 {len(initial_paths)} 条，每条长度为 {len(initial_paths[0]) if initial_paths else 0}")
    # === 保存初始路径 ===
    save_dir = "./saved_initial_paths"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/speck{2*word_size}_{start_layer+2}round_{root_diff_l}_initial_paths.pkl"
    txt_path = save_path.replace(".pkl", ".txt")
    with open(save_path, 'wb') as f:
        pickle.dump(initial_paths, f)
    save_paths_to_txt(initial_paths, txt_path)

    print(f"\n📁 结果已保存至:\n  {save_path}\n  {txt_path}")
    print("🎯 搜索任务完成！")    
    return initial_paths

def process_candidate(args):
    cand, diff_r, prefix, t1, t2, top1, expand_threshold = args
    local_tree = {}
    local_paths = []
    # 基础前缀：到 new_node 为止
    base_path = prefix.copy()
    new_node = (cand, cand ^ rol(diff_r, BETA()))
    base_path.append(new_node)
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
    # Step 3a: 保存 exact child path（长度仍然是 6 轮）
    for child_node in local_tree[new_node][:2]:
        exact_path = base_path.copy()
        exact_path.append(child_node)
        local_paths.append(exact_path)
    # Step 3b: 只对前两个子节点做二次变异
    # 注意：变异得到的是“同层替代节点”，不是再往后多加一层
    for new_son_node in local_tree[new_node][:2]:
        new_vary_son, sorted_scores = vary_difference(ror(new_node[0], ALPHA()),new_node[1],new_son_node[0],delta=4,num_samples=10)
        for son in new_vary_son:
            mutated_path = base_path.copy()
            son_vary_node = (son, son ^ rol(new_node[1], BETA()))
            mutated_path.append(son_vary_node)
            local_paths.append(mutated_path)
    return local_tree, local_paths

def find_initial_paths_parallel(first_path, start_layer, sorted_children, tree, top1=20, t1=14, t2=35, expand_threshold=2**(-24), top_n=12, vary_times=2, n_jobs=8, require_multi_parent=True):
    """根据变异结果生成初始路径（并行版本）"""
    diff_l, diff_r = first_path[start_layer - 1]
    root_diff_l = first_path[0][0]
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
    # === 保存初始路径 ===
    save_dir = "./saved_initial_paths"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/speck{2*word_size}_{start_layer+2}round_{root_diff_l}_initial_paths_{require_multi_parent}.pkl"
    txt_path = save_path.replace(".pkl", ".txt")
    with open(save_path, 'wb') as f:
        pickle.dump(initial_paths, f)
    save_paths_to_txt(initial_paths, txt_path)
    print(f"\n📁 初始路径已保存至:\n  {save_path}\n  {txt_path}")
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
    # save_path1 = f"./saved_ea_txt/speck{speck}/speck{speck}_result_ea_{nr}round_{diff_l}_5.txt"
    save_path1 = project_path("saved_ea_txt", f"speck{speck}", f"speck{speck}_result_ea_{nr}round_{diff_l}_single-parent.txt")
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
    save_path2 = project_path("saved_trail", f"speck{speck}_{nr}round_{root_diff_l}_single-parent.pkl")
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
            evaluated_paths.sort(key=lambda x: x[0])  # 升序
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


def run_one_experiment(diff_l=0x8000000, diff_r=0x0, base_nr=8, search_nr=11, n_jobs=32):
    print(f"diff_l = {diff_l}")
    tree = defaultdict(list)
    tree[(diff_l, diff_r)] = []
    t1 = time.time()
    saved_tree_path1 = project_path("saved_trail", f"speck{2*word_size}_{base_nr}round_{diff_l}_evolution_data.pkl")
    with open(saved_tree_path1, "rb") as f:
        current_paths, tree = pickle.load(f)
    diff_nodes, counts = count_difference(current_paths, 4)
    sorted_children = [diff_nodes[0][0], diff_nodes[1][0]]
    initial_paths_path = f"./saved_initial_paths/speck{2*word_size}_6round_{diff_l}_initial_paths.pkl"
    with open(initial_paths_path, "rb") as f:
        initial_paths = pickle.load(f)
    paths_search_add_parallel(diff_l, diff_r, tree, nr=search_nr, speck=2*word_size, initial_paths=initial_paths, n_jobs=n_jobs)
    t2 = time.time()
    elapsed_sec = t2 - t1
    print(f"总耗时: {elapsed_sec/60:.2f} 分钟")
    return elapsed_sec


def state_key(state):
    return (int(state[0]), int(state[1]))


def serialize_path(path):
    return tuple((int(l), int(r)) for (l, r) in path)


def path_terminal_state(path):
    return state_key(path[-1])


def terminal_hamming_distance(a, b):
    return ham_w(a[0] ^ b[0]) + ham_w(a[1] ^ b[1])


def average_pairwise_state_distance(states, max_pairs=50000):
    n = len(states)
    if n <= 1:
        return 0.0
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        pairs_iter = combinations(states, 2)
    else:
        m = min(n, int((2 * max_pairs) ** 0.5) + 2)
        pairs_iter = combinations(states[:m], 2)
    dist_sum = 0.0
    cnt = 0
    for s1, s2 in pairs_iter:
        dist_sum += terminal_hamming_distance(s1, s2)
        cnt += 1
    return dist_sum / cnt if cnt else 0.0



def shannon_entropy_from_counter(counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0, 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log2(p)
    max_ent = math.log2(len(counter)) if len(counter) > 1 else 0.0
    norm_ent = ent / max_ent if max_ent > 0 else 0.0
    return ent, norm_ent



def build_state_summary(scored_paths, selected_state_keys=None, topk_entropy=200):
    selected_state_keys = selected_state_keys or set()
    path_cnt = len(scored_paths)
    terminal_states = [path_terminal_state(path) for _, path in scored_paths]
    uniq_states = list(dict.fromkeys(terminal_states))
    focus = scored_paths[: min(topk_entropy, path_cnt)]
    focus_states = [path_terminal_state(path) for _, path in focus]
    focus_counter = Counter(focus_states)
    ent, norm_ent = shannon_entropy_from_counter(focus_counter)

    state2scores = defaultdict(list)
    for rank, (score, path) in enumerate(scored_paths, start=1):
        sk = path_terminal_state(path)
        state2scores[sk].append((score, rank))

    sorted_states = sorted(
        state2scores.items(),
        key=lambda kv: (min(x[0] for x in kv[1]), kv[0][0], kv[0][1])
    )
    total_states = max(len(sorted_states), 1)
    state_records = []
    for state_rank, (sk, items) in enumerate(sorted_states, start=1):
        scores = [x[0] for x in items]
        ranks = [x[1] for x in items]
        state_records.append({
            "state": [sk[0], sk[1]],
            "path_count": len(items),
            "best_score": min(scores),
            "mean_score": sum(scores) / len(scores),
            "best_path_rank": min(ranks),
            "state_rank": state_rank,
            "state_rank_pct": state_rank / total_states,
            "selected": sk in selected_state_keys,
        })

    return {
        "path_count": path_cnt,
        "unique_state_count": len(set(terminal_states)),
        "avg_pairwise_hamming": average_pairwise_state_distance(uniq_states),
        "topk_entropy": ent,
        "topk_entropy_norm": norm_ent,
        "topk_unique_states": len(focus_counter),
        "state_records": state_records,
    }



def snapshot_from_paths(paths, score_fn, selected_paths=None, topk_entropy=200):
    scored = [(score_fn(p), p) for p in paths]
    scored.sort(key=lambda x: (x[0], serialize_path(x[1])))
    selected_state_keys = set()
    if selected_paths is not None:
        selected_state_keys = {path_terminal_state(p) for p in selected_paths}
    return build_state_summary(scored, selected_state_keys=selected_state_keys, topk_entropy=topk_entropy)



def write_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



def dump_pickle(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)



def _score_fn_from_nr(nr):
    return lambda p: evaluate_path_add(p, nr)



def annotate_final_good_states(round_logs, final_paths, nr, good_top_m=100, score_margin=None):
    score_fn = _score_fn_from_nr(nr)
    final_scored = [(score_fn(p), p) for p in final_paths]
    final_scored.sort(key=lambda x: (x[0], serialize_path(x[1])))
    if not final_scored:
        return round_logs

    best_score = final_scored[0][0]
    if score_margin is not None:
        good_paths = [p for s, p in final_scored if s <= best_score + score_margin]
    else:
        good_paths = [p for _, p in final_scored[: min(good_top_m, len(final_scored))]]

    good_states_by_round = defaultdict(set)
    for path in good_paths:
        for ridx, state in enumerate(path):
            good_states_by_round[ridx].add(state_key(state))

    for log in round_logs:
        r = log["round"]
        good_state_set = good_states_by_round.get(r, set())
        bridge_scores = []
        bridge_total = 0
        bridge_selected = 0
        bridge_pruned = 0
        for rec in log["pre_selection"]["state_records"]:
            sk = tuple(rec["state"])
            on_good = sk in good_state_set
            rec["on_final_good_trail"] = on_good
            if on_good:
                bridge_total += 1
                bridge_scores.append(rec["state_rank_pct"])
                if rec["selected"]:
                    bridge_selected += 1
                else:
                    bridge_pruned += 1
        bridge_scores_sorted = sorted(bridge_scores)
        med = None
        if bridge_scores_sorted:
            med = bridge_scores_sorted[len(bridge_scores_sorted) // 2]
        log["bridge_stats"] = {
            "final_good_round_state_count": len(good_state_set),
            "bridge_total": bridge_total,
            "bridge_selected": bridge_selected,
            "bridge_pruned": bridge_pruned,
            "bridge_selection_rate": bridge_selected / bridge_total if bridge_total else 0.0,
            "bridge_rank_pct_mean": sum(bridge_scores) / len(bridge_scores) if bridge_scores else None,
            "bridge_rank_pct_median": med,
        }
    return round_logs



def compare_bridge_states_against_baseline(baseline_round_logs, other_round_logs):
    baseline_by_round = {x["round"]: x for x in baseline_round_logs}
    rows = []
    for log in other_round_logs:
        r = log["round"]
        if r not in baseline_by_round:
            continue
        base_log = baseline_by_round[r]
        base_selected = {
            tuple(rec["state"]) for rec in base_log["pre_selection"]["state_records"] if rec["selected"]
        }
        base_pre = {tuple(rec["state"]): rec for rec in base_log["pre_selection"]["state_records"]}
        other_bridge_selected = {
            tuple(rec["state"])
            for rec in log["pre_selection"]["state_records"]
            if rec.get("on_final_good_trail") and rec["selected"]
        }
        rescued = [s for s in other_bridge_selected if s not in base_selected]
        rescued_from_pruned = [s for s in rescued if s in base_pre and not base_pre[s]["selected"]]
        rows.append({
            "round": r,
            "other_bridge_selected": len(other_bridge_selected),
            "rescued_vs_baseline": len(rescued),
            "rescued_from_baseline_pruned": len(rescued_from_pruned),
            "rescued_states": rescued,
        })
    return rows



def _candidate_record(cand, source, iteration_idx=0, score=None, parents=None, occurrence_idx=0):
    parents = parents or set()
    distinct_parents = sorted({int(p) for (p, _) in parents})
    local_ranks = sorted({int(rank) for (_, rank) in parents})
    return {
        "cand": int(cand),
        "candidate_source": source,
        "iteration_idx": int(iteration_idx),
        "candidate_score": None if score is None else float(score),
        "parent_support": len(distinct_parents) if distinct_parents else (1 if source == "exact" else 0),
        "parent_ids": distinct_parents,
        "parent_local_ranks": local_ranks,
        "best_local_rank": min(local_ranks) if local_ranks else None,
        "occurrence_idx": int(occurrence_idx),
    }



def process_candidate_logged_two_stage(args):
    """
    与你原始 process_candidate 保持相同搜索语义，只额外记录 metadata。
    关键点：
    1) 先用 cand 构造 new_node；
    2) 生成 new_node 的 exact children；
    3) 仅对前两个 child 做“第二轮连续变异”；
    4) 返回的 local_paths 与原始代码保持一致。
    """
    cand, diff_r, prefix, t1, t2, top1, expand_threshold, cand_meta = args
    local_tree = {}
    local_paths = []
    local_meta = []

    # 基础前缀：到 new_node 为止
    base_path = prefix.copy()
    new_node = (cand, cand ^ rol(diff_r, BETA()))
    base_path.append(new_node)
    local_tree[new_node] = []

    # Step 1: 生成子节点（与原代码一致）
    x_ham = ham_w(new_node[0])
    y_ham = ham_w(new_node[1])
    if x_ham <= t1 and y_ham <= t1:
        possible_children, sorted_scores = generate_possible_children_add(new_node[0], new_node[1], delta=2, t1=t1)
        top_children = possible_children[:top1]
        for child_rank, child in enumerate(top_children, start=1):
            z_ham = ham_w(child)
            if (x_ham + y_ham + z_ham <= t2) and (
                xdp_add(ror(new_node[0], ALPHA()), new_node[1], child) >= expand_threshold
            ):
                l_new = child
                r_new = child ^ rol(new_node[1], BETA())
                child_node = (l_new, r_new)
                local_tree[new_node].append(child_node)

    # Step 2: 若该节点无子节点，则跳过
    if len(local_tree[new_node]) == 0:
        return local_tree, local_paths, local_meta

    # Step 3a: 保存 exact child path（与你原实现一致）
    for child_pos, child_node in enumerate(local_tree[new_node][:2], start=1):
        exact_path = base_path.copy()
        exact_path.append(child_node)
        local_paths.append(exact_path)
        local_meta.append({
            "candidate_value": int(cand),
            "candidate_state": [int(new_node[0]), int(new_node[1])],
            "terminal_state": [int(child_node[0]), int(child_node[1])],
            "candidate_source": cand_meta["candidate_source"],
            "candidate_mutation_stage": int(cand_meta["candidate_mutation_stage"]),
            "candidate_occurrence_idx": int(cand_meta["candidate_occurrence_idx"]),
            "candidate_local_rank": cand_meta.get("candidate_local_rank"),
            "path_origin": "exact_child",
            "second_stage_mutation": False,
            "child_pos": child_pos,
        })

    # Step 3b: 只对前两个子节点做第二轮连续变异（与你原实现一致）
    # 注意：这里是“同层替代节点”，不是再往后加一层
    for child_pos, new_son_node in enumerate(local_tree[new_node][:2], start=1):
        new_vary_son, sorted_scores = vary_difference(
            ror(new_node[0], ALPHA()),
            new_node[1],
            new_son_node[0],
            delta=4,
            num_samples=10,
        )
        for mut_rank, son in enumerate(new_vary_son, start=1):
            mutated_path = base_path.copy()
            son_vary_node = (son, son ^ rol(new_node[1], BETA()))
            mutated_path.append(son_vary_node)
            local_paths.append(mutated_path)
            local_meta.append({
                "candidate_value": int(cand),
                "candidate_state": [int(new_node[0]), int(new_node[1])],
                "terminal_state": [int(son_vary_node[0]), int(son_vary_node[1])],
                "candidate_source": cand_meta["candidate_source"],
                "candidate_mutation_stage": int(cand_meta["candidate_mutation_stage"]),
                "candidate_occurrence_idx": int(cand_meta["candidate_occurrence_idx"]),
                "candidate_local_rank": cand_meta.get("candidate_local_rank"),
                "path_origin": "mutated_child",
                "second_stage_mutation": True,
                "child_pos": child_pos,
                "second_stage_mut_rank": mut_rank,
            })

    return local_tree, local_paths, local_meta



def _safe_write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)




def _summarize_initial_path_meta(path_meta):
    path_origin_counter = Counter(row["path_origin"] for row in path_meta)
    terminal_state_origin_counter = defaultdict(Counter)
    candidate_source_counter = Counter()
    candidate_support = []
    for row in path_meta:
        tstate = tuple(row["terminal_state"])
        terminal_state_origin_counter[tstate][row["path_origin"]] += 1
        candidate_source_counter[row["candidate_source"]] += 1
        if row.get("parent_support") is not None:
            candidate_support.append(row["parent_support"])
    state_origin_summary = [
        {
            "terminal_state": [s[0], s[1]],
            "origin_breakdown": dict(counter),
            "total_count": int(sum(counter.values())),
        }
        for s, counter in sorted(terminal_state_origin_counter.items())
    ]
    return {
        "path_origin_counter": dict(path_origin_counter),
        "candidate_source_counter": dict(candidate_source_counter),
        "candidate_support_mean": sum(candidate_support) / len(candidate_support) if candidate_support else None,
        "state_origin_summary": state_origin_summary,
    }



def find_initial_paths_parallel_logged_two_stage(
    first_path,
    start_layer,
    sorted_children,
    tree,
    top1=20,
    t1=14,
    t2=35,
    expand_threshold=2**(-24),
    top_n=12,
    vary_times=2,
    n_jobs=8,
    require_multi_parent=True,
    nr_total=None,
    method_name="egbs_mp",
    metrics_dir=project_path("saved_metrics"),
):
    """
    完全保持你原 find_initial_paths_parallel 的搜索语义：
    - 第一阶段：对 sorted_children 连续做 vary_times 轮 candidate 扩展；
    - 第二阶段：在 process_candidate 里再对前两个 child 做一轮连续变异。

    这里只增加记录，不改变候选生成顺序、不去重、不改排序、不改扩展逻辑。
    """
    diff_l, diff_r = first_path[start_layer - 1]
    root_diff_l = first_path[0][0]
    diff_l_new = ror(diff_l, ALPHA())

    # 记录“第一阶段 candidate 变异”的来源信息，但不改变原始语义
    candidate_meta_records = []
    original_len = len(sorted_children)
    for occ_idx, cand in enumerate(list(sorted_children)):
        candidate_meta_records.append({
            "candidate_value": int(cand),
            "candidate_source": "exact_candidate",
            "candidate_mutation_stage": 0,
            "candidate_occurrence_idx": occ_idx,
            "candidate_local_rank": occ_idx + 1,
        })

    candidates = sorted_children
    stage1_logs = []
    next_occ_idx = original_len
    for i in range(vary_times):
        candidates = expand_population_with_rank(
            candidates,
            diff_l_new,
            diff_r,
            delta=4,
            num_samples=100,
            require_multi_parent=require_multi_parent,
            top_n=top_n ** (i + 1),
            output_as_list=True,
        )

        stage1_logs.append({
            "candidate_mutation_stage": i + 1,
            "new_candidate_count": len(candidates),
            "require_multi_parent": bool(require_multi_parent),
            "top_n_used": int(top_n ** (i + 1)),
        })

        for local_rank, cand in enumerate(candidates, start=1):
            candidate_meta_records.append({
                "candidate_value": int(cand),
                "candidate_source": f"candidate_mutation_stage_{i + 1}",
                "candidate_mutation_stage": i + 1,
                "candidate_occurrence_idx": next_occ_idx,
                "candidate_local_rank": local_rank,
            })
            next_occ_idx += 1

        sorted_children.extend(candidates)

    # 到这里，candidate_meta_records 的长度应与 sorted_children 完全一致
    assert len(candidate_meta_records) == len(sorted_children), (
        f"metadata 数量 {len(candidate_meta_records)} 与候选数量 {len(sorted_children)} 不一致"
    )

    prefix = first_path[:start_layer]
    tasks = [
        (cand, diff_r, prefix, t1, t2, top1, expand_threshold, cand_meta)
        for cand, cand_meta in zip(sorted_children, candidate_meta_records)
    ]

    initial_paths = []
    local_trees = []
    path_meta = []
    with Pool(processes=n_jobs) as pool:
        for local_tree, local_paths, local_meta in tqdm(
            pool.imap_unordered(process_candidate_logged_two_stage, tasks),
            total=len(tasks),
            desc="生成初始路径进度",
            unit="节点",
            ncols=100,
        ):
            local_trees.append(local_tree)
            initial_paths.extend(local_paths)
            path_meta.extend(local_meta)

    # 汇总 tree（与你原代码一致）
    for local_tree in local_trees:
        for k, v in local_tree.items():
            if k not in tree:
                tree[k] = []
            tree[k].extend(v)

    print(f"生成初始路径 {len(initial_paths)} 条，每条长度为 {len(initial_paths[0]) if initial_paths else 0}")

    # 原始保存逻辑保留
    save_dir = "./saved_initial_paths"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/speck{2*word_size}_{start_layer+2}round_{root_diff_l}_initial_paths_{require_multi_parent}.pkl"
    txt_path = save_path.replace(".pkl", ".txt")
    with open(save_path, 'wb') as f:
        pickle.dump(initial_paths, f)
    save_paths_to_txt(initial_paths, txt_path)
    print(f"\n📁 初始路径已保存至:\n  {save_path}\n  {txt_path}")

    # -------- 新增：保存 mutation layer 的记录 --------
    meta_dir = os.path.join(metrics_dir, f"speck{2*word_size}", method_name, f"r{nr_total if nr_total is not None else 'unknown'}")
    os.makedirs(meta_dir, exist_ok=True)

    origin_counter = Counter(row["path_origin"] for row in path_meta)
    candidate_source_counter = Counter(row["candidate_source"] for row in path_meta)
    candidate_stage_counter = Counter(str(row["candidate_mutation_stage"]) for row in path_meta)

    payload = {
        "method_name": method_name,
        "root_diff_l": int(root_diff_l),
        "start_layer": int(start_layer),
        "nr_total": None if nr_total is None else int(nr_total),
        "require_multi_parent": bool(require_multi_parent),
        "candidate_stage1_logs": stage1_logs,
        "candidate_count_total": len(sorted_children),
        "initial_path_count": len(initial_paths),
        "path_origin_counter": dict(origin_counter),
        "candidate_source_counter": dict(candidate_source_counter),
        "candidate_stage_counter": dict(candidate_stage_counter),
        "candidate_meta_records": candidate_meta_records,
        "path_meta": path_meta,
    }

    json_path = os.path.join(meta_dir, f"diff_{root_diff_l}_initial_population_two_stage.json")
    pkl_path = os.path.join(meta_dir, f"diff_{root_diff_l}_initial_population_two_stage.pkl")
    _safe_write_json(json_path, {
        "method_name": method_name,
        "root_diff_l": int(root_diff_l),
        "start_layer": int(start_layer),
        "nr_total": None if nr_total is None else int(nr_total),
        "require_multi_parent": bool(require_multi_parent),
        "candidate_stage1_logs": stage1_logs,
        "candidate_count_total": len(sorted_children),
        "initial_path_count": len(initial_paths),
        "path_origin_counter": dict(origin_counter),
        "candidate_source_counter": dict(candidate_source_counter),
        "candidate_stage_counter": dict(candidate_stage_counter),
    })
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n📁 两阶段变异记录已保存至:\n  {json_path}\n  {pkl_path}")
    return initial_paths, payload



def expand_single_path_add_logged(args):
    path, tree, top1, top2, delta, t1, t2, expand_threshold = args
    next_paths = []
    expanded_nodes = 0
    last_node = path[-1]
    if len(tree[last_node]) == 0:
        x_ham = ham_w(last_node[0])
        y_ham = ham_w(last_node[1])
        if x_ham <= t1 and y_ham <= t1:
            possible_children, _ = generate_possible_children_add(last_node[0], last_node[1], delta, t1)
            for child in possible_children[:top1]:
                z_ham = ham_w(child)
                if (x_ham + y_ham + z_ham <= t2) and (xdp_add(ror(last_node[0], ALPHA()), last_node[1], child) >= expand_threshold):
                    child_node = (child, child ^ rol(last_node[1], BETA()))
                    tree[last_node].append(child_node)
                    expanded_nodes += 1
    for child_node in tree[last_node][:top2]:
        new_path = path.copy()
        new_path.append(child_node)
        next_paths.append(new_path)
    return next_paths, expanded_nodes



def paths_search_add_parallel_logged(
    diff_l,
    diff_r,
    tree,
    nr,
    top1=10,
    top2=10,
    delta=2,
    t1=14,
    t2=30,
    expand_threshold=2**(-24),
    speck=64,
    population_size=2000,
    initial_paths=None,
    n_jobs=8,
    method_name="egbs_mp",
    metrics_dir=project_path("saved_metrics"),
    topk_entropy=200,
    good_top_m=100,
    good_score_margin=None,
    initial_population_payload=None,
):
    if initial_paths is not None and len(initial_paths) > 0:
        current_paths = initial_paths.copy()
        start_layer = len(initial_paths[0]) - 1
        print(f"=== 从 {len(initial_paths)} 条部分初始路径继续搜索 (起始层: {start_layer + 1}/{nr}) ===")
    else:
        root = (diff_l, diff_r)
        current_paths = [[root]]
        start_layer = 0
        print(f"=== 从根节点开始分层搜索 (总层数: {nr}) ===")

    total_start_time = time.time()
    round_logs = []
    metrics_base = os.path.join(metrics_dir, f"speck{speck}", method_name, f"r{nr}")
    os.makedirs(metrics_base, exist_ok=True)
    jsonl_path = os.path.join(metrics_base, f"diff_{diff_l}_round_metrics.jsonl")
    pkl_path = os.path.join(metrics_base, f"diff_{diff_l}_round_metrics.pkl")

    init_snapshot = snapshot_from_paths(
        current_paths,
        score_fn=_score_fn_from_nr(nr),
        selected_paths=current_paths,
        topk_entropy=topk_entropy,
    )
    init_record = {
        "method": method_name,
        "round": start_layer,
        "phase": "initial_beam",
        "pre_selection": init_snapshot,
        "post_selection": init_snapshot,
        "layer_elapsed_sec": 0.0,
        "expanded_nodes": 0,
        "candidate_path_count": len(current_paths),
        "selected_path_count": len(current_paths),
    }
    round_logs.append(init_record)
    write_jsonl(jsonl_path, init_record)

    for layer in range(start_layer, nr):
        round_idx = layer + 1
        layer_start = time.time()
        print(f"\n--- 正在处理第 {round_idx} 层 (当前路径数: {len(current_paths)}) ---")
        next_paths = []
        expanded_nodes_total = 0
        task_args = [(path, tree, top1, top2, delta, t1, t2, expand_threshold) for path in current_paths]
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(expand_single_path_add_logged, args) for args in task_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"第 {round_idx} 层处理中", ncols=90):
                expanded, expanded_nodes = future.result()
                next_paths.extend(expanded)
                expanded_nodes_total += expanded_nodes
        elapsed = time.time() - layer_start
        print(f"  ✅ 本层扩展 {expanded_nodes_total} 个新节点，共生成 {len(next_paths)} 条路径，用时 {elapsed:.1f}s")

        if not next_paths:
            empty_record = {
                "method": method_name,
                "round": round_idx,
                "phase": "empty",
                "pre_selection": {
                    "path_count": 0,
                    "unique_state_count": 0,
                    "avg_pairwise_hamming": 0.0,
                    "topk_entropy": 0.0,
                    "topk_entropy_norm": 0.0,
                    "topk_unique_states": 0,
                    "state_records": [],
                },
                "post_selection": None,
                "layer_elapsed_sec": elapsed,
                "expanded_nodes": expanded_nodes_total,
                "candidate_path_count": 0,
                "selected_path_count": 0,
            }
            round_logs.append(empty_record)
            write_jsonl(jsonl_path, empty_record)
            current_paths = []
            print("⚠️ 无可扩展路径，搜索提前结束。")
            break

        pre_snapshot = snapshot_from_paths(
            next_paths,
            score_fn=_score_fn_from_nr(nr),
            selected_paths=None,
            topk_entropy=topk_entropy,
        )

        selected_paths = next_paths
        if layer >= 3 and len(next_paths) > population_size:
            print(f"  正在进行路径选择 (当前路径数: {len(next_paths)})...")
            evaluated = [(evaluate_path_add(path, nr), path) for path in next_paths]
            evaluated.sort(key=lambda x: (x[0], serialize_path(x[1])))
            selected_paths = [path for score, path in evaluated[:population_size]]
            print(f"  ✅ 选择完成: 保留前{population_size}条最优路径 (最低分: {evaluated[0][0]:.6f})")

        post_snapshot = snapshot_from_paths(
            selected_paths,
            score_fn=_score_fn_from_nr(nr),
            selected_paths=selected_paths,
            topk_entropy=topk_entropy,
        )
        selected_state_keys = {path_terminal_state(p) for p in selected_paths}
        for rec in pre_snapshot["state_records"]:
            rec["selected"] = tuple(rec["state"]) in selected_state_keys

        record = {
            "method": method_name,
            "round": round_idx,
            "phase": "search",
            "pre_selection": pre_snapshot,
            "post_selection": post_snapshot,
            "layer_elapsed_sec": elapsed,
            "expanded_nodes": expanded_nodes_total,
            "candidate_path_count": len(next_paths),
            "selected_path_count": len(selected_paths),
        }
        round_logs.append(record)
        write_jsonl(jsonl_path, record)
        current_paths = selected_paths

        if not current_paths:
            print("⚠️ 无可扩展路径，搜索提前结束。")
            break

    total_elapsed = time.time() - total_start_time
    print(f"\n🎯 搜索完成，总耗时 {total_elapsed/60:.2f} 分钟")

    round_logs = annotate_final_good_states(
        round_logs,
        final_paths=current_paths,
        nr=nr,
        good_top_m=good_top_m,
        score_margin=good_score_margin,
    )

    payload = {
        "method": method_name,
        "diff_l": int(diff_l),
        "diff_r": int(diff_r),
        "nr": int(nr),
        "population_size": int(population_size),
        "total_elapsed_sec": float(total_elapsed),
        "round_logs": round_logs,
        "final_paths": current_paths,
        "initial_population_payload": initial_population_payload,
    }
    dump_pickle(pkl_path, payload)

    save_txt_dir = project_path("saved_ea_txt", f"speck{speck}")
    save_trail_dir = project_path("saved_trail")
    os.makedirs(save_txt_dir, exist_ok=True)
    os.makedirs(save_trail_dir, exist_ok=True)

    save_txt = os.path.join(save_txt_dir, f"speck{speck}_result_{method_name}_{nr}round_{diff_l}.txt")
    with open(save_txt, "w", encoding="utf-8") as f:
        f.write("=== 差分路径搜索结果 ===\n")
        f.write(f"Speck版本: {speck}\n")
        f.write(f"方法: {method_name}\n")
        f.write(f"总层数: {nr}\n")
        f.write(f"起始层: {start_layer + 1}\n")
        f.write(f"最终保留路径数: {len(current_paths)}\n")
        f.write(f"总耗时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)\n\n")
        for i, path in enumerate(current_paths):
            score = evaluate_path_add(path, nr)
            f.write(f"\n---- 路径 {i + 1} (总分: {score:.4f}) ----\n")
            for j in range(len(path) - 1):
                dl, dr = path[j]
                child = path[j + 1][0]
                node_score = -math.log2(xdp_add(ror(dl, ALPHA()), dr, child))
                f.write(f"层 {j}: ({hex(dl)}, {hex(dr)}) | 分数: {node_score:.4f}\n")
            f.write(f"层 {len(path) - 1}: ({hex(path[-1][0])}, {hex(path[-1][1])})\n")

    save_pkl = os.path.join(save_trail_dir, f"speck{speck}_{nr}round_{diff_l}_{method_name}.pkl")
    with open(save_pkl, "wb") as f:
        pickle.dump((current_paths, tree), f)

    print(f"\n📁 结果已保存至:\n  {save_txt}\n  {save_pkl}\n  {jsonl_path}\n  {pkl_path}")
    return current_paths, tree, payload

def find_initial_paths_parallel_logged_two_stage_compat(*args, **kwargs):
    """兼容旧调用方式：只返回 initial_paths，不改你原来的主流程。"""
    initial_paths, payload = find_initial_paths_parallel_logged_two_stage(*args, **kwargs)
    return initial_paths



import os
import copy
import pickle
from collections import defaultdict

def make_tree():
    return defaultdict(list)

def safe_remove(path: str):
    if os.path.exists(path):
        os.remove(path)
        print(f"[clean] removed: {path}")

def clean_method_outputs(metrics_dir: str, speck_bits: int, method_name: str, nr_total: int, diff_l: int):
    """
    删除当前 diff 在该 method 下的旧输出，避免 jsonl 追加导致重复记录
    """
    method_dir = os.path.join(metrics_dir, f"speck{speck_bits}", method_name, f"r{nr_total}")
    os.makedirs(method_dir, exist_ok=True)

    candidates = [
        os.path.join(method_dir, f"diff_{diff_l}.pkl"),
        os.path.join(method_dir, f"diff_{diff_l}.json"),
        os.path.join(method_dir, f"diff_{diff_l}_round_metrics.jsonl"),
        os.path.join(method_dir, f"diff_{diff_l}_initial_population_two_stage.json"),
        os.path.join(method_dir, f"diff_{diff_l}_initial_population_two_stage.pkl"),
    ]
    for p in candidates:
        safe_remove(p)

if __name__ == "__main__":
    diff_l = 0x8000000
    diff_r = 0x0
    nr_seed = 8
    nr_total = 14
    speck_bits = 2 * word_size
    metrics_dir = project_path("saved_metrics")

    # # ============================================================
    # # 0. 先清洗旧文件，避免 round_metrics.jsonl 重复追加
    # # ============================================================
    # for method in ["egbs", "egbs_mp", "egbs_sp"]:
    #     clean_method_outputs(metrics_dir, speck_bits, method, nr_total, diff_l)

    # # ============================================================
    # # 1. 跑 pure EGBS（使用独立 tree）
    # # ============================================================
    # tree_egbs = make_tree()
    # tree_egbs[(diff_l, diff_r)] = []

    # final_paths_egbs, tree_egbs, payload_egbs = paths_search_add_parallel_logged(
    #     diff_l=diff_l,
    #     diff_r=diff_r,
    #     tree=tree_egbs,
    #     nr=nr_total,
    #     top1=10,
    #     top2=10,
    #     delta=2,
    #     t1=12,
    #     t2=24,
    #     expand_threshold=2**(-16),
    #     speck=speck_bits,
    #     population_size=2000,
    #     initial_paths=None,
    #     n_jobs=32,
    #     method_name="egbs",
    #     metrics_dir=metrics_dir,
    # )

    # # ============================================================
    # # 2. 读取 8 轮种子数据（作为 MP / SP 的共同起点）
    # # ============================================================
    # saved_tree_path = f"./saved_trail/speck{speck_bits}_{nr_seed}round_{diff_l}_evolution_data.pkl"
    # with open(saved_tree_path, "rb") as f:
    #     current_paths_seed, base_tree_seed = pickle.load(f)

    # diff_nodes, counts = count_difference(current_paths_seed, 4)
    # sorted_children = [diff_nodes[0][0], diff_nodes[1][0]]

    # print("=== seed children ===")
    # print([hex(x) for x in sorted_children])

    # # ============================================================
    # # 3. 跑 EGBS + MP（使用 base_tree_seed 的独立拷贝）
    # # ============================================================
    # tree_mp_init = copy.deepcopy(base_tree_seed)
    # initial_paths_mp, init_meta_mp = find_initial_paths_parallel_logged_two_stage(
    #     first_path=current_paths_seed[0],
    #     start_layer=4,
    #     sorted_children=sorted_children.copy(),
    #     tree=tree_mp_init,
    #     top1=20,
    #     t1=14,
    #     t2=35,
    #     expand_threshold=2**(-24),
    #     top_n=12,
    #     vary_times=2,
    #     n_jobs=32,
    #     require_multi_parent=True,
    #     nr_total=nr_total,
    #     method_name="egbs_mp",
    #     metrics_dir=metrics_dir,
    # )

    # tree_mp_search = copy.deepcopy(base_tree_seed)
    # final_paths_mp, tree_mp_search, payload_mp = paths_search_add_parallel_logged(
    #     diff_l=diff_l,
    #     diff_r=0,
    #     tree=tree_mp_search,
    #     nr=nr_total,
    #     top1=10,
    #     top2=10,
    #     delta=2,
    #     t1=12,
    #     t2=24,
    #     expand_threshold=2**(-16),
    #     population_size=2000,
    #     initial_paths=initial_paths_mp,
    #     n_jobs=32,
    #     speck=speck_bits,
    #     method_name="egbs_mp",
    #     metrics_dir=metrics_dir,
    #     initial_population_payload=init_meta_mp,
    # )

    # # ============================================================
    # # 4. 跑 EGBS + SP（使用 base_tree_seed 的独立拷贝）
    # # ============================================================
    # tree_sp_init = copy.deepcopy(base_tree_seed)
    # initial_paths_sp, init_meta_sp = find_initial_paths_parallel_logged_two_stage(
    #     first_path=current_paths_seed[0],
    #     start_layer=4,
    #     sorted_children=sorted_children.copy(),
    #     tree=tree_sp_init,
    #     top1=20,
    #     t1=14,
    #     t2=35,
    #     expand_threshold=2**(-24),
    #     top_n=12,
    #     vary_times=2,
    #     n_jobs=32,
    #     require_multi_parent=False,
    #     nr_total=nr_total,
    #     method_name="egbs_sp",
    #     metrics_dir=metrics_dir,
    # )

    # tree_sp_search = copy.deepcopy(base_tree_seed)
    # final_paths_sp, tree_sp_search, payload_sp = paths_search_add_parallel_logged(
    #     diff_l=diff_l,
    #     diff_r=0,
    #     tree=tree_sp_search,
    #     nr=nr_total,
    #     top1=10,
    #     top2=10,
    #     delta=2,
    #     t1=12,
    #     t2=24,
    #     expand_threshold=2**(-16),
    #     population_size=2000,
    #     initial_paths=initial_paths_sp,
    #     n_jobs=32,
    #     speck=speck_bits,
    #     method_name="egbs_sp",
    #     metrics_dir=metrics_dir,
    #     initial_population_payload=init_meta_sp,
    # )

    # ============================================================
    # 5. 读取清洗后生成的 payload，做 bridge-state 对比
    # ============================================================
    egbs_pkl = os.path.join(metrics_dir, f"speck{speck_bits}", "egbs",    f"r{nr_total}", f"diff_{diff_l}_round_metrics.pkl")
    mp_pkl   = os.path.join(metrics_dir, f"speck{speck_bits}", "egbs_mp", f"r{nr_total}", f"diff_{diff_l}_round_metrics.pkl")
    sp_pkl   = os.path.join(metrics_dir, f"speck{speck_bits}", "egbs_sp", f"r{nr_total}", f"diff_{diff_l}_round_metrics.pkl")

    with open(egbs_pkl, "rb") as f:
        egbs_payload = pickle.load(f)
    with open(mp_pkl, "rb") as f:
        mp_payload = pickle.load(f)
    with open(sp_pkl, "rb") as f:
        sp_payload = pickle.load(f)

    print("\n=== Bridge states: MP vs EGBS ===")
    rows_mp = compare_bridge_states_against_baseline(
        egbs_payload["round_logs"],
        mp_payload["round_logs"]
    )
    print(rows_mp)

    print("\n=== Bridge states: SP vs EGBS ===")
    rows_sp = compare_bridge_states_against_baseline(
        egbs_payload["round_logs"],
        sp_payload["round_logs"]
    )
    print(rows_sp)

    print("\nDone.")










