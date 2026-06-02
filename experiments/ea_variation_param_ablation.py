import sys
sys.stdout.reconfigure(line_buffering=True)
import os
from math import *
import itertools
from collections import defaultdict, OrderedDict, Counter
import copy
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import statistics
from math import log2
import ea_initial_population as init
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

word_size = 64
MASK_VAL = 2 ** word_size - 1

# 重要：ea_initial_population.py 在并行 worker 中会直接读取它自己模块内的全局 word_size / MASK_VAL
# 如果这里只设置了本文件的 word_size，而没有同步给 init 模块，worker 里会报 NameError: word_size is not defined
init.word_size = word_size
init.MASK_VAL = MASK_VAL


def _sync_init_globals():
    init.word_size = word_size
    init.MASK_VAL = MASK_VAL


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

def paths_search_add(diff_l, diff_r, tree, nr, top1=20, top2=10, delta=2, t1=14, t2=35, expand_threshold=2**(-24), speck=128, population_size=2000, initial_paths=None):
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

def paths_search_add_parallel(diff_l, diff_r, tree, nr, top1=10, top2=10, delta=2, t1=14, t2=30, expand_threshold=2**(-24), speck=128, population_size=2000, initial_paths=None, n_jobs=8):
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

def paths_search_sub(diff_l, diff_r, tree, nr, top1=20, top2=10, delta=2, t1=14, t2=35, expand_threshold=2**(-24), speck=128, population_size=2000, initial_paths=None):
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

def paths_search_sub_parallel(diff_l, diff_r, tree, nr, top1=20, top2=10, delta=2, t1=14, t2=30, expand_threshold=2**(-24), speck=128, population_size=2000, initial_paths=None, n_jobs=8):
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



# ===== original main removed for ablation runner =====


# ==================== Parameter ablation runner for SPECK96 (1<<27, 0) ====================
import csv


def _sanitize_tag(tag: str) -> str:
    return str(tag).replace(" ", "-").replace("/", "-")


def _build_setting_tag(setting_name, delta, top1, population_size, q):
    return _sanitize_tag(f"{setting_name}_d{delta}_top1_{top1}_pop_{population_size}_q_{q}")


def _ensure_dirs_for_speck(speck):
    os.makedirs(project_path("saved_ea_txt", f"speck{speck}"), exist_ok=True)
    os.makedirs(project_path("saved_trail"), exist_ok=True)
    os.makedirs("./saved_initial_paths", exist_ok=True)
    os.makedirs("./param_ablation", exist_ok=True)


def _save_base_stage_outputs(current_paths, tree, diff_l, diff_r, nr, speck, elapsed_sec, save_tag):
    """保存 base 阶段(例如 8 轮 first_best_search)的结果，避免不同参数组互相覆盖。"""
    _ensure_dirs_for_speck(speck)
    tag = _sanitize_tag(save_tag)
    save_path1 = project_path("saved_ea_txt", f"speck{speck}", f"speck{speck}_base_result_{nr}round_{diff_l}_{tag}.txt")
    save_path2 = project_path("saved_trail", f"speck{speck}_{nr}round_{diff_l}_{tag}_base.pkl")

    with open(save_path1, 'w', encoding='utf-8') as f:
        f.write("=== Base Stage Result ===\n")
        f.write(f"Speck版本: {speck}\n")
        f.write(f"起始差分: ({hex(diff_l)}, {hex(diff_r)})\n")
        f.write(f"轮数: {nr}\n")
        f.write(f"保存标签: {tag}\n")
        f.write(f"路径数: {len(current_paths)}\n")
        f.write(f"耗时: {elapsed_sec:.2f} 秒 ({elapsed_sec/60:.2f} 分钟)\n\n")
        for i, path in enumerate(current_paths[:min(len(current_paths), 200)]):
            try:
                score = evaluate_path_add(path, nr)
            except Exception:
                score = float('nan')
            f.write(f"---- 路径 {i+1} (总分: {score:.4f}) ----\n")
            for j in range(len(path) - 1):
                l0, r0 = path[j]
                child = path[j + 1][0]
                node_score = -log2(xdp_add(ror(l0, ALPHA()), r0, child))
                f.write(f"层 {j}: ({hex(l0)}, {hex(r0)}) | 分数: {node_score:.4f}\n")
            if path:
                f.write(f"层 {len(path)-1}: ({hex(path[-1][0])}, {hex(path[-1][1])})\n\n")

    with open(save_path2, 'wb') as f:
        pickle.dump((current_paths, tree), f)

    print(f"\n📁 base 阶段结果已保存至:\n  {save_path1}\n  {save_path2}")
    return save_path1, save_path2


def find_initial_paths_parallel_tagged(first_path, start_layer, sorted_children, tree,
                                       top1=20, t1=14, t2=35,
                                       expand_threshold=2**(-24), top_n=12,
                                       vary_times=2, n_jobs=8,
                                       require_multi_parent=True,
                                       save_tag="default"):
    """带标签保存版本，逻辑与 find_initial_paths_parallel 基本一致。"""
    diff_l, diff_r = first_path[start_layer - 1]
    root_diff_l = first_path[0][0]
    diff_l_new = ror(diff_l, ALPHA())
    candidates = list(sorted_children)
    all_sorted_children = list(sorted_children)

    for i in range(vary_times):
        candidates = expand_population_with_rank(
            candidates,
            diff_l_new,
            diff_r,
            delta=4,
            num_samples=100,
            require_multi_parent=require_multi_parent,
            top_n=top_n**(i+1),
            output_as_list=True,
        )
        all_sorted_children.extend(candidates)

    prefix = first_path[:start_layer]
    tasks = [(cand, diff_r, prefix, t1, t2, top1, expand_threshold) for cand in all_sorted_children]
    initial_paths = []
    local_trees = []

    with Pool(processes=n_jobs) as pool:
        for local_tree, local_paths in tqdm(
            pool.imap_unordered(process_candidate, tasks),
            total=len(tasks),
            desc="生成初始路径进度",
            unit="节点",
            ncols=100,
        ):
            local_trees.append(local_tree)
            initial_paths.extend(local_paths)

    for local_tree in local_trees:
        for k, v in local_tree.items():
            if k not in tree:
                tree[k] = []
            tree[k].extend(v)

    print(f"生成初始路径 {len(initial_paths)} 条，每条长度为 {len(initial_paths[0]) if initial_paths else 0}")

    save_dir = "./saved_initial_paths"
    os.makedirs(save_dir, exist_ok=True)
    tag = _sanitize_tag(save_tag)
    save_path = f"{save_dir}/speck{2*word_size}_{start_layer+2}round_{root_diff_l}_initial_paths_{tag}.pkl"
    txt_path = save_path.replace(".pkl", ".txt")
    with open(save_path, 'wb') as f:
        pickle.dump(initial_paths, f)
    save_paths_to_txt(initial_paths, txt_path)
    print(f"\n📁 初始路径已保存至:\n  {save_path}\n  {txt_path}")
    return initial_paths, save_path, txt_path


def paths_search_add_parallel_tagged(diff_l, diff_r, tree, nr,
                                     top1=10, top2=10, delta=2,
                                     t1=14, t2=30,
                                     expand_threshold=2**(-24),
                                     speck=128, population_size=2000,
                                     initial_paths=None, n_jobs=8,
                                     save_tag="default"):
    """带标签保存版本，逻辑与 paths_search_add_parallel 基本一致。"""
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
    for layer in range(start_layer, nr):
        layer_start = time.time()
        print(f"\n--- 正在处理第 {layer + 1} 层 (当前路径数: {len(current_paths)}) ---")
        next_paths = []
        expanded_nodes_total = 0
        task_args = [(path, tree, top1, top2, delta, t1, t2, expand_threshold) for path in current_paths]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(expand_single_path_add, args) for args in task_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"第 {layer+1} 层处理中", ncols=90):
                expanded, expanded_nodes = future.result()
                next_paths.extend(expanded)
                expanded_nodes_total += expanded_nodes

        elapsed = time.time() - layer_start
        print(f"  ✅ 本层扩展 {expanded_nodes_total} 个新节点，共生成 {len(next_paths)} 条路径，用时 {elapsed:.1f}s")
        current_paths = next_paths

        if layer >= 3 and len(current_paths) > population_size:
            print(f"  正在进行路径选择 (当前路径数: {len(current_paths)})...")
            evaluated_paths = [(evaluate_path_add(path, nr), path) for path in current_paths]
            evaluated_paths.sort(key=lambda x: x[0])
            current_paths = [path for (score, path) in evaluated_paths[:population_size]]
            print(f"  ✅ 选择完成: 保留前{population_size}条最优路径 (最低分: {evaluated_paths[0][0]:.6f})")

        if not current_paths:
            print("⚠️ 无可扩展路径，搜索提前结束。")
            break

    total_elapsed = time.time() - total_start_time
    print(f"\n🎯 搜索完成，总耗时 {total_elapsed/60:.2f} 分钟")

    _ensure_dirs_for_speck(speck)
    tag = _sanitize_tag(save_tag)
    save_path1 = project_path("saved_ea_txt", f"speck{speck}", f"speck{speck}_result_ea_{nr}round_{diff_l}_{tag}.txt")
    save_path2 = project_path("saved_trail", f"speck{speck}_{nr}round_{diff_l}_{tag}.pkl")

    with open(save_path1, 'w', encoding='utf-8') as f:
        f.write("=== 差分路径搜索结果 ===\n")
        f.write(f"Speck版本: {speck}\n")
        f.write(f"总层数: {nr}\n")
        f.write(f"起始层: {start_layer + 1}\n")
        f.write(f"保存标签: {tag}\n")
        f.write(f"最终保留路径数: {len(current_paths)}\n")
        f.write(f"总耗时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)\n\n")
        for i, path in enumerate(current_paths):
            score = evaluate_path_add(path, nr)
            f.write(f"\n---- 路径 {i + 1} (总分: {score:.4f}) ----\n")
            for j in range(len(path) - 1):
                l0, r0 = path[j]
                child = path[j + 1][0]
                node_score = -log2(xdp_add(ror(l0, ALPHA()), r0, child))
                f.write(f"层 {j}: ({hex(l0)}, {hex(r0)}) | 分数: {node_score:.4f}\n")
            f.write(f"层 {len(path) - 1}: ({hex(path[-1][0])}, {hex(path[-1][1])})\n")

    with open(save_path2, 'wb') as f:
        pickle.dump((current_paths, tree), f)

    print(f"\n📁 结果已保存至:\n  {save_path1}\n  {save_path2}")
    return current_paths, tree, total_elapsed, save_path1, save_path2


def _get_best_result_from_paths(paths, nr):
    if not paths:
        return None, None
    scored = [(evaluate_path_add(path, nr), path) for path in paths]
    scored.sort(key=lambda x: x[0])
    return scored[0]


def _get_cached_tree_path(diff_l, base_nr, save_tag):
    return project_path(
        "saved_trail",
        f"speck{2*word_size}_{base_nr}round_{diff_l}_{save_tag}_base.pkl",
    )


def _get_cached_initial_paths_path(diff_l, save_tag, initial_round=6):
    return project_path(
        "saved_initial_paths",
        f"speck{2*word_size}_{initial_round}round_{diff_l}_initial_paths_{save_tag}.pkl",
    )

def run_one_ablation(setting_name,
                     diff_l=0x8000000, diff_r=0x0,
                     base_nr=8, search_nr=11,
                     delta=2, top1=10, population_size=2000, q=2,
                     top2=10, t1=12, t2=24, expand_threshold=2**(-16),
                     n_jobs=32):
    """
    严格可比版：
    1) 不重跑 base 搜索
    2) 不重算 initial_paths
    3) 固定加载该 setting 已保存好的 tree 和 initial_paths
    4) 只计“加载 + 最终搜索”时间，口径与 run_one_experiment 基本一致
    """
    speck = 2 * word_size
    save_tag = _build_setting_tag(setting_name, delta, top1, population_size, q)

    cached_tree_pkl = _get_cached_tree_path(diff_l, base_nr, save_tag)
    cached_initial_paths_pkl = _get_cached_initial_paths_path(diff_l, save_tag, initial_round=6)

    if not os.path.exists(cached_tree_pkl):
        raise FileNotFoundError(
            f"[ERROR] 未找到缓存 tree 文件：\n{cached_tree_pkl}\n"
            f"请先确认 base 阶段是否已按该 setting 保存。"
        )
    if not os.path.exists(cached_initial_paths_pkl):
        raise FileNotFoundError(
            f"[ERROR] 未找到缓存 initial_paths 文件：\n{cached_initial_paths_pkl}\n"
            f"请先确认 initial_paths 是否已按该 setting 保存。"
        )

    print("\n" + "=" * 100)
    print(f"开始运行参数组: {setting_name}")
    print(f"save_tag = {save_tag}")
    print(f"delta = {delta}, top1 = {top1}, pop = {population_size}, q = {q}")
    print(f"cached_tree_pkl         = {cached_tree_pkl}")
    print(f"cached_initial_paths_pkl= {cached_initial_paths_pkl}")
    print("=" * 100)

    # 和你原来的 run_one_experiment 一样，从加载缓存开始计时
    total_t1 = time.time()

    with open(cached_tree_pkl, "rb") as f:
        current_paths, tree = pickle.load(f)

    with open(cached_initial_paths_pkl, "rb") as f:
        initial_paths = pickle.load(f)

    # 只跑最终搜索阶段
    final_paths, final_tree, search_elapsed_inner, result_txt, result_pkl = paths_search_add_parallel_tagged(
        diff_l=diff_l,
        diff_r=diff_r,
        tree=tree,
        nr=search_nr,
        top1=top1,
        top2=top2,
        delta=delta,
        t1=t1,
        t2=t2,
        expand_threshold=expand_threshold,
        speck=speck,
        population_size=population_size,
        initial_paths=initial_paths,
        n_jobs=n_jobs,
        save_tag=save_tag,
    )

    total_t2 = time.time()
    elapsed_sec = total_t2 - total_t1   # 加载 + 最终搜索，和你原先口径一致

    best_weight, best_path = _get_best_result_from_paths(final_paths, search_nr)

    result = {
        "setting": setting_name,
        "tag": save_tag,
        "delta": delta,
        "top1": top1,
        "population_size": population_size,
        "q": q,
        "best_weight": best_weight,
        "best_path": best_path,
        "num_final_paths": len(final_paths),

        # 严格可比版：base 不重跑，所以置 0
        "base_time_sec": 0.0,

        # 这里直接用“加载缓存 + 最终搜索”的总耗时，和 run_one_experiment 对齐
        "search_time_sec": elapsed_sec,
        "total_time_sec": elapsed_sec,

        # 额外保留一下内层 search 函数自己统计的纯搜索时间，方便你核对
        "search_time_inner_sec": search_elapsed_inner,

        "cached_tree_pkl": cached_tree_pkl,
        "cached_initial_paths_pkl": cached_initial_paths_pkl,

        "result_txt": result_txt,
        "result_pkl": result_pkl,
    }

    print("-" * 100)
    print(f"setting              : {setting_name}")
    print(f"best_weight          : {best_weight}")
    print(f"num_final_paths      : {len(final_paths)}")
    print(f"search_time_sec      : {elapsed_sec:.3f}")
    print(f"search_time_inner_sec: {search_elapsed_inner:.3f}")
    print(f"result_txt           : {result_txt}")
    print(f"result_pkl           : {result_pkl}")
    print("-" * 100)

    return result


def run_param_ablation_suite(diff_l=0x8000000, diff_r=0x0,
                             base_nr=8, search_nr=11,
                             n_jobs=32):
    """
    一次跑完 9 组参数消融（严格可比版）：
    - 不重跑 base
    - 不重算 initial_paths
    - 每组固定加载各自已经保存好的 tree 和 initial_paths
    """
    settings = [
        ("Full",        {"delta": 2, "top1": 10, "population_size": 2000, "q": 2}),
        ("delta-small", {"delta": 1, "top1": 10, "population_size": 2000, "q": 2}),
        ("delta-large", {"delta": 3, "top1": 10, "population_size": 2000, "q": 2}),
        ("top1-small",  {"delta": 2, "top1": 5,  "population_size": 2000, "q": 2}),
        ("top1-large",  {"delta": 2, "top1": 20, "population_size": 2000, "q": 2}),
        ("pop-small",   {"delta": 2, "top1": 10, "population_size": 1000, "q": 2}),
        ("pop-large",   {"delta": 2, "top1": 10, "population_size": 4000, "q": 2}),
        ("q-1",         {"delta": 2, "top1": 10, "population_size": 2000, "q": 1}),
        ("q-3",         {"delta": 2, "top1": 10, "population_size": 2000, "q": 3}),
    ]

    save_dir = "./param_ablation"
    os.makedirs(save_dir, exist_ok=True)

    summary_txt = os.path.join(save_dir, f"speck{2*word_size}_r{search_nr}_k27_param_ablation_summary.txt")
    summary_csv = os.path.join(save_dir, f"speck{2*word_size}_r{search_nr}_k27_param_ablation_summary.csv")
    summary_pkl = os.path.join(save_dir, f"speck{2*word_size}_r{search_nr}_k27_param_ablation_summary.pkl")

    all_results = []

    for setting_name, cfg in settings:
        res = run_one_ablation(
            setting_name=setting_name,
            diff_l=diff_l,
            diff_r=diff_r,
            base_nr=base_nr,
            search_nr=search_nr,
            n_jobs=n_jobs,
            **cfg,
        )
        all_results.append(res)

        # 每做完一组就刷新保存
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write("SPECK96 参数消融实验汇总（固定加载 tree + initial_paths 的严格可比版）\n")
            f.write(f"input difference = ({hex(diff_l)}, {hex(diff_r)})\n")
            f.write(f"base_nr = {base_nr}, search_nr = {search_nr}\n\n")

            for item in all_results:
                f.write("=" * 100 + "\n")
                f.write(f"setting                  : {item['setting']}\n")
                f.write(f"tag                      : {item['tag']}\n")
                f.write(f"delta / top1             : {item['delta']} / {item['top1']}\n")
                f.write(f"pop / q                  : {item['population_size']} / {item['q']}\n")
                f.write(f"best_weight              : {item['best_weight']}\n")
                f.write(f"base_time_sec            : {item['base_time_sec']:.3f}\n")
                f.write(f"search_time_sec          : {item['search_time_sec']:.3f}\n")
                f.write(f"search_time_inner_sec    : {item['search_time_inner_sec']:.3f}\n")
                f.write(f"total_time_sec           : {item['total_time_sec']:.3f}\n")
                f.write(f"num_final_paths          : {item['num_final_paths']}\n")
                f.write(f"cached_tree_pkl          : {item['cached_tree_pkl']}\n")
                f.write(f"cached_initial_paths_pkl : {item['cached_initial_paths_pkl']}\n")
                f.write(f"result_txt               : {item['result_txt']}\n")
                f.write(f"result_pkl               : {item['result_pkl']}\n")
                f.write(f"best_path                : {item['best_path']}\n\n")

        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'setting', 'tag', 'delta', 'top1', 'population_size', 'q',
                'best_weight', 'base_time_sec', 'search_time_sec', 'search_time_inner_sec',
                'total_time_sec', 'num_final_paths',
                'cached_tree_pkl', 'cached_initial_paths_pkl',
                'result_txt', 'result_pkl'
            ])
            for item in all_results:
                writer.writerow([
                    item['setting'], item['tag'], item['delta'], item['top1'],
                    item['population_size'], item['q'], item['best_weight'],
                    item['base_time_sec'], item['search_time_sec'], item['search_time_inner_sec'],
                    item['total_time_sec'], item['num_final_paths'],
                    item['cached_tree_pkl'], item['cached_initial_paths_pkl'],
                    item['result_txt'], item['result_pkl']
                ])

        with open(summary_pkl, 'wb') as f:
            pickle.dump(all_results, f)

    print("\n" + "#" * 100)
    print("全部 9 组参数消融实验完成！")
    print(f"汇总 TXT: {summary_txt}")
    print(f"汇总 CSV: {summary_csv}")
    print(f"汇总 PKL: {summary_pkl}")
    print("#" * 100)

    return all_results

def _build_speck128_tag(setting_name, delta, top1, population_size):
    return f"{setting_name}_d{delta}_top1_{top1}_pop_{population_size}"

def _sync_speck128_globals():
    global word_size, MASK_VAL
    word_size = 64
    MASK_VAL = 2 ** word_size - 1

    # 如果你这里也依赖了外部模块里的全局 word_size / MASK_VAL，就同步一下
    try:
        import ea_initial_population as init
        init.word_size = word_size
        init.MASK_VAL = MASK_VAL
    except Exception:
        pass

def _save_speck128_result(current_paths, tree, diff_l, diff_r, nr, speck, save_tag, elapsed_sec):
    save_dir_txt = project_path("saved_ea_txt", f"speck{speck}")
    save_dir_pkl = project_path("saved_trail")
    os.makedirs(save_dir_txt, exist_ok=True)
    os.makedirs(save_dir_pkl, exist_ok=True)

    txt_path = f"{save_dir_txt}/speck{speck}_result_egbs_{nr}round_{diff_l}_{save_tag}.txt"
    pkl_path = f"{save_dir_pkl}/speck{speck}_{nr}round_{diff_l}_{save_tag}.pkl"

    scored_paths = [(evaluate_path_sub(path, nr), path) for path in current_paths]
    scored_paths.sort(key=lambda x: x[0])
    best_score, best_path = scored_paths[0]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== SPECK128 EGBS Parameter Ablation Result ===\n")
        f.write(f"Speck version: {speck}\n")
        f.write(f"Input diff   : ({hex(diff_l)}, {hex(diff_r)})\n")
        f.write(f"Rounds       : {nr}\n")
        f.write(f"Tag          : {save_tag}\n")
        f.write(f"Best weight  : {best_score}\n")
        f.write(f"Time (sec)   : {elapsed_sec:.3f}\n")
        f.write(f"Time (min)   : {elapsed_sec/60:.2f}\n")
        f.write(f"Num paths    : {len(current_paths)}\n\n")
        f.write(f"Best path    : {[(hex(a), hex(b)) for (a, b) in best_path]}\n")

    with open(pkl_path, "wb") as f:
        pickle.dump((current_paths, tree), f)

    return best_score, best_path, txt_path, pkl_path

def run_one_speck128_ablation(setting_name,
                              delta=2,
                              top1=10,
                              population_size=2000,
                              diff_l=0x80,
                              diff_r=0x0,
                              nr=16,
                              top2=10,
                              t1=12,
                              t2=24,
                              expand_threshold=2**(-16),
                              n_jobs=32):
    _sync_speck128_globals()

    speck = 2 * word_size
    save_tag = _build_speck128_tag(setting_name, delta, top1, population_size)

    print("\n" + "=" * 100)
    print(f"开始运行参数组: {setting_name}")
    print(f"save_tag = {save_tag}")
    print(f"delta = {delta}, top1 = {top1}, pop = {population_size}")
    print("=" * 100)

    tree = defaultdict(list)
    tree[(diff_l, diff_r)] = []

    t0 = time.time()
    ret = init.first_best_search_sub_parallel(
        diff_l=diff_l,
        diff_r=diff_r,
        tree=tree,
        nr=nr,
        top1=top1,
        top2=top2,
        delta=delta,
        t1=t1,
        t2=t2,
        expand_threshold=expand_threshold,
        speck=speck,
        population_size=population_size,
        n_jobs=n_jobs
    )
    elapsed = time.time() - t0

    if isinstance(ret, tuple) and len(ret) == 2:
        current_paths, tree = ret
    else:
        current_paths = ret

    best_score, best_path, txt_path, pkl_path = _save_speck128_result(
        current_paths=current_paths,
        tree=tree,
        diff_l=diff_l,
        diff_r=diff_r,
        nr=nr,
        speck=speck,
        save_tag=save_tag,
        elapsed_sec=elapsed
    )

    result = {
        "setting": setting_name,
        "tag": save_tag,
        "block_size": speck,
        "diff": (hex(diff_l), hex(diff_r)),
        "delta": delta,
        "top1": top1,
        "population_size": population_size,
        "q": "--",
        "best_weight": best_score,
        "time_sec": elapsed,
        "time_min": elapsed / 60,
        "num_final_paths": len(current_paths),
        "result_txt": txt_path,
        "result_pkl": pkl_path,
        "best_path": [(hex(a), hex(b)) for (a, b) in best_path],
    }

    print(f"\n✅ 参数组 {setting_name} 完成: best_weight = {best_score}, time = {elapsed/60:.2f} min")
    return result

def run_speck128_ablation_suite():
    settings = [
        # ("Full",        {"delta": 2, "top1": 10, "population_size": 2000}),
        # ("delta-small", {"delta": 1, "top1": 10, "population_size": 2000}),
        # ("delta-large", {"delta": 3, "top1": 10, "population_size": 2000}),
        # ("top1-small",  {"delta": 2, "top1": 5,  "population_size": 2000}),
        # ("top1-large",  {"delta": 2, "top1": 20, "population_size": 2000}),
        ("pop-small",   {"delta": 2, "top1": 10, "population_size": 500}),
        # ("pop-large",   {"delta": 2, "top1": 10, "population_size": 4000}),
    ]

    save_dir = "./param_ablation"
    os.makedirs(save_dir, exist_ok=True)

    summary_txt = os.path.join(save_dir, "speck128_r16_param_ablation_summary.txt")
    summary_csv = os.path.join(save_dir, "speck128_r16_param_ablation_summary.csv")
    summary_pkl = os.path.join(save_dir, "speck128_r16_param_ablation_summary.pkl")

    all_results = []
    for setting_name, cfg in settings:
        res = run_one_speck128_ablation(setting_name=setting_name, **cfg)
        all_results.append(res)

        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write("SPECK128 参数消融实验汇总\n")
            f.write("input difference = (0x80, 0x0)\n")
            f.write("rounds = 16\n\n")
            for item in all_results:
                f.write("=" * 100 + "\n")
                f.write(f"setting         : {item['setting']}\n")
                f.write(f"tag             : {item['tag']}\n")
                f.write(f"block size      : {item['block_size']}\n")
                f.write(f"diff            : {item['diff']}\n")
                f.write(f"delta / top1    : {item['delta']} / {item['top1']}\n")
                f.write(f"pop / q         : {item['population_size']} / {item['q']}\n")
                f.write(f"best_weight     : {item['best_weight']}\n")
                f.write(f"time_sec        : {item['time_sec']:.3f}\n")
                f.write(f"time_min        : {item['time_min']:.2f}\n")
                f.write(f"num_final_paths : {item['num_final_paths']}\n")
                f.write(f"result_txt      : {item['result_txt']}\n")
                f.write(f"result_pkl      : {item['result_pkl']}\n")
                f.write(f"best_path       : {item['best_path']}\n\n")

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "setting", "block_size", "diff_l", "diff_r",
                "delta", "top1", "population_size", "q",
                "best_weight", "time_sec", "time_min",
                "num_final_paths", "result_txt", "result_pkl"
            ])
            for item in all_results:
                writer.writerow([
                    item["setting"], item["block_size"], item["diff"][0], item["diff"][1],
                    item["delta"], item["top1"], item["population_size"], item["q"],
                    item["best_weight"], item["time_sec"], item["time_min"],
                    item["num_final_paths"], item["result_txt"], item["result_pkl"]
                ])

        with open(summary_pkl, "wb") as f:
            pickle.dump(all_results, f)

    print("\n" + "#" * 100)
    print("SPECK128 全部 7 组参数消融完成！")
    print(f"汇总 TXT: {summary_txt}")
    print(f"汇总 CSV: {summary_csv}")
    print(f"汇总 PKL: {summary_pkl}")
    print("#" * 100)

    return all_results



# 你要直接跑 8 组参数消融时，把 __main__ 替换成这个即可：
if __name__ == "__main__":
    run_speck128_ablation_suite()
    # run_param_ablation_suite(
    #     diff_l=0x8000000,
    #     diff_r=0x0,
    #     base_nr=8,
    #     search_nr=14,
    #     n_jobs=32,
    # )
    
    

