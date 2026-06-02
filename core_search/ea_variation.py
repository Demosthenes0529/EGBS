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



if __name__ == "__main__":
    # RUN_TIMES = 10
    # output_file = "./time_experiment_results.txt"
    # j_list = [12,14,16,18,20,21,23,25,29,31,33,36,38]

    # all_results = []  # 用于保存每个 j 的结果

    # for j in j_list:
    #     diff_l = 1 << j
    #     run_seconds = []

    #     for i in range(1, RUN_TIMES + 1):
    #         print(f"\n================ 第 {i}/{RUN_TIMES} 次运行 (j={j}) ================")
    #         elapsed_sec = run_one_experiment(diff_l=diff_l, diff_r=0x0)
    #         run_seconds.append(elapsed_sec)

    #     avg_sec = statistics.mean(run_seconds)
    #     std_sec = statistics.stdev(run_seconds) if len(run_seconds) > 1 else 0.0

    #     # 保存结果到列表
    #     all_results.append({
    #         "j": j,
    #         "run_times": RUN_TIMES,
    #         "run_seconds": run_seconds,
    #         "avg_sec": avg_sec,
    #         "std_sec": std_sec
    #     })

    #     # 打印到终端
    #     print("\n================ 统计结果 ================")
    #     print(f"j={j}, 运行次数: {RUN_TIMES}")
    #     print("每次耗时(秒):", [round(x, 3) for x in run_seconds])
    #     print(f"平均耗时: {avg_sec:.3f} 秒 ({avg_sec/60:.3f} 分钟)")
    #     print(f"标准差: {std_sec:.3f} 秒 ({std_sec/60:.3f} 分钟)")

    # # 一次性写入文件
    # with open(output_file, "w", encoding="utf-8") as f:
    #     f.write("SPECK 实验运行结果\n\n")
    #     for res in all_results:
    #         f.write(f"================ j={res['j']} =================\n")
    #         f.write(f"运行次数: {res['run_times']}\n")
    #         f.write(f"每次耗时(秒): {[round(x,3) for x in res['run_seconds']]}\n")
    #         f.write(f"平均耗时: {res['avg_sec']:.3f} 秒 ({res['avg_sec']/60:.3f} 分钟)\n")
    #         f.write(f"标准差: {res['std_sec']:.3f} 秒 ({res['std_sec']/60:.3f} 分钟)\n\n")

    diff_l = 0x8000000
    diff_r = 0x0
    print(f"diff_l = {diff_l}")
    nr = 8
    def make_tree():
        return defaultdict(list)  
    tree = make_tree()
    tree[(diff_l, diff_r)] = []     
    current_paths = init.first_best_search_sub_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=nr, top1=20, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size=2000, n_jobs=32)
    saved_tree_path1 = project_path("saved_trail", f"speck{2*word_size}_{nr}round_{diff_l}_evolution_data.pkl")
    with open(saved_tree_path1, "rb") as f:
        current_paths, tree = pickle.load(f)
    diff_nodes, counts = count_difference(current_paths, 4)  
    # print(diff_nodes)
    sorted_children = [diff_nodes[0][0], diff_nodes[1][0]]
    initial_paths = find_initial_paths_parallel(current_paths[0], start_layer=4, sorted_children=sorted_children, tree=tree, top_n=12, n_jobs=32)
    # print(initial_paths[0])
    paths_search_add_parallel(diff_l, diff_r, tree, nr=11, top1=10, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-16), population_size=2000, speck=2*word_size, initial_paths=initial_paths, n_jobs=32)






















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







    # # diff_l = 0x0808080800fe
    # # diff_r = 0x4a08480800ee
    # # diff_l_new = ror(4718600, ALPHA())
    # diff_l = 0x41010103
    # diff_r = 0x49410901
    # # diff_l_new = ror(8800354961416, ALPHA())
    # print(f"变异前节点: (L={hex(diff_l)}, R={hex(diff_r)})")
    # sorted_children, sorted_scores = generate_possible_children_add(diff_l, diff_r, delta=3, t1=14)
    # # print(sorted_children[:10])
    # print(hex(sorted_children[0]), hex(sorted_children[1]), hex(sorted_children[2]), hex(sorted_children[3]), hex(sorted_children[4]))
    # alpha = ror(diff_l, ALPHA())
    # beta = diff_r
    # gamma = sorted_children[3]
    # # target_gamma = 0x400040000772
    # target_gamma = 0x49800802
    # print(target_gamma)
    # alpha_bits = hex_2_bin(alpha)
    # beta_bits = hex_2_bin(beta)
    # gamma_bits = hex_2_bin(gamma)
    # target_bits = hex_2_bin(target_gamma)
    # print(gamma_bits)
    # print(target_bits)
    # diff_positions = [i for i in range(word_size) if gamma_bits[i] != target_bits[i]]
    # ok_positions = [i for i in diff_positions if alpha_bits[i] == beta_bits[i] == gamma_bits[i]]
    # bad_positions = [i for i in diff_positions if not (alpha_bits[i] == beta_bits[i] == gamma_bits[i])]
    # print(f"gamma={hex(gamma)}, target={hex(target_gamma)}")
    # print(f"汉明距离={len(diff_positions)}")
    # print(f"差异位(高位索引): {diff_positions}")
    # print(f"满足 alpha==beta==gamma 的差异位: {ok_positions}")
    # print(f"不满足 alpha==beta==gamma 的差异位: {bad_positions}")
    # if bad_positions:
    #     print("这些位在当前 vary_difference 规则下不可直接翻转。")
    # # 按新版 vary_difference（include_free_bits=True）检查可变异位覆盖性
    # eq_positions = [i for i in range(word_size - 1) if alpha_bits[i] == beta_bits[i] == gamma_bits[i]]
    # free_positions = [i for i in range(word_size - 1) if alpha_bits[i] != beta_bits[i]]
    # mutable_positions = sorted(set(eq_positions) | set(free_positions))
    # not_mutable_diff_positions = [i for i in diff_positions if i not in mutable_positions]
    # print(f"include_free_bits=True 时可变异位数量: {len(mutable_positions)}")
    # print(f"目标差异位是否全部在可变异位上: {len(not_mutable_diff_positions) == 0}")
    # if not_mutable_diff_positions:
    #     print(f"仍不可变异的目标差异位: {not_mutable_diff_positions}")
    # # 复现 vary_difference 的 positionsLists 生成逻辑，检查“连续变动规则”是否可覆盖目标差异位
    # positionsLists = []
    # for i in range(6 + 1):
    #     itera = itertools.combinations(mutable_positions, i)
    #     for comb in itera:
    #         positionsLists.append(list(comb))
    #         if i > 0 and comb[0] > i:
    #             positionsLists.append(list(range(comb[0] - i + 1, comb[0] + 1)))
    # target_set = set(diff_positions)
    # exact_match = any(set(pos_combo) == target_set for pos_combo in positionsLists)
    # exact_match_continuous = any(
    #     (set(pos_combo) == target_set) and any(p not in eq_positions for p in pos_combo)
    #     for pos_combo in positionsLists
    # )
    # contain_bad_by_continuous = [pos_combo for pos_combo in positionsLists if all(p in pos_combo for p in bad_positions)]
    # print(f"positionsLists 中是否存在与目标差异位完全一致的组合: {exact_match}")
    # print(f"其中是否可由连续变动规则命中: {exact_match_continuous}")
    # print(f"可同时覆盖 bad_positions={bad_positions} 的组合数量: {len(contain_bad_by_continuous)}")
    # # print(hex_2_bin(sorted_children[0]))
    # # print(hex_2_bin(0x772400040))
    # # print(hex_2_bin(49161437248))
    # candidates, sorted_scores = vary_difference(ror(diff_l, ALPHA()), diff_r, sorted_children[1], delta=2, num_samples=100)
    # print(hex(candidates[0]), hex(candidates[1]), hex(candidates[2]), hex(candidates[3]), hex(candidates[4]))
    # print(hex(candidates[5]), hex(candidates[6]), hex(candidates[7]), hex(candidates[8]), hex(candidates[9]))
    # print(candidates)
    # # print(0x8000e080808)
    # # print(0x8001e080808)
    # # print(0x8003e080808)
    # # print(0x800fe080808)
    # sorted_children = [0x48000802, 0x4a000802]
    # # sorted_children = [0x180006080808, 0x8000e080808, 0x80006081808, 0x80006080818, 0x8001e080808, 0x18000e080808, 0x80006088808, 0x180006080818, 0x180006081808, 0x8000e081808, 0x80006089808]
    # candidates = expand_population_with_rank(sorted_children[:10], ror(diff_l, ALPHA()), diff_r, delta=3, num_samples=200, require_multi_parent=True, top_n=200, output_as_list=True)
    # print(candidates)
    # print(hex(candidates[0]), hex(candidates[1]), hex(candidates[2]), hex(candidates[3]), hex(candidates[4]))
    # print(hex(candidates[5]), hex(candidates[6]), hex(candidates[7]), hex(candidates[8]), hex(candidates[9]))
    # node_info, origin_map = expand_population_with_rank(candidates, diff_l_new, diff_r, delta=4, num_samples=100, require_multi_parent=True, top_n=144)
    # print(f"{'节点':<10} {'分数':<8} {'来源(父节点,排名,分数)'}")
    # print("-" * 60)
    # for node, info in node_info.items():
    #     parents = ", ".join([f"({hex(p)},{r})" for (p, r) in info["parents"]]) if info["parents"] else "-"
    #     score = f"{info['score']:.2f}" if info["score"] is not None else "-"
    #     print(f"{hex(node):<10} {score:<8} {parents}")



















