import sys
sys.stdout.reconfigure(line_buffering=True)
from math import *
import itertools
from collections import defaultdict
import copy
import time
import pickle
import numpy as np
from math import log2
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
import os

def _base_dir():
    # Directory containing this script; works on Windows/Linux
    return os.path.dirname(os.path.abspath(__file__))


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _out_path(*parts):
    return os.path.join(_base_dir(), *parts)


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
                if (index == word_size - 2 and graph[index].lsb == j) or \
                    (index <= word_size - 3 and (graph[index + 2].successors[0][j] or graph[index + 2].successors[1][j])):
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
    return graphs


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

def evaluate_grandchild_score_add(args):
    diff_l, diff_r, child = args
    score1 = -log2(xdp_add(ror(diff_l, ALPHA()), diff_r, child))
    l_new = child
    r_new = child ^ rol(diff_r, BETA())
    score2 = construct_best_gamma_add(l_new, r_new)
    return (child, score1 + score2)

def generate_possible_children_add(diff_l, diff_r, delta, t1):
    graphs = delta_optimal(ror(diff_l, ALPHA()), diff_r, delta)  
    all_gamma = output_all_possible_gamma(graphs, t1)
    possible_children = convert_list_gamma_to_int(all_gamma)
    scored_children = []
    for child in possible_children:
        scored_children.append(evaluate_grandchild_score_add((diff_l, diff_r, child)))
    scored_children.sort(key=lambda x: x[1])
    sorted_children = [child for child, _ in scored_children]
    sorted_scores = [score for _, score in scored_children]
    return sorted_children, sorted_scores

def evaluate_grandchild_score_sub(args):
    diff_l, diff_r_new, child = args
    score1 = -log2(xdp_add(diff_l, diff_r_new, child))
    l_new = rol(child, ALPHA())    
    r_new = ror(l_new ^ diff_r_new, BETA())
    score2 = construct_best_gamma_sub(l_new, r_new)
    return (child, score1 + score2)

def generate_possible_children_sub(diff_l, diff_r, delta, t1):
    diff_r_new = ror(diff_l ^ diff_r, BETA())
    graphs = delta_optimal(diff_l, diff_r_new, delta)  
    all_gamma = output_all_possible_gamma(graphs, t1)
    possible_children = convert_list_gamma_to_int(all_gamma)
    scored_children = []
    for child in possible_children:
        scored_children.append(evaluate_grandchild_score_sub((diff_l, diff_r_new, child)))
    scored_children.sort(key=lambda x: x[1])
    sorted_children = [child for child, _ in scored_children]
    sorted_scores = [score for _, score in scored_children]
    return sorted_children, sorted_scores

# sorted_children, sorted_scores = generate_possible_children_add(4718600, 34095112, delta=2, t1=12)
# print(sorted_children[:30], sorted_scores[:30])

def first_best_search_add(diff_l, diff_r, tree, nr, top1=20, top2=10, delta=2, t1=10, t2=22, expand_threshold=2**(-12), speck=64, population_size =2000):
    root = (diff_l, diff_r)
    population = [[root]]  # 按层存储所有路径
    current_paths = [[root]]  # 当前层的所有路径  
    print(f"=== 开始分层搜索 (总层数: {nr}) ===")
    for layer in range(nr):
        print(f"\n--- 正在处理第 {layer+1} 层 (当前路径数: {len(current_paths)}) ---")
        next_paths = []
        expanded_nodes = 0
        for path_idx, path in enumerate(current_paths):
            last_node = path[-1]
            # 进度输出（每100条路径输出一次）
            if path_idx % 100 == 0:
                print(f"  处理进度: {path_idx}/{len(current_paths)} 条路径", end='\r')
            # 如果节点没有子节点，则生成子节点
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
            # 添加有效子节点到新路径
            valid_children = tree[last_node]
            top = min(len(valid_children), top2)
            for child_node in valid_children[:top]:
                new_path = path.copy()
                new_path.append(child_node)
                next_paths.append(new_path)
        print(f"\n  本层扩展了 {expanded_nodes} 个新节点")
        population.append(next_paths)
        current_paths = next_paths
        # 从第4层开始进行选择
        if layer >= 3 and len(current_paths) > population_size:
            print(f"  正在进行路径选择 (当前路径数: {len(current_paths)})...")
            evaluated_paths = [(evaluate_path_add(path, nr), path) for path in current_paths]
            evaluated_paths.sort(key=lambda x: x[0])  # 按分数排序
            selected_paths = [path for (score, path) in evaluated_paths[:population_size]]
            print(f"  选择完成: 保留前{population_size}条最优路径 (最低分数: {evaluated_paths[0][0]})")
            current_paths = selected_paths
    # 结果输出到文件
    save_path1 = f"./saved_ea_txt/speck{speck}/speck{speck}_result_ea_{nr}round_({diff_l},{diff_r}).txt"
    root_diff_l = diff_l
    root_diff_r = diff_r
    with open(save_path1, 'w') as f:
        f.write("=== 差分路径搜索结果 ===\n")
        f.write(f"总层数: {nr}\n")
        f.write(f"最终保留路径数: {len(current_paths)}\n\n")
        if len(current_paths) > 0:
            for i, path in enumerate(current_paths):
                f.write(f"\n---- 路径 {i+1} (总分: {evaluate_path_add(path, nr)}) ----\n")
                for j in range(len(path) - 1):
                    diff_l = path[j][0]
                    diff_r = path[j][1]
                    child = path[j + 1][0]
                    node_score = -log2(xdp_add(ror(diff_l, ALPHA()), diff_r, child)) if j < len(path) else 0
                    f.write(f"层 {j}: {diff_l}, {diff_r}, ({hex(diff_l)}, {hex(diff_r)})")
                    if j < len(path):
                        f.write(f" | 分数: {node_score}")
                    f.write("\n")
                f.write(f"层 {len(path)-1}: {path[len(path)-1][0]}, {path[len(path)-1][1]}, ({hex(path[len(path)-1][0])}, {hex(path[len(path)-1][1])})")
                f.write("\n")
    # 保存结果
    save_path2 = f"./saved_trail/speck{speck}_{nr}round_({root_diff_l},{root_diff_r})_evolution_data.pkl"
    with open(save_path2, 'wb') as f:
        pickle.dump((current_paths, tree), f) 
    return current_paths, tree

def evaluate_path_add(path, nr):
    """评估路径的适应度分数"""
    fitness_score = 0
    for i in range(len(path) - 1):
        diff_l, diff_r = path[i]
        child = path[i + 1][0]
        fitness_score += -log2(xdp_add(ror(diff_l, ALPHA()), diff_r, child))
    # 若路径长度足够（接近完整轮数），仅返回实际得分
    if len(path) > nr - 2:
        return fitness_score
    # 否则加入预估得分（基于最后节点的左右差分）
    else:
        diff_l_last, diff_r_last = path[-1]
        estimated_score = construct_best_gamma_add(diff_l_last, diff_r_last)
        return fitness_score + estimated_score

def first_best_search_sub(diff_l, diff_r, tree, nr, top1=20, top2=10, delta=2, t1=10, t2=22, expand_threshold=2**(-12), speck=64, population_size =2000):
    root = (diff_l, diff_r)
    population = [[root]]  # 按层存储所有路径
    current_paths = [[root]]  # 当前层的所有路径  
    print(f"=== 开始分层搜索 (总层数: {nr}) ===")
    for layer in range(nr):
        print(f"\n--- 正在处理第 {layer+1} 层 (当前路径数: {len(current_paths)}) ---")
        next_paths = []
        expanded_nodes = 0
        for path_idx, path in enumerate(current_paths):
            last_node = path[-1]
            # 进度输出（每100条路径输出一次）
            if path_idx % 100 == 0:
                print(f"  处理进度: {path_idx}/{len(current_paths)} 条路径", end='\r')
            # 如果节点没有子节点，则生成子节点
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
                            r_new = ror(last_node[0] ^ last_node[1], BETA())
                            child_node = (l_new, r_new)
                            p = xdp_add(ror(last_node[0], ALPHA()), last_node[1], l_new)
                            tree[last_node].append(child_node)
                            expanded_nodes += 1
            # 添加有效子节点到新路径
            valid_children = tree[last_node]
            top = min(len(valid_children), top2)
            for child_node in valid_children[:top]:
                new_path = path.copy()
                new_path.append(child_node)
                next_paths.append(new_path)
        print(f"\n  本层扩展了 {expanded_nodes} 个新节点")
        population.append(next_paths)
        current_paths = next_paths
        # 从第4层开始进行选择
        if layer >= 3 and len(current_paths) > population_size:
            print(f"  正在进行路径选择 (当前路径数: {len(current_paths)})...")
            evaluated_paths = [(evaluate_path_sub(path, nr), path) for path in current_paths]
            evaluated_paths.sort(key=lambda x: x[0])  # 按分数排序
            selected_paths = [path for (score, path) in evaluated_paths[:population_size]]
            print(f"  选择完成: 保留前{population_size}条最优路径 (最低分数: {evaluated_paths[0][0]})")
            current_paths = selected_paths
    # 结果输出到文件
    save_path1 = f"./saved_ea_txt/speck{speck}/speck{speck}_result_ea_{nr}round_{diff_l}.txt"
    root_diff_l = diff_l
    with open(save_path1, 'w') as f:
        f.write("=== 差分路径搜索结果 ===\n")
        f.write(f"总层数: {nr}\n")
        f.write(f"最终保留路径数: {len(current_paths)}\n\n")
        if len(current_paths) > 0:
            for i, path in enumerate(current_paths):
                f.write(f"\n---- 路径 {i+1} (总分: {evaluate_path_sub(path, nr)}) ----\n")
                for j in range(len(path) - 1):
                    diff_l = path[j][0]
                    diff_r = path[j][1]
                    child = path[j + 1][0]
                    node_score = -log2(xdp_add(diff_l, ror(diff_l ^ diff_r, BETA()), ror(child, ALPHA()))) if j < len(path) else 0
                    f.write(f"层 {j}: {diff_l}, {diff_r}, ({hex(diff_l)}, {hex(diff_r)})")
                    if j < len(path):
                        f.write(f" | 分数: {node_score}")
                    f.write("\n")
                f.write(f"层 {len(path)-1}: {path[len(path)-1][0]}, {path[len(path)-1][1]}, ({hex(path[len(path)-1][0])}, {hex(path[len(path)-1][1])})")
                f.write("\n")   
    # 保存结果
    save_path2 = f"./saved_trail/speck{speck}_{nr}round_{root_diff_l}_evolution_data.pkl"
    with open(save_path2, 'wb') as f:
        pickle.dump((current_paths, tree), f)                 
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

def expand_single_path_add(args):
    """
    单条路径的扩展逻辑，用于多进程调用（add 方向）。
    args = (path, tree, top1, top2, delta, t1, t2, expand_threshold)
    返回: (expanded_paths, expanded_nodes)
    """
    path, tree, top1, top2, delta, t1, t2, expand_threshold = args
    expanded_paths = []
    expanded_nodes = 0
    last_node = path[-1]
    # 若该节点尚未扩展，则生成子节点
    if len(tree[last_node]) == 0:
        x_ham = ham_w(last_node[0])
        y_ham = ham_w(last_node[1])
        if x_ham <= t1 and y_ham <= t1:
            possible_children, sorted_scores = generate_possible_children_add(last_node[0], last_node[1], delta, t1)
            top_children = possible_children[:top1]
            for child in top_children:
                z_ham = ham_w(child)
                # expand 条件（和你原来的一致）
                if (x_ham + y_ham + z_ham <= t2) and (xdp_add(ror(last_node[0], ALPHA()), last_node[1], child) >= expand_threshold):
                    l_new = child
                    r_new = child ^ rol(last_node[1], BETA())
                    child_node = (l_new, r_new)
                    # 注意：在多进程中直接修改普通 dict 不会回写到主进程
                    tree[last_node].append(child_node)
                    expanded_nodes += 1
    # 生成新路径
    valid_children = tree[last_node]
    top = min(len(valid_children), top2)
    for child_node in valid_children[:top]:
        new_path = path.copy()
        new_path.append(child_node)
        expanded_paths.append(new_path)
    return expanded_paths, expanded_nodes

def first_best_search_add_parallel(diff_l, diff_r, tree, nr,top1=20, top2=10, delta=2,t1=10, t2=22, expand_threshold=2**(-12), speck=64, population_size=2000, n_jobs=8):
    """
    多进程版本的 first_best_search_add。
    如果 tree 需要在子进程中修改并且希望父进程看到这些修改，请在外部使用:
        from multiprocessing import Manager
        manager = Manager()
        tree = manager.dict(tree)  # 并且 tree 的每个 value 也应当是 manager.list 或可共享结构
    或者改为让 expand_single_path_add 返回新增子节点并在主进程合并（更安全，推荐）。
    """
    root = (diff_l, diff_r)
    population = [[root]]
    current_paths = [[root]]
    print(f"=== 开始分层搜索 (总层数: {nr}) ===")
    total_start_time = time.time()
    for layer in range(nr):
        layer_start = time.time()
        print(f"\n--- 正在处理第 {layer+1} 层 (当前路径数: {len(current_paths)}) ---")
        next_paths = []
        expanded_nodes_total = 0
        # 组装任务参数
        task_args = [(path, tree, top1, top2, delta, t1, t2, expand_threshold) for path in current_paths]
        # 并行执行扩展任务
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(expand_single_path_add, args) for args in task_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"第 {layer+1} 层处理中", ncols=90):
                expanded, expanded_nodes = future.result()
                next_paths.extend(expanded)
                expanded_nodes_total += expanded_nodes
        print(f"  ✅ 本层扩展了 {expanded_nodes_total} 个新节点，共生成 {len(next_paths)} 条路径，用时 {time.time()-layer_start:.1f}s")
        population.append(next_paths)
        current_paths = next_paths
        # 从第4层开始进行选择
        # === 每层都进行排序（保证并行情况下顺序稳定） ===
        if len(current_paths) > 0:
            print(f"  正在进行路径排序 (当前路径数: {len(current_paths)})...")

            evaluated_paths = [(evaluate_path_add(path, nr), path) for path in current_paths]
            evaluated_paths.sort(key=lambda x: x[0])  # 按分数从小到大排序

            # 是否需要剪枝
            if len(evaluated_paths) > population_size:
                evaluated_paths = evaluated_paths[:population_size]
                print(f"  ✅ 剪枝完成: 保留前{population_size}条最优路径 (最低分数: {evaluated_paths[0][0]})")
            else:
                print(f"  ✅ 排序完成: 当前最优分数 {evaluated_paths[0][0]}")

            current_paths = [path for (_, path) in evaluated_paths]

        # 若无可扩展路径则提前结束
        if not current_paths:
            print("⚠️ 无可扩展路径，搜索提前结束。")
            break
    total_elapsed = time.time() - total_start_time
    print(f"\n🎯 搜索完成，总耗时 {total_elapsed/60:.2f} 分钟")
    # === 输出结果 ===
    save_path1 = f"./saved_ea_txt/speck{speck}/speck{speck}_result_egbs_{nr}round_({diff_l},{diff_r}).txt"
    root_diff_l = diff_l
    root_diff_r = diff_r
    with open(save_path1, 'w') as f:
        f.write("=== 差分路径搜索结果 ===\n")
        f.write(f"Speck版本: {speck}\n")
        f.write(f"总层数: {nr}\n")
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
    # === 保存结果 ===
    save_path2 = f"./saved_trail/speck{speck}_{nr}round_({root_diff_l},{root_diff_r})_evolution_egbs_data.pkl"
    with open(save_path2, 'wb') as f:
        pickle.dump((current_paths, tree), f)
    print(f"\n📁 结果已保存至:\n  {save_path1}\n  {save_path2}")
    return current_paths, tree

def expand_single_path_sub(args):
    """单条路径的扩展逻辑，用于多进程调用"""
    path, tree, top1, top2, delta, t1, t2, expand_threshold = args
    expanded_paths = []
    expanded_nodes = 0
    last_node = path[-1]
    # 若该节点尚未扩展，则生成子节点
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
                    r_new = ror(last_node[0] ^ last_node[1], BETA())
                    child_node = (l_new, r_new)
                    tree[last_node].append(child_node)
                    expanded_nodes += 1
    # 生成新路径
    valid_children = tree[last_node]
    top = min(len(valid_children), top2)
    for child_node in valid_children[:top]:
        new_path = path.copy()
        new_path.append(child_node)
        expanded_paths.append(new_path)
    return expanded_paths, expanded_nodes

def first_best_search_sub_parallel(diff_l, diff_r, tree, nr, top1=20, top2=10, delta=2, t1=10, t2=22, expand_threshold=2**(-12), speck=64, population_size=2000, n_jobs=8):
    """
    多进程版本的 first_best_search_sub。
    """
    root = (diff_l, diff_r)
    population = [[root]]
    current_paths = [[root]]
    print(f"=== 开始分层搜索 (总层数: {nr}) ===")
    total_start_time = time.time()
    for layer in range(nr):
        layer_start = time.time()
        print(f"\n--- 正在处理第 {layer + 1} 层 (当前路径数: {len(current_paths)}) ---")
        next_paths = []
        expanded_nodes_total = 0
        # 组装任务参数
        task_args = [(path, tree, top1, top2, delta, t1, t2, expand_threshold) for path in current_paths]
        # 并行执行扩展任务
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(expand_single_path_sub, args) for args in task_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"第 {layer+1} 层处理中", ncols=90):
                expanded, expanded_nodes = future.result()
                next_paths.extend(expanded)
                expanded_nodes_total += expanded_nodes
        print(f"  ✅ 本层扩展了 {expanded_nodes_total} 个新节点，共生成 {len(next_paths)} 条路径，用时 {time.time()-layer_start:.1f}s")
        population.append(next_paths)
        current_paths = next_paths
        # 从第4层开始进行选择
        # === 每层都进行排序（保证结果稳定） ===
        if len(current_paths) > 0:
            print(f"  正在进行路径排序 (当前路径数: {len(current_paths)})...")

            evaluated_paths = [(evaluate_path_sub(path, nr), path) for path in current_paths]
            evaluated_paths.sort(key=lambda x: x[0])  # 按分数从小到大排序

            # 是否需要剪枝
            if len(evaluated_paths) > population_size:
                evaluated_paths = evaluated_paths[:population_size]
                print(f"  ✅ 选择完成: 保留前{population_size}条最优路径 (最低分数: {evaluated_paths[0][0]})")
            else:
                print(f"  ✅ 排序完成: 当前最优分数 {evaluated_paths[0][0]}")

            current_paths = [path for (_, path) in evaluated_paths]

        # 若无可扩展路径则提前结束
        if not current_paths:
            print("⚠️ 无可扩展路径，搜索提前结束。")
            break
    total_elapsed = time.time() - total_start_time
    print(f"\n🎯 搜索完成，总耗时 {total_elapsed/60:.2f} 分钟")
    # === 输出结果 ===
    root_diff_l = diff_l
    root_diff_r = diff_r
    save_path1 = f"./saved_ea_txt/speck{speck}/speck{speck}_result_sub_egbs_{nr}round_({diff_l},{diff_r}).txt"
    with open(save_path1, 'w') as f:
        f.write("=== 差分路径搜索结果 ===\n")
        f.write(f"Speck版本: {speck}\n")
        f.write(f"总层数: {nr}\n")
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
    # === 保存结果 ===
    save_path2 = f"./saved_trail/speck{speck}_{nr}round_({root_diff_l},{root_diff_r})_evolution_sub_egbs_data.pkl"
    with open(save_path2, 'wb') as f:
        pickle.dump((current_paths, tree), f)

    print(f"\n📁 结果已保存至:\n  {save_path1}\n  {save_path2}")
    return current_paths, tree

def _batched(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def first_best_search_sub_parallel_batch(diff_l, diff_r, tree, nr,
                                   top1=20, top2=10, delta=2,
                                   t1=10, t2=22, expand_threshold=2**(-12),
                                   speck=64, population_size=2000,
                                   n_jobs=8, batch_size=64):
    """
    多进程版本的 first_best_search_sub。
    改进点：
    1. 不再一层内一次性 submit 全部任务，而是按 batch 分批提交，降低峰值内存。
    2. 仍然保持“整层结束后再统一排序/截断”的逻辑，不改变层级搜索语义。
    """
    root = (diff_l, diff_r)
    population = [[root]]
    current_paths = [[root]]

    print(f"=== 开始分层搜索 (总层数: {nr}) ===")
    total_start_time = time.time()

    for layer in range(nr):
        layer_start = time.time()
        print(f"\n--- 正在处理第 {layer + 1} 层 (当前路径数: {len(current_paths)}) ---")

        next_paths = []
        expanded_nodes_total = 0

        task_args = [
            (path, tree, top1, top2, delta, t1, t2, expand_threshold)
            for path in current_paths
        ]

        total_batches = (len(task_args) + batch_size - 1) // batch_size

        # 每层只创建一次进程池；层内按 batch 分批提交
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for batch_idx, batch_args in enumerate(_batched(task_args, batch_size), start=1):
                futures = [executor.submit(expand_single_path_sub, args) for args in batch_args]

                desc = f"第 {layer + 1} 层 batch {batch_idx}/{total_batches}"
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc, ncols=100):
                    expanded, expanded_nodes = future.result()
                    next_paths.extend(expanded)
                    expanded_nodes_total += expanded_nodes

        print(
            f"  ✅ 本层扩展了 {expanded_nodes_total} 个新节点，"
            f"共生成 {len(next_paths)} 条路径，用时 {time.time() - layer_start:.1f}s"
        )

        population.append(next_paths)
        current_paths = next_paths

        # 每层统一排序/剪枝，保持和原逻辑一致
        if len(current_paths) > 0:
            print(f"  正在进行路径排序 (当前路径数: {len(current_paths)})...")

            evaluated_paths = [(evaluate_path_sub(path, nr), path) for path in current_paths]
            evaluated_paths.sort(key=lambda x: x[0])

            if len(evaluated_paths) > population_size:
                evaluated_paths = evaluated_paths[:population_size]
                print(f"  ✅ 选择完成: 保留前{population_size}条最优路径 (最低分数: {evaluated_paths[0][0]})")
            else:
                print(f"  ✅ 排序完成: 当前最优分数 {evaluated_paths[0][0]}")

            current_paths = [path for (_, path) in evaluated_paths]

        if not current_paths:
            print("⚠️ 无可扩展路径，搜索提前结束。")
            break

    total_elapsed = time.time() - total_start_time
    print(f"\n🎯 搜索完成，总耗时 {total_elapsed/60:.2f} 分钟")

    # === 输出结果 ===
    root_diff_l = diff_l
    root_diff_r = diff_r
    save_path1 = f"./saved_ea_txt/speck{speck}/speck{speck}_result_sub_egbs_{nr}round_({diff_l},{diff_r}).txt"
    with open(save_path1, 'w') as f:
        f.write("=== 差分路径搜索结果 ===\n")
        f.write(f"Speck版本: {speck}\n")
        f.write(f"总层数: {nr}\n")
        f.write(f"最终保留路径数: {len(current_paths)}\n")
        f.write(f"总耗时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)\n\n")
        for i, path in enumerate(current_paths):
            # 这里顺手改成了 evaluate_path_sub，和本函数语义一致
            score = evaluate_path_sub(path, nr)
            f.write(f"\n---- 路径 {i + 1} (总分: {score:.4f}) ----\n")
            for j in range(len(path) - 1):
                diff_l, diff_r = path[j]
                child = path[j + 1][0]
                node_score = -log2(xdp_add(ror(diff_l, ALPHA()), diff_r, child))
                f.write(f"层 {j}: ({hex(diff_l)}, {hex(diff_r)}) | 分数: {node_score:.4f}\n")
            f.write(f"层 {len(path) - 1}: ({hex(path[-1][0])}, {hex(path[-1][1])})\n")

    save_path2 = f"./saved_trail/speck{speck}_{nr}round_({root_diff_l},{root_diff_r})_evolution_sub_egbs_data.pkl"
    with open(save_path2, 'wb') as f:
        pickle.dump((current_paths, tree), f)

    print(f"\n📁 结果已保存至:\n  {save_path1}\n  {save_path2}")
    return current_paths, tree



if __name__ == "__main__":
    # (diff_l, diff_r) = (0x80, 0x0)
    # def make_tree():
    #     return defaultdict(list) 
    # tree = make_tree()
    # tree[(diff_l, diff_r)] = []     
    # current_paths = first_best_search_sub_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=15, top1=10, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size=2000, n_jobs=16)


    # for i in range(48):
    #     diff_l = 1 << i
    #     diff_r = 0x0
    #     def make_tree():
    #         return defaultdict(list)  
    #     tree = make_tree()
    #     tree[(diff_l, diff_r)] = []     
    #     current_paths = first_best_search_sub_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=3, top1=20, top2=20, delta=2, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size =2000, n_jobs=16)

    # word_size = 32
    # MASK_VAL = 2 ** word_size - 1
    # diff_l = 0x8000000
    # diff_r = 0x0
    # def make_tree():
    #     return defaultdict(list)  
    # tree = make_tree()
    # tree[(diff_l, diff_r)] = []     
    # current_paths = first_best_search_add_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=12, top1=10, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-18), speck=2*word_size, population_size =100, n_jobs=32)
    

    # diff_l, diff_r  = 0x41010103, 0x49410901
    # # new_r = 0x19000000001 ^ rol(0x1202000001, BETA())
    # # print(hex(new_r))
    # # print(judge_valid(ror(diff_l, ALPHA()), diff_r, child))
    # sorted_children, sorted_scores = generate_possible_children_add(diff_l, diff_r, delta=3, t1=12)
    
    # print(sorted_children[:30], sorted_scores[:30])
    # # print(hex(sorted_children[0]))
    # print(hex(sorted_children[0]),hex(sorted_children[1]),hex(sorted_children[2]),hex(sorted_children[3]),hex(sorted_children[4]))
    # print(hex(sorted_children[5]),hex(sorted_children[6]),hex(sorted_children[7]),hex(sorted_children[8]),hex(sorted_children[9]))
    # print(hex(sorted_children[10]),hex(sorted_children[11]),hex(sorted_children[12]),hex(sorted_children[13]),hex(sorted_children[14]))
    # print(hex_2_bin(sorted_children[0]))
    # print(hex_2_bin(sorted_children[3]))
    # print(hex_2_bin(0x49800802))
    
    # # print(hex_2_bin(0x4007f040404))
    # print(hex_2_bin(0x772400040))
    
    # diff_l = 0x80
    # diff_r = 0x0
    # def make_tree():
    #     return defaultdict(list)  
    # tree = make_tree()
    # tree[(diff_l, diff_r)] = []     
    # current_paths = first_best_search_add_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=9, top1=10, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size =2000, n_jobs=32)
    # current_paths = first_best_search_sub_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=16, top1=10, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size =2000, n_jobs=32)
    # current_paths = first_best_search_sub_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=15, top1=10, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size =2000, n_jobs=32)
    
    
    # diff_l = 0x80
    # diff_r = 0x0

    # def make_tree():
    #     return defaultdict(list)

    # # 运行次数
    # repeat = 10

    # times_add = []
    # times_sub16 = []
    # times_sub15 = []

    # for _ in range(repeat):

    #     tree = make_tree()
    #     tree[(diff_l, diff_r)] = []

    #     # ---- add_parallel ----
    #     start = time.perf_counter()
    #     current_paths = first_best_search_add_parallel(
    #         diff_l=diff_l, diff_r=diff_r, tree=tree,
    #         nr=4, top1=10, top2=10, delta=2,
    #         t1=12, t2=24,
    #         expand_threshold=2**(-16),
    #         speck=2*word_size,
    #         population_size=2000,
    #         n_jobs=32
    #     )
    #     end = time.perf_counter()
    #     times_add.append(end - start)

    #     # ---- sub_parallel nr=16 ----
    #     start = time.perf_counter()
    #     current_paths = first_best_search_sub_parallel(
    #         diff_l=diff_l, diff_r=diff_r, tree=tree,
    #         nr=16, top1=10, top2=10, delta=2,
    #         t1=12, t2=24,
    #         expand_threshold=2**(-16),
    #         speck=2*word_size,
    #         population_size=2000,
    #         n_jobs=32
    #     )
    #     end = time.perf_counter()
    #     times_sub16.append(end - start)

    #     # ---- sub_parallel nr=15 ----
    #     start = time.perf_counter()
    #     current_paths = first_best_search_sub_parallel(
    #         diff_l=diff_l, diff_r=diff_r, tree=tree,
    #         nr=15, top1=10, top2=10, delta=2,
    #         t1=12, t2=24,
    #         expand_threshold=2**(-16),
    #         speck=2*word_size,
    #         population_size=2000,
    #         n_jobs=32
    #     )
    #     end = time.perf_counter()
    #     times_sub15.append(end - start)


    # print("Average time add_parallel (nr=4):", np.mean(times_add))
    # print("Average time sub_parallel (nr=16):", np.mean(times_sub16))
    # print("Average time sub_parallel (nr=15):", np.mean(times_sub15))
        
    word_size = 64
    MASK_VAL = 2 ** word_size - 1
    diff_l = 0x80
    diff_r = 0x0
    def make_tree():
        return defaultdict(list)  
    tree = make_tree()
    tree[(diff_l, diff_r)] = []     
    current_paths2 = first_best_search_sub_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=16, top1=10, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size =200, n_jobs=32)

    # word_size = 48
    # MASK_VAL = 2 ** word_size - 1
    # for i in [12,14,16,18,20,21,23,25,27,29,31,33,36,38]:
    #     diff_l = 1 << i
    #     # diff_l = 0x8000000
    #     diff_r = 0x0
    #     def make_tree():
    #         return defaultdict(list)  
    #     tree = make_tree()
    #     tree[(diff_l, diff_r)] = []     
    #     current_paths = first_best_search_add_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=11, top1=10, top2=10, delta=2, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size =2000, n_jobs=32)

    # for i in range(10):
    #     # diff_l = 0x0
    #     diff_l = 0x0
    #     diff_r = 0x800000
    #     def make_tree():
    #         return defaultdict(list)  
    #     tree = make_tree()
    #     tree[(diff_l, diff_r)] = []     
    #     current_paths = first_best_search_add_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=i+1, top1=20, top2=20, delta=2, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size =2000, n_jobs=16)
    #     current_paths = first_best_search_sub_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=i+1, top1=20, top2=20, delta=2, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size =2000, n_jobs=16)

    # for i in range(10):
    #     diff_l = 0x0
    #     diff_l = 0x80
    #     diff_r = 0x0
    #     def make_tree():
    #         return defaultdict(list)  
    #     tree = make_tree()
    #     tree[(diff_l, diff_r)] = []     
    #     current_paths = first_best_search_add_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=10, top1=20, top2=20, delta=4, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size =2000, n_jobs=16)
    #     current_paths = first_best_search_sub_parallel(diff_l=diff_l, diff_r=diff_r, tree=tree, nr=i+1, top1=20, top2=20, delta=2, t1=12, t2=24, expand_threshold=2**(-16), speck=2*word_size, population_size =2000, n_jobs=16)


    # word_size = 64
    # MASK_VAL = 2 ** word_size - 1

    # diff_l = 0x80
    # diff_r = 0x0
    # nr = 7
    # top1 = 10
    # delta = 2
    # t1_limit = 12
    # t2_limit = 24
    # expand_threshold = 2 ** (-18)

    # t_start = time.time()
    # best_path, best_score = brute_force_best_path_sub_mp(
    #     diff_l,
    #     diff_r,
    #     nr=nr,
    #     top1=top1,
    #     delta=delta,
    #     t1=t1_limit,
    #     t2=t2_limit,
    #     expand_threshold=expand_threshold,
    #     processes=32,
    #     chunk_states=500,
    #     use_children_cache=True,
    #     verbose=True,
    #     show_progress=True,
    # )
    # t_end = time.time()

    # # Verify score with the same evaluation used elsewhere (should match best_score)
    # verify_score = evaluate_path_sub(best_path) if best_path and len(best_path) > 1 else float('inf')

    # speck = 2 * word_size
    # out_txt_dir = _ensure_dir(_out_path("saved_ea_txt"))
    # out_pkl_dir = _ensure_dir(_out_path("saved_trail"))
    # result_txt = os.path.join(
    #     out_txt_dir,
    #     f"speck{speck}_sub_bruteforce_mp_nr{nr}_top{top1}_dl{diff_l}.txt",
    # )
    # result_pkl = os.path.join(
    #     out_pkl_dir,
    #     f"speck{speck}_sub_bruteforce_mp_nr{nr}_top{top1}_dl{diff_l}.pkl",
    # )

    # with open(result_txt, "w") as f:
    #     f.write("=== brute_force_best_path_sub_mp result ===\n")
    #     f.write(f"word_size={word_size}, speck={speck}\n")
    #     f.write(f"diff_l={diff_l} ({hex(diff_l)}), diff_r={diff_r} ({hex(diff_r)})\n")
    #     f.write(
    #         f"nr={nr}, top1={top1}, delta={delta}, t1={t1_limit}, t2={t2_limit}, expand_threshold={expand_threshold}\n"
    #     )
    #     f.write(f"best_score={best_score}\n")
    #     f.write(f"verify_score(evaluate_path_sub)={verify_score}\n")
    #     f.write(f"runtime_sec={t_end - t_start}\n\n")

    #     f.write("-- Path (state per round) --\n")
    #     for i, (l, r) in enumerate(best_path):
    #         f.write(f"round {i}: l={l} r={r} ({hex(l)}, {hex(r)})\n")

    #     # Also output per-round chosen gamma and step score
    #     f.write("\n-- Per-round chosen gamma and step_score --\n")
    #     for i in range(len(best_path) - 1):
    #         l, r = best_path[i]
    #         next_l, _next_r = best_path[i + 1]
    #         gamma = ror(next_l, ALPHA())
    #         diff_r_new = ror(l ^ r, BETA())
    #         p = xdp_add(l, diff_r_new, gamma)
    #         step_score = -log2(p)
    #         f.write(
    #             f"round {i}: gamma={gamma} ({hex(gamma)}), p={p}, step_score={step_score}\n"
    #         )

    # with open(result_pkl, "wb") as f:
    #     pickle.dump(
    #         {
    #             "params": {
    #                 "word_size": word_size,
    #                 "diff_l": diff_l,
    #                 "diff_r": diff_r,
    #                 "nr": nr,
    #                 "top1": top1,
    #                 "delta": delta,
    #                 "t1": t1_limit,
    #                 "t2": t2_limit,
    #                 "expand_threshold": expand_threshold,
    #             },
    #             "best_path": best_path,
    #             "best_score": best_score,
    #             "verify_score": verify_score,
    #             "runtime_sec": (t_end - t_start),
    #         },
    #         f,
    #     )

    # print("Best score = ", best_score)
    # print("Verify score = ", verify_score)
    # print("Total run time = ", t_end - t_start)
    # print("Saved txt to:", result_txt)
    # print("Saved pkl to:", result_pkl)















