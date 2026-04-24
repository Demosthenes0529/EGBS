#!/usr/bin/env python3
"""
speck_diff_cnf.py
生成 Speck(32/48/64/96/128) XOR-差分路径搜索的 DIMACS CNF，
用于 CryptoMiniSat 等 SAT 求解器。
建模依据（核心思想）：
- 每轮 Speck: x_{r+1} = (ROTR(x_r, alpha) ⊞ y_r) （圆密钥 XOR 不影响 XOR-差分）
 y_{r+1} = ROTL(y_r, beta) XOR x_{r+1}
- 模加 XOR-差分可行性与 weight 使用 Lipmaa–Moriai 条件：
 eq(a<<1,b<<1,c<<1) ∧ (a ⊕ b ⊕ c ⊕ (b<<1)) = 0
 weight = h(¬eq(a,b,c)) (不含 MSB)
- 用 w[r][j] 表示 ¬eq(a_j,b_j,c_j)（不含 MSB），并可选加入 Σw ≤ W 的 CNF 计数器
输出：
- out.cnf : DIMACS CNF
- out.map.json : 变量编号 ↔ 名称、参数、以及解码所需信息
使用示例见脚本末尾 main() 的帮助信息。
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
# -----------------------------
# Speck 参数（按分组长度 2n）

# -----------------------------
SPECK_PARAMS = {
    # block_bits: (word_bits n, alpha, beta)
    32: (16, 7, 2),
    48: (24, 8, 3),
    64: (32, 8, 3),
    96: (48, 8, 3),
    128: (64, 8, 3),
}
# -----------------------------
# CNF 构建器
# -----------------------------
class CNFBuilder:
    def __init__(self) -> None:
        self._var_counter: int = 0
        self.clauses: List[List[int]] = []
        self.name2var: Dict[str, int] = {}
        self.var2name: Dict[int, str] = {}
        
    def new_var(self, name: str) -> int:
        if name in self.name2var:
            raise ValueError(f"Variable name already used: {name}")
        self._var_counter += 1
        vid = self._var_counter
        self.name2var[name] = vid
        self.var2name[vid] = name
        return vid
    
    def add_clause(self, lits: List[int]) -> None:
        if not lits:
            raise ValueError("Empty clause is not allowed (would mean UNSAT).")
        if any(x == 0 for x in lits):
            lits = [x for x in lits if x != 0]
        self.clauses.append(lits)
        
    def add_unit(self, lit: int) -> None:
        self.add_clause([lit])
        
    @property
    def nvars(self) -> int:
        return self._var_counter
    
    @property
    def nclauses(self) -> int:
        return len(self.clauses)
    
    def write_dimacs(self, path: str, extra_comments: Optional[List[str]] = None) -> None:
        with open(path, "w", encoding="utf-8") as f:
            if extra_comments:
                for c in extra_comments:
                    f.write(f"c {c}\n")
            f.write(f"p cnf {self.nvars} {self.nclauses}\n")
            for clause in self.clauses:
                f.write(" ".join(str(x) for x in clause) + " 0\n")
# -----------------------------
# 基础工具：旋转 / 解析输入
# -----------------------------
def rotr(bits: List[int], alpha: int) -> List[int]:
    n = len(bits)
    return [bits[(i + alpha) % n] for i in range(n)]

def rotl(bits: List[int], beta: int) -> List[int]:
    n = len(bits)
    # out[i] = in[(i-beta) mod n]
    return [bits[(i - beta) % n] for i in range(n)]

def int_to_bits_le(value: int, n: int) -> List[int]:
    """把整数按 little-endian 位序展开：bits[i] = (value>>i)&1"""
    return [(value >> i) & 1 for i in range(n)]

def parse_hex_or_bitstring(s: str, n: int) -> int:
    """
    支持：
    - 0x... 或 ... 形式的 hex
    - 由 0/1 组成的 bitstring（MSB->LSB），长度必须等于 n
    """
    s = s.strip().lower()
    if s.startswith("0x"):
        v = int(s, 16)
    elif re.fullmatch(r"[01]+", s):
        if len(s) != n:
            raise ValueError(f"Bitstring length {len(s)} != word size {n}")
        v = int(s, 2)
    else:
        v = int(s, 16)
    return v & ((1 << n) - 1)

def parse_pair(s: str) -> Tuple[str, str]:
    """
    解析形如：
    - "2040,0040"
    - "0x2040:0x0040"
    - "2040 0040"
    """
    10
    s = s.strip()
    for sep in [",", ":", " "]:
        if sep in s:
            parts = [p for p in s.split(sep) if p]
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
    raise ValueError("Cannot parse pair. Use formats like '2040,0040' or '0x2040:0x0040'.")

# -----------------------------
# CNF 原语：奇偶校验 / XOR / all-equal
# -----------------------------
def _forbid_assignment_clause(vars_ids: List[int], assignment_bits: Tuple[int, ...]) -> List[int]:
    # 通过子句排除某个赋值：若某变量在该赋值里为 1，则子句里放 -var；为 0 则放 +var
    clause: List[int] = []
    for v, b in zip(vars_ids, assignment_bits):
        clause.append(v if b == 0 else -v)
    return clause

def add_parity_even(builder: CNFBuilder, vars_ids: List[int], guard_lit: Optional[int] = None) -> None:
    """
    约束 XOR(vars_ids)=0（偶校验）。
    CNF 方法：禁止所有“奇校验”的赋值（共有 2^{m-1} 个）。
    guard_lit 若给出（通常是 -eq），则实现： eq -> parity_even(...)
    """
    m = len(vars_ids)
    for bits in itertools.product([0, 1], repeat=m):
        if sum(bits) % 2 == 1:
            cl = _forbid_assignment_clause(vars_ids, bits)
            if guard_lit is not None:
                cl = [guard_lit] + cl
            builder.add_clause(cl)
    
def add_xor_gate(builder: CNFBuilder, a: int, b: int, out: int) -> None:
# out = a XOR b <=> a XOR b XOR out = 0
    add_parity_even(builder, [a, b, out])

def add_not_equiv(builder: CNFBuilder, a: int, b: int) -> None:
    """a <-> ¬b"""
    builder.add_clause([a, b])
    builder.add_clause([-a, -b])

def add_eq_all3(builder: CNFBuilder, a: int, b: int, c: int, eq: int) -> None:
    """
    eq <-> (a=b=c)
    等价于：eq <-> (000 or 111)
    11
    用 6 个 CNF 子句编码：
    - 若 a=b=c=0 或 1，则强制 eq=1
    - 若 eq=1，则强制 a==b 且 a==c
    """
    # all zeros -> eq
    builder.add_clause([a, b, c, eq])
    # all ones -> eq
    builder.add_clause([-a, -b, -c, eq])
    # eq -> (a==b)
    builder.add_clause([-eq, -a, b])
    builder.add_clause([-eq, a, -b])
    # eq -> (a==c)
    builder.add_clause([-eq, -a, c])
    builder.add_clause([-eq, a, -c])

# -----------------------------
# 模加差分：AddDiff(α,β->γ) 与 weight 位
# -----------------------------
def add_addition_diff(
builder: CNFBuilder,
alpha_bits: List[int],
beta_bits: List[int],
gamma_bits: List[int],
round_idx: int,
) -> List[int]:
    """
    使用 Lipmaa–Moriai 条件的 CNF 化（参考论文 Eq.(1)(2)(3)）：
    eq(α<<1,β<<1,γ<<1) ∧ (α ⊕ β ⊕ γ ⊕ (β<<1)) = 0
    并按 weight = h(¬eq(α,β,γ))（不含 MSB）生成 w[r][0..n-2]。
    """
    n = len(alpha_bits)
    if len(beta_bits) != n or len(gamma_bits) != n:
        raise ValueError("Bit-width mismatch in addition diff.")
    # i=0: α0 ⊕ β0 ⊕ γ0 = 0
    add_parity_even(builder, [alpha_bits[0], beta_bits[0], gamma_bits[0]])
    weight_bits: List[int] = []
    # i=1..n-1
    for i in range(1, n):
        # eq_i 表示 eq(α_{i-1},β_{i-1},γ_{i-1})
        eq = builder.new_var(f"eq_r{round_idx}_b{i}")
        add_eq_all3(builder, alpha_bits[i - 1], beta_bits[i - 1], gamma_bits[i - 1], eq)
        # 当 eq=1 时，必须满足：α_i ⊕ β_i ⊕ γ_i ⊕ β_{i-1} = 0
        add_parity_even(
            builder,
            [alpha_bits[i], beta_bits[i], gamma_bits[i], beta_bits[i - 1]],
            guard_lit=-eq,
        )
        # weight 位：w_{i-1} <-> ¬eq_i （对 j=i-1，覆盖 0..n-2）
        w = builder.new_var(f"w_r{round_idx}_b{i-1}")
        add_not_equiv(builder, w, eq)
        weight_bits.append(w)
    # 共 n-1 个 weight 位
    return weight_bits

# -----------------------------
# Σ weight ≤ W：Sinz 顺序计数器
# -----------------------------
def add_atmost_k_sinz(builder: CNFBuilder, lits: List[int], k: int, prefix: str = "sc") -> None:
    """
    使用 Sinz sequential counter 生成 CNF：sum(lits) <= k
    lits 必须是“正文字”（变量为真表示计数 1）。
    """
    n = len(lits)
    if k >= n:
        return
    if k == 0:
        for x in lits:
            builder.add_unit(-x)
        return
    if k < 0:
        raise ValueError("k must be >= 0")
    # s[i][j], i=0..n-1, j=0..k-1
    s: List[List[int]] = [[builder.new_var(f"{prefix}_{i}_{j}") for j in range(k)] for i in range(n)]
    # s[0][j]=false for j>=1
    for j in range(1, k):
        builder.add_unit(-s[0][j])
    # (¬x_i ∨ s[i][0])
    for i in range(n):
        builder.add_clause([-lits[i], s[i][0]])
    # propagate + carry + overflow
    for i in range(1, n):
        # (¬s[i-1][j] ∨ s[i][j])
        for j in range(k):
            builder.add_clause([-s[i - 1][j], s[i][j]])
        # (¬x_i ∨ ¬s[i-1][j-1] ∨ s[i][j]) for j>=1
        for j in range(1, k):
            builder.add_clause([-lits[i], -s[i - 1][j - 1], s[i][j]])
        # overflow: (¬x_i ∨ ¬s[i-1][k-1])
        builder.add_clause([-lits[i], -s[i - 1][k - 1]])
        
# -----------------------------
# Speck CNF 构建
# -----------------------------
@dataclass
class BuildResult:
    builder: CNFBuilder
    meta: Dict
    round_bits: Dict[str, List[List[int]]]
    weight_vars: List[int]

def build_speck_cnf(
    block_bits: int,
    rounds: int,
    in_dx: Optional[int],
    in_dy: Optional[int],
    out_dx: Optional[int],
    out_dy: Optional[int],
    max_weight: Optional[int],
    mid_round: Optional[int],
    mid_word: Optional[str],
    mid_diff: Optional[int],
) -> BuildResult:
    if block_bits not in SPECK_PARAMS:
        raise ValueError(f"Unsupported Speck block size: {block_bits}")
    n, alpha, beta = SPECK_PARAMS[block_bits]
    b = CNFBuilder()
    # state vars: x[r][i], y[r][i]
    x: List[List[int]] = []
    y: List[List[int]] = []
    for r in range(rounds + 1):
        xb = [b.new_var(f"x{r}_b{i}") for i in range(n)]
        yb = [b.new_var(f"y{r}_b{i}") for i in range(n)]
        x.append(xb)
        y.append(yb)
    # fix input (optional)
    if in_dx is not None:
        for i, bit in enumerate(int_to_bits_le(in_dx, n)):
            b.add_unit(x[0][i] if bit else -x[0][i])

    if in_dy is not None:
        for i, bit in enumerate(int_to_bits_le(in_dy, n)):
            b.add_unit(y[0][i] if bit else -y[0][i])
    # fix output (optional)
    if out_dx is not None:
        for i, bit in enumerate(int_to_bits_le(out_dx, n)):
            b.add_unit(x[rounds][i] if bit else -x[rounds][i])
    if out_dy is not None:
        for i, bit in enumerate(int_to_bits_le(out_dy, n)):
            b.add_unit(y[rounds][i] if bit else -y[rounds][i])
    # fix intermediate diff (optional heuristic)
    if mid_round is not None:
        if not (0 <= mid_round <= rounds):
            raise ValueError("--mid-round out of range")
        if mid_word not in ("x", "y"):
            raise ValueError("--mid-word must be 'x' or 'y'")
        if mid_diff is None:
            raise ValueError("--mid-diff must be provided when using --mid-round")
        target = x[mid_round] if mid_word == "x" else y[mid_round]
        for i, bit in enumerate(int_to_bits_le(mid_diff, n)):
            b.add_unit(target[i] if bit else -target[i])
    weight_vars: List[int] = []
    # per round constraints
    for r in range(rounds):
        a = rotr(x[r], alpha) # α = ROTR(Δx_r, alpha)
        wbits = add_addition_diff(b, a, y[r], x[r + 1], round_idx=r) # γ = Δx_{r+1}
        weight_vars.extend(wbits)
        yl = rotl(y[r], beta) # ROTL(Δy_r, beta)
        for i in range(n):
            add_xor_gate(b, yl[i], x[r + 1][i], y[r + 1][i])
    if max_weight is not None:
        add_atmost_k_sinz(b, weight_vars, max_weight, prefix="sc")
    meta = {
    "speck_block_bits": block_bits,
    "word_bits": n,
    "alpha": alpha,
    "beta": beta,
    "rounds": rounds,
    "in_dx": None if in_dx is None else f"0x{in_dx:0{n//4}x}",
    "in_dy": None if in_dy is None else f"0x{in_dy:0{n//4}x}",
    "out_dx": None if out_dx is None else f"0x{out_dx:0{n//4}x}",
    "out_dy": None if out_dy is None else f"0x{out_dy:0{n//4}x}",
    "max_weight": max_weight,
    "num_weight_bits": len(weight_vars),
    "num_vars": b.nvars,
    "num_clauses": b.nclauses,
    "bit_order": "little-endian: bit0=LSB",
    }
    return BuildResult(
    builder=b,
    meta=meta,
    round_bits={"x": x, "y": y},
    weight_vars=weight_vars,
    )

# -----------------------------
# 模型解码：把 SAT 输出转回逐轮差分
# -----------------------------
def read_dimacs_model(path: str) -> Dict[int, bool]:
    """
    解析 SAT solver 的 DIMACS 输出：
    - 读取所有以 'v' 开头的行
    - 返回 var_id -> bool
    """
    assigns: Dict[int, bool] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("v") or line.startswith("V"):
                parts = line.split()[1:]
                for p in parts:
                    lit = int(p)
                    if lit == 0:
                        continue
                    var = abs(lit)
                    assigns[var] = (lit > 0)
    return assigns

def bits_to_int_le(bits01: List[int]) -> int:
    v = 0
    for i, b in enumerate(bits01):
        v |= (b & 1) << i
    return v

def decode_trail(map_json: str, model_path: str) -> None:
    with open(map_json, "r", encoding="utf-8") as f:
        mp = json.load(f)
    meta = mp["meta"]
    var2name = {int(k): v for k, v in mp["var2name"].items()}
    name2var = {k: int(v) for k, v in mp["name2var"].items()}
    assigns = read_dimacs_model(model_path)
    if not assigns:
        print("No assignments found in model file. Make sure CryptoMiniSat prints 'v ...'.")
        return
    n = meta["word_bits"]
    rounds = meta["rounds"]
    hexw = n // 4

    def get_bit(var_id: int) -> int:
        return 1 if assigns.get(var_id, False) else 0
    # decode state diffs
    print(f"[decode] Speck{meta['speck_block_bits']} word_bits={n}, rounds={rounds}")
    for r in range(rounds + 1):
        xb = [get_bit(name2var[f"x{r}_b{i}"]) for i in range(n)]
        yb = [get_bit(name2var[f"y{r}_b{i}"]) for i in range(n)]
        dx = bits_to_int_le(xb)
        dy = bits_to_int_le(yb)
        print(f"r={r:02d} dx=0x{dx:0{hexw}x} dy=0x{dy:0{hexw}x}")
    # decode total weight (if present)
    w_vars = mp.get("weight_vars", [])
    if w_vars:
        w = sum(1 for vid in w_vars if assigns.get(int(vid), False))
        print(f"[decode] total_weight(bits counted) = {w}")
     
     
def parse_solver_status(output: str) -> str:
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("s "):
            if "UNSATISFIABLE" in line:
                return "UNSAT"
            if "SATISFIABLE" in line:
                return "SAT"
            if "UNKNOWN" in line:
                return "UNKNOWN"
    return "UNKNOWN"


def append_time_log(path: str, row: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cryptominisat(
    cnf_path: str,
    solver: str,
    threads: int,
    model_path: str,
    time_log: str,
    weight: int,
) -> str:
    cmd = [solver, f"--threads={threads}", "--printsol", "1", cnf_path]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    elapsed = time.perf_counter() - t0

    with open(model_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    status = parse_solver_status(proc.stdout)

    append_time_log(time_log, {
        "weight": weight,
        "status": status,
        "seconds": elapsed,
        "returncode": proc.returncode,
        "threads": threads,
        "cnf": cnf_path,
        "model": model_path,
        "cmd": " ".join(cmd),
    })

    print(f"[solve] W={weight} status={status} time={elapsed:.6f}s")
    return status
        
        
def write_case(
    block_bits: int,
    rounds: int,
    in_dx: Optional[int],
    in_dy: Optional[int],
    out_dx: Optional[int],
    out_dy: Optional[int],
    max_weight: Optional[int],
    mid_round: Optional[int],
    mid_word: Optional[str],
    mid_diff: Optional[int],
    cnf_path: str,
    map_path: str,
) -> BuildResult:
    res = build_speck_cnf(
        block_bits=block_bits,
        rounds=rounds,
        in_dx=in_dx,
        in_dy=in_dy,
        out_dx=out_dx,
        out_dy=out_dy,
        max_weight=max_weight,
        mid_round=mid_round,
        mid_word=mid_word,
        mid_diff=mid_diff,
    )

    comments = [
        f"Speck{block_bits} XOR-differential SAT CNF",
        f"word_bits={res.meta['word_bits']} alpha={res.meta['alpha']} beta={res.meta['beta']} rounds={res.meta['rounds']}",
        f"in_dx={res.meta['in_dx']} in_dy={res.meta['in_dy']}",
        f"out_dx={res.meta['out_dx']} out_dy={res.meta['out_dy']}",
        f"max_weight={res.meta['max_weight']}",
        f"num_vars={res.meta['num_vars']} num_clauses={res.meta['num_clauses']}",
    ]
    res.builder.write_dimacs(cnf_path, extra_comments=comments)

    mp = {
        "meta": res.meta,
        "name2var": res.builder.name2var,
        "var2name": {str(k): v for k, v in res.builder.var2name.items()},
        "weight_vars": res.weight_vars,
    }
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mp, f, indent=2, sort_keys=True)

    return res      
        
        
# -----------------------------
# main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate DIMACS CNF for Speck XOR-differential trail search (SAT).")
    ap.add_argument("--variant", type=int, choices=sorted(SPECK_PARAMS.keys()),
        required=True,
        help="Speck block size variant: 32/48/64/96/128 (means Speck32/.. by block bits)."
    )
    ap.add_argument("--free-in", action="store_true", help="Do not fix round-0 input difference.")
    ap.add_argument("--rounds", type=int, required=True, help="Number of rounds to model (R).")
    ap.add_argument("--in-diff", type=str, default=None, help="Input difference pair 'dx,dy' in hex or bitstring. If omitted, defaults to dx=1, dy=0.")
    ap.add_argument("--in-diff-x", type=str, default=None, help="Input dx only (overrides --in￾diff).")
    ap.add_argument("--in-diff-y", type=str, default=None, help="Input dy only (overrides --in￾diff).")
    ap.add_argument("--out-diff", type=str, default=None, help="Fix output difference pair 'dx,dy' (optional).")
    ap.add_argument("--out-diff-x", type=str, default=None, help="Fix output dx only (optional).")
    ap.add_argument("--out-diff-y", type=str, default=None, help="Fix output dy only (optional).")
    ap.add_argument("--max-weight", type=int, default=None, help="Optional upper bound W for total weight bits (Σw ≤ W).")
    ap.add_argument("--mid-round", type=int, default=None, help="Optional heuristic: fix intermediate difference at this round.")
    ap.add_argument("--mid-word", type=str, choices=["x", "y"], default=None, help="When using --mid-round, choose to fix x or y at that round.")
    ap.add_argument("--mid-diff", type=str, default=None,help="Difference value (hex/bitstring) used for --mid-round/--mid-word.")
    ap.add_argument("-o", "--out", type=str, required=False, default="out.cnf", help="Output CNF path.")
    ap.add_argument("--map", type=str, required=False, default="out.map.json", help="Output mapping JSON.")
    ap.add_argument("--decode", action="store_true", help="Decode a SAT model (requires --map and --model).")
    ap.add_argument("--model", type=str, default=None, help="Path to SAT solver output model file (for --decode).")
    ap.add_argument("--solve", action="store_true",
    help="Invoke CryptoMiniSat after generating CNF.")
    ap.add_argument("--solver", type=str, default="cryptominisat5", help="SAT solver executable path.")
    ap.add_argument("--threads", type=int, default=8, help="Number of CryptoMiniSat threads.")
    ap.add_argument("--time-log", type=str, default="cms_time.jsonl", help="Append one JSON line per solver run.")
    ap.add_argument("--search-mode", choices=["binary", "linear"], default="binary", help="How to search the optimal weight when --max-weight is omitted.")
    args = ap.parse_args()
    program_t0 = time.perf_counter()
    # decode mode
    if args.decode:
        if not args.model:
            ap.error("--decode requires --model <solver_output>")
        decode_trail(args.map, args.model)
        return
    # build mode
    block_bits = args.variant
    n, alpha, beta = SPECK_PARAMS[block_bits]

    # parse input diffs
    if args.free_in:
        in_dx = None
        in_dy = None
    else:
        if args.in_diff_x is not None or args.in_diff_y is not None:
            dx_s = args.in_diff_x if args.in_diff_x is not None else "0"
            dy_s = args.in_diff_y if args.in_diff_y is not None else "0"
        elif args.in_diff is not None:
            dx_s, dy_s = parse_pair(args.in_diff)
        else:
            dx_s, dy_s = "0x1", "0x0"   # 默认：单字差分在 x 上

        in_dx = parse_hex_or_bitstring(dx_s, n)
        in_dy = parse_hex_or_bitstring(dy_s, n)

    # parse output diffs (optional)
    out_dx: Optional[int] = None
    out_dy: Optional[int] = None
    if args.out_diff_x is not None:
        out_dx = parse_hex_or_bitstring(args.out_diff_x, n)
    if args.out_diff_y is not None:
        out_dy = parse_hex_or_bitstring(args.out_diff_y, n)
    if args.out_diff is not None:
        ox_s, oy_s = parse_pair(args.out_diff)
        out_dx = parse_hex_or_bitstring(ox_s, n)
        out_dy = parse_hex_or_bitstring(oy_s, n)

    # parse mid diff (optional)
    mid_diff_val: Optional[int] = None
    if args.mid_diff is not None:
        mid_diff_val = parse_hex_or_bitstring(args.mid_diff, n)

    # ---------- build only ----------
    if not args.solve:
        res = write_case(
            block_bits=block_bits,
            rounds=args.rounds,
            in_dx=in_dx,
            in_dy=in_dy,
            out_dx=out_dx,
            out_dy=out_dy,
            max_weight=args.max_weight,
            mid_round=args.mid_round,
            mid_word=args.mid_word,
            mid_diff=mid_diff_val,
            cnf_path=args.out,
            map_path=args.map,
        )
        print(f"[ok] CNF written: {args.out}")
        print(f"[ok] map written: {args.map}")
        print(f"[stats] vars={res.builder.nvars} clauses={res.builder.nclauses}")
        if args.max_weight is not None:
            print(f"[stats] weight_bits={len(res.weight_vars)} max_weight={args.max_weight}")
        return

    # ---------- solve once if max_weight is given ----------
    if args.max_weight is not None:
        res = write_case(
            block_bits=block_bits,
            rounds=args.rounds,
            in_dx=in_dx,
            in_dy=in_dy,
            out_dx=out_dx,
            out_dy=out_dy,
            max_weight=args.max_weight,
            mid_round=args.mid_round,
            mid_word=args.mid_word,
            mid_diff=mid_diff_val,
            cnf_path=args.out,
            map_path=args.map,
        )
        model_path = str(Path(args.out).with_suffix(".solver.txt"))
        status = run_cryptominisat(
            cnf_path=args.out,
            solver=args.solver,
            threads=args.threads,
            model_path=model_path,
            time_log=args.time_log,
            weight=args.max_weight,
        )
        print(f"[ok] CNF written: {args.out}")
        print(f"[ok] map written: {args.map}")
        print(f"[ok] solver output: {model_path}")
        print(f"[stats] vars={res.builder.nvars} clauses={res.builder.nclauses}")
        print(f"[stats] weight_bits={len(res.weight_vars)} max_weight={args.max_weight}")
        print(f"[result] {status}")
        return

    # ---------- auto-search optimal weight ----------
    upper = 2 * n - 1

    best_w = None
    best_tmp_cnf = None
    best_tmp_map = None
    best_tmp_model = None

    with tempfile.TemporaryDirectory(prefix="specksat_") as tmpdir:
        tmpdir = Path(tmpdir)

        def try_weight(W: int) -> str:
            cnf_path = str(tmpdir / f"w{W}.cnf")
            map_path = str(tmpdir / f"w{W}.map.json")
            model_path = str(tmpdir / f"w{W}.solver.txt")

            write_case(
                block_bits=block_bits,
                rounds=args.rounds,
                in_dx=in_dx,
                in_dy=in_dy,
                out_dx=out_dx,
                out_dy=out_dy,
                max_weight=W,
                mid_round=args.mid_round,
                mid_word=args.mid_word,
                mid_diff=mid_diff_val,
                cnf_path=cnf_path,
                map_path=map_path,
            )

            status = run_cryptominisat(
                cnf_path=cnf_path,
                solver=args.solver,
                threads=args.threads,
                model_path=model_path,
                time_log=args.time_log,
                weight=W,
            )
            return status

        status_cache = {}
        unknown_weights = set()
        last_unsat = -1
        proven_optimal = False

        def get_status(W: int) -> str:
            if W not in status_cache:
                status_cache[W] = try_weight(W)
            return status_cache[W]

        if args.search_mode == "linear":
            for W in range(upper + 1):
                status = get_status(W)

                if status == "UNSAT":
                    last_unsat = W
                    continue

                elif status == "SAT":
                    best_w = W
                    best_tmp_cnf = str(tmpdir / f"w{W}.cnf")
                    best_tmp_map = str(tmpdir / f"w{W}.map.json")
                    best_tmp_model = str(tmpdir / f"w{W}.solver.txt")
                    break

                else:  # UNKNOWN
                    unknown_weights.add(W)
                    continue

            if best_w is not None:
                unresolved_below_best = sorted(w for w in unknown_weights if w < best_w)
                proven_optimal = (len(unresolved_below_best) == 0)

        else:
            lo, hi = 0, upper
            while lo <= hi:
                mid = (lo + hi) // 2
                status = get_status(mid)

                if status == "SAT":
                    best_w = mid
                    best_tmp_cnf = str(tmpdir / f"w{mid}.cnf")
                    best_tmp_map = str(tmpdir / f"w{mid}.map.json")
                    best_tmp_model = str(tmpdir / f"w{mid}.solver.txt")
                    hi = mid - 1

                elif status == "UNSAT":
                    last_unsat = max(last_unsat, mid)
                    lo = mid + 1

                else:  # UNKNOWN
                    unknown_weights.add(mid)
                    # 启发式：UNKNOWN 时也向更小的 W 继续试
                    hi = mid - 1

            # 若已经找到某个 SAT，上下界之间再做一次顺序验证
            if best_w is not None:
                for W in range(max(last_unsat + 1, 0), best_w):
                    status = get_status(W)

                    if status == "UNSAT":
                        last_unsat = max(last_unsat, W)
                        continue

                    elif status == "SAT":
                        best_w = W
                        best_tmp_cnf = str(tmpdir / f"w{W}.cnf")
                        best_tmp_map = str(tmpdir / f"w{W}.map.json")
                        best_tmp_model = str(tmpdir / f"w{W}.solver.txt")
                        break

                    else:  # UNKNOWN
                        unknown_weights.add(W)
                        continue

                unresolved_below_best = sorted(w for w in unknown_weights if w < best_w)
                proven_optimal = (len(unresolved_below_best) == 0)

        if best_w is None:
            total_elapsed = time.perf_counter() - program_t0
            append_time_log(args.time_log, {
                "event": "search_summary",
                "mode": "auto_optimal_weight",
                "search_mode": args.search_mode,
                "variant": block_bits,
                "rounds": args.rounds,
                "threads": args.threads,
                "optimal_weight": None,
                "proven_optimal": False,
                "last_unsat": last_unsat,
                "unknown_weights": sorted(unknown_weights),
                "total_seconds": total_elapsed,
                "result": "NO_PROVEN_SAT_FOUND",
            })
            print("[result] No proven SAT weight found within the searched range.")
            print(f"[result] largest proven UNSAT = {last_unsat}")
            if unknown_weights:
                print(f"[warning] unresolved UNKNOWN weights = {sorted(unknown_weights)}")
            print(f"[result] total_search_time = {total_elapsed:.6f}s")
            return

        shutil.copyfile(best_tmp_cnf, args.out)
        shutil.copyfile(best_tmp_map, args.map)
        final_model = str(Path(args.out).with_suffix(".solver.txt"))
        shutil.copyfile(best_tmp_model, final_model)

        total_elapsed = time.perf_counter() - program_t0
        unresolved_below_best = sorted(w for w in unknown_weights if w < best_w)

        append_time_log(args.time_log, {
            "event": "search_summary",
            "mode": "auto_optimal_weight",
            "search_mode": args.search_mode,
            "variant": block_bits,
            "rounds": args.rounds,
            "threads": args.threads,
            "optimal_weight": best_w,
            "proven_optimal": proven_optimal,
            "last_unsat": last_unsat,
            "unknown_weights": sorted(unknown_weights),
            "unresolved_below_best": unresolved_below_best,
            "total_seconds": total_elapsed,
            "final_cnf": args.out,
            "final_map": args.map,
            "final_model": final_model,
        })

        if proven_optimal:
            print(f"[result] proven optimal_weight = {best_w}")
        else:
            print(f"[result] best known SAT weight = {best_w}")
            if unresolved_below_best:
                print(f"[warning] optimality not proven; unresolved lower weights: {unresolved_below_best}")

        print(f"[result] total_search_time = {total_elapsed:.6f}s")
        print(f"[ok] best CNF written: {args.out}")
        print(f"[ok] best map written: {args.map}")
        print(f"[ok] best solver output: {final_model}")
        print(f"[ok] time log: {args.time_log}")



if __name__ == "__main__":
    main()