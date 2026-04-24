from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

State = Tuple[int, int]
ScoreKey = Tuple[int, State]


# -----------------------------
# SPECK / arithmetic utilities
# -----------------------------


def rol(x: int, k: int, word_size: int) -> int:
    mask = (1 << word_size) - 1
    return ((x << k) & mask) | (x >> (word_size - k))


def ror(x: int, k: int, word_size: int) -> int:
    mask = (1 << word_size) - 1
    return ((x >> k) | ((x << (word_size - k)) & mask)) & mask


def hamming_weight(x: int) -> int:
    # Compatible with Python versions that do not provide int.bit_count().
    try:
        return int(x.bit_count())
    except AttributeError:
        return bin(x & ((1 << max(x.bit_length(), 1)) - 1)).count("1")


def _render_progress(label: str, done: int, total: int, bar_width: int = 30) -> None:
    if total <= 0:
        return
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(ratio * bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\r{label} [{bar}] {done}/{total} ({ratio * 100:5.1f}%)", end="", flush=True)
    if done >= total:
        print("", flush=True)


def _update_progress_every_percent(
    label: str, it: int, total: int, last_percent: int, enabled: bool
) -> int:
    if not enabled or total <= 0:
        return last_percent
    current_percent = int(((it + 1) * 100) / total)
    if current_percent > last_percent:
        _render_progress(label, it + 1, total)
        return current_percent
    return last_percent


def _fmt_best(best: Optional["Characteristic"]) -> str:
    if best is None:
        return "best_w=None best_p=None"
    return f"best_w={best.total_weight} best_p=2^(-{best.total_weight})"


@dataclass(frozen=True)
class SpeckParams:
    word_size: int
    alpha: int
    beta_rot: int

    @property
    def mask(self) -> int:
        return (1 << self.word_size) - 1

    @staticmethod
    def for_block_size(block_size: int) -> "SpeckParams":
        if block_size not in (32, 48, 64, 96, 128):
            raise ValueError("block_size must be one of 32, 48, 64, 96, 128")
        word_size = block_size // 2
        if block_size == 32:
            return SpeckParams(word_size=16, alpha=7, beta_rot=2)
        return SpeckParams(word_size=word_size, alpha=8, beta_rot=3)


@dataclass
class SearchConfig:
    block_size: int = 48
    total_rounds: int = 11
    split: Optional[int] = None
    delta: Optional[int] = None
    t1: Optional[int] = None
    t2: Optional[int] = None
    expand_threshold_weight: Optional[int] = None
    forward_iterations: int = 100_000
    backward_iterations: Optional[int] = 200_000
    k: int = 5
    uct_c: float = 0.25
    uct_d: float = 100.0
    beta_mix: float = 0.2
    geom_p: float = 0.25
    seed_transition_weight: int = 1
    target_weight: Optional[int] = None
    rng_seed: int = 0
    initial_difference: Optional[State] = None
    seed_mode: str = "paper_low_weight"  # "paper_low_weight" / "fixed" / "mixed"
    seed_states_override: Optional[List[State]] = None
    deadline_ts: Optional[float] = None
    stop_flag_file: Optional[str] = None

    def finalize(self) -> "SearchConfig":
        cfg = SearchConfig(**self.__dict__)
        params = SpeckParams.for_block_size(cfg.block_size)
        if cfg.delta is None:
            cfg.delta = 1 if params.word_size == 16 else 2
        if cfg.t1 is None:
            cfg.t1 = 8 if params.word_size == 16 else 12
        if cfg.t2 is None:
            cfg.t2 = 20 if params.word_size == 16 else 24
        if cfg.expand_threshold_weight is None:
            cfg.expand_threshold_weight = 12 if params.word_size == 16 else 16
        if cfg.split is None:
            cfg.split = cfg.total_rounds // 2
        if cfg.seed_mode not in {"paper_low_weight", "fixed", "mixed"}:
            raise ValueError('seed_mode must be one of: "paper_low_weight", "fixed", "mixed"')
        return cfg


# -----------------------------
# xdp+ and generalized Lipmaa-Moriai
# -----------------------------


def ne(value: int, mask: int) -> int:
    return (value & mask) ^ mask


def eq(x: int, y: int, z: int, mask: int) -> int:
    return (ne(x, mask) ^ y) & (ne(x, mask) ^ z) & mask


def judge_valid(alpha: int, beta: int, gamma: int, params: SpeckParams) -> bool:
    mask = params.mask
    tmp1 = eq((alpha << 1) & mask, (beta << 1) & mask, (gamma << 1) & mask, mask)
    tmp2 = alpha ^ beta ^ gamma ^ (((beta << 1) & mask))
    return (tmp1 & tmp2) == 0


def xdp_add_weight(alpha: int, beta: int, gamma: int, params: SpeckParams) -> Optional[int]:
    """Return -log2(xdp+) for a valid transition, else None."""
    if not judge_valid(alpha, beta, gamma, params):
        return None
    mask = (1 << (params.word_size - 1)) - 1
    return hamming_weight(ne(eq(alpha, beta, gamma, params.mask), params.mask) & mask)


def int_to_bits(value: int, word_size: int) -> List[int]:
    return [(value >> i) & 1 for i in range(word_size - 1, -1, -1)]


def aop(x: int, params: SpeckParams) -> int:
    log2n = int(math.log2(params.word_size))
    rx = [0] * log2n
    ry = [0] * log2n
    rx[0] = x & (x >> 1)
    for i in range(1, log2n - 1):
        rx[i] = rx[i - 1] & (rx[i - 1] >> (1 << i))
    ry[0] = x & ne(rx[0], params.mask)
    for i in range(1, log2n):
        ry[i] = ry[i - 1] | ((ry[i - 1] >> (1 << i)) & rx[i - 1])
    return ry[-1]


def cac_fixed(alpha: int, beta: int, params: SpeckParams) -> List[int]:
    """Fixed common alternation parity described in the paper and mirrored in the uploaded code."""
    tmp = ne(alpha ^ beta, params.mask)
    p_list = int_to_bits(aop(tmp & (tmp >> 1) & (alpha ^ (alpha >> 1)), params), params.word_size)
    alpha_bits = int_to_bits(alpha, params.word_size)
    beta_bits = int_to_bits(beta, params.word_size)
    for i in range(params.word_size - 1, 0, -1):
        j = params.word_size - i
        if (
            alpha_bits[j - 1] == beta_bits[j - 1]
            and alpha_bits[j] == beta_bits[j]
            and alpha_bits[j - 1] != alpha_bits[j]
        ):
            p_list[j - 1] = 0
        else:
            break
    return p_list


@dataclass
class GammaGraphNode:
    lsb: int = -1
    successors: List[List[bool]] = field(default_factory=lambda: [[False, False], [False, False]])


def delta_optimal_graphs(alpha: int, beta: int, delta: int, params: SpeckParams) -> List[List[GammaGraphNode]]:
    graphs: List[List[GammaGraphNode]] = []
    alpha_bits = int_to_bits(alpha, params.word_size)
    beta_bits = int_to_bits(beta, params.word_size)
    p_list = cac_fixed(alpha, beta, params)

    possible_c_positions: List[int] = []
    for i in range(1, params.word_size):
        idx = params.word_size - 1 - i
        if alpha_bits[idx] == beta_bits[idx]:
            possible_c_positions.append(idx)

    positions_lists: List[List[int]] = [[]]
    if delta > 0:
        import itertools

        positions_lists = []
        for d in range(delta + 1):
            for item in itertools.combinations(possible_c_positions, d):
                positions_lists.append(list(item))

    for positions in positions_lists:
        graph = [GammaGraphNode() for _ in range(params.word_size)]
        for node in graph:
            node.lsb = alpha_bits[params.word_size - 1] ^ beta_bits[params.word_size - 1]
        for i in range(1, params.word_size):
            idx = params.word_size - 1 - i
            for prev_bit in (0, 1):
                reachable = False
                if idx == params.word_size - 2 and graph[idx].lsb == prev_bit:
                    reachable = True
                elif idx <= params.word_size - 3 and (
                    graph[idx + 2].successors[0][prev_bit] or graph[idx + 2].successors[1][prev_bit]
                ):
                    reachable = True

                if not reachable:
                    continue

                if alpha_bits[idx + 1] == beta_bits[idx + 1] and alpha_bits[idx + 1] == prev_bit:
                    graph[idx + 1].successors[prev_bit][
                        alpha_bits[idx] ^ beta_bits[idx] ^ beta_bits[idx + 1]
                    ] = True
                elif idx == 0 or alpha_bits[idx] != beta_bits[idx] or p_list[idx] == 1:
                    graph[idx + 1].successors[prev_bit] = [True, True]
                else:
                    if idx in positions:
                        graph[idx + 1].successors[prev_bit][1 - alpha_bits[idx]] = True
                    else:
                        graph[idx + 1].successors[prev_bit][alpha_bits[idx]] = True
        graphs.append(graph)
    return graphs


def _path_weight_bits(bits: Sequence[int]) -> int:
    return sum(1 for bit in bits if bit)


def enumerate_gamma_from_graphs(
    graphs: Sequence[Sequence[GammaGraphNode]], params: SpeckParams, gamma_hw_limit: int
) -> List[int]:
    def dfs(graph: Sequence[GammaGraphNode], step: int, path: List[int], out: List[List[int]]) -> None:
        if step == params.word_size - 1:
            final_path = list(reversed(path))
            if _path_weight_bits(final_path) <= gamma_hw_limit:
                out.append(final_path)
            return
        node_idx = params.word_size - 1 - step
        last = path[-1]
        for nxt in (0, 1):
            if graph[node_idx].successors[last][nxt]:
                path.append(nxt)
                dfs(graph, step + 1, path, out)
                path.pop()

    gamma_values: set[int] = set()
    for graph in graphs:
        paths: List[List[int]] = []
        for start in (0, 1):
            dfs(graph, 0, [start], paths)
        for bits in paths:
            value = 0
            for i, bit in enumerate(bits):
                value |= bit << (params.word_size - 1 - i)
            gamma_values.add(value)
    return list(gamma_values)


# -----------------------------
# Seed generation for start-in-the-middle
# -----------------------------


def generate_low_weight_seed_states(params: SpeckParams, max_transition_weight: int = 1) -> List[State]:
    """
    Generate start-in-the-middle seed states.

    We recursively enumerate all one-round addition transitions whose weight is
    <= max_transition_weight, then map the modular-addition inputs back to SPECK
    state differences (ΔL, ΔR), where α = ROR(ΔL, alpha) and β = ΔR.
    """

    seeds: set[State] = set()

    def rec(i: int, alpha_bits: List[int], beta_bits: List[int], gamma_bits: List[int], cost: int) -> None:
        if i == params.word_size - 1:
            alpha = sum(bit << j for j, bit in enumerate(alpha_bits))
            beta = sum(bit << j for j, bit in enumerate(beta_bits))
            if alpha == 0 and beta == 0:
                return
            delta_l = rol(alpha, params.alpha, params.word_size)
            delta_r = beta
            seeds.add((delta_l, delta_r))
            return

        all_equal = alpha_bits[i] == beta_bits[i] == gamma_bits[i]
        next_cost = cost + (0 if all_equal else 1)
        if next_cost > max_transition_weight:
            return

        for alpha_next in (0, 1):
            for beta_next in (0, 1):
                if all_equal:
                    gamma_next = alpha_next ^ beta_next ^ beta_bits[i]
                    rec(
                        i + 1,
                        alpha_bits + [alpha_next],
                        beta_bits + [beta_next],
                        gamma_bits + [gamma_next],
                        next_cost,
                    )
                else:
                    for gamma_next in (0, 1):
                        rec(
                            i + 1,
                            alpha_bits + [alpha_next],
                            beta_bits + [beta_next],
                            gamma_bits + [gamma_next],
                            next_cost,
                        )

    for alpha0 in (0, 1):
        for beta0 in (0, 1):
            gamma0 = alpha0 ^ beta0
            rec(0, [alpha0], [beta0], [gamma0], 0)

    return sorted(seeds)


def resolve_seed_states(cfg: SearchConfig, params: SpeckParams) -> List[State]:
    """Build the seed-state pool used at the split point."""
    if cfg.seed_states_override:
        seen: set[State] = set()
        out: List[State] = []
        for state in cfg.seed_states_override:
            normalized = (int(state[0]), int(state[1]))
            if normalized not in seen:
                seen.add(normalized)
                out.append(normalized)
        if not out:
            raise RuntimeError("seed_states_override was provided but empty after deduplication.")
        return sorted(out)

    paper_seeds = generate_low_weight_seed_states(params, cfg.seed_transition_weight)
    if cfg.seed_mode == "paper_low_weight":
        out = list(paper_seeds)
    elif cfg.seed_mode == "fixed":
        if cfg.initial_difference is None:
            raise RuntimeError("seed_mode='fixed' requires initial_difference to be set.")
        out = [cfg.initial_difference]
    else:
        out = list(paper_seeds)
        if cfg.initial_difference is not None and cfg.initial_difference not in set(out):
            out.append(cfg.initial_difference)

    if not out:
        raise RuntimeError("No seed states were generated.")
    return sorted(out)


def deadline_reached(cfg: SearchConfig) -> bool:
    return cfg.deadline_ts is not None and time.time() >= cfg.deadline_ts


def stop_requested(cfg: SearchConfig) -> bool:
    if deadline_reached(cfg):
        return True
    return cfg.stop_flag_file is not None and os.path.exists(cfg.stop_flag_file)


def signal_target_hit(cfg: SearchConfig) -> None:
    if cfg.stop_flag_file is None:
        return
    try:
        os.makedirs(os.path.dirname(cfg.stop_flag_file) or ".", exist_ok=True)
        with open(cfg.stop_flag_file, "w", encoding="utf-8") as f:
            f.write("target_hit\n")
    except OSError:
        pass


# -----------------------------
# Search state / helpers
# -----------------------------


@dataclass
class TreeNode:
    visits: int = 0
    children: List[State] = field(default_factory=list)
    payouts: List[float] = field(default_factory=list)


@dataclass
class CacheEntry:
    path: List[State] = field(default_factory=list)
    path_weights: List[int] = field(default_factory=list)
    best_weight: float = math.inf


class SearchStateStore:
    def __init__(self, seed_states: Sequence[State]):
        self.tree: Dict[State, TreeNode] = {state: TreeNode() for state in seed_states}
        self.cache: Dict[State, CacheEntry] = {state: CacheEntry() for state in seed_states}
        self.scores: Dict[ScoreKey, List[float]] = {(0, state): [0.0, 0.0, 0.0] for state in seed_states}

    def ensure_node(self, round_idx: int, state: State) -> None:
        if state not in self.tree:
            self.tree[state] = TreeNode()
        if (round_idx, state) not in self.scores:
            self.scores[(round_idx, state)] = [0.0, 0.0, 0.0]


@dataclass
class Characteristic:
    split: int
    total_weight: int
    full_path: List[State]
    full_path_weights: List[int]
    seed_state: State
    forward_suffix: List[State] = field(default_factory=list)
    backward_prefix_reversed: List[State] = field(default_factory=list)


# -----------------------------
# Forward / backward expansion
# -----------------------------


def generate_possible_forward_children(state: State, cfg: SearchConfig, params: SpeckParams) -> List[State]:
    delta_l, delta_r = state
    x = ror(delta_l, params.alpha, params.word_size)
    y = delta_r
    if hamming_weight(delta_l) > cfg.t1 or hamming_weight(delta_r) > cfg.t1:
        return []
    graphs = delta_optimal_graphs(x, y, cfg.delta, params)
    candidates = enumerate_gamma_from_graphs(graphs, params, cfg.t1)
    out: List[State] = []
    seen: set[State] = set()
    for gamma in candidates:
        gamma_weight = hamming_weight(gamma)
        if hamming_weight(delta_l) + hamming_weight(delta_r) + gamma_weight > cfg.t2:
            continue
        w = xdp_add_weight(x, y, gamma, params)
        if w is None or w > cfg.expand_threshold_weight:
            continue
        child = (gamma, gamma ^ rol(delta_r, params.beta_rot, params.word_size))
        if child not in seen:
            seen.add(child)
            out.append(child)
    return out


def generate_possible_backward_children(state: State, cfg: SearchConfig, params: SpeckParams) -> List[State]:
    curr_l, curr_r = state
    prev_r = ror(curr_l ^ curr_r, params.beta_rot, params.word_size)
    if hamming_weight(curr_l) > cfg.t1 or hamming_weight(prev_r) > cfg.t1:
        return []

    # Reverse round: ror(prev_l, alpha) = curr_l - prev_r.
    # The paper notes xdp+(a,b,c) = xdp-(a,b,c), so we can reuse the same
    # generalized Lipmaa-Moriai machinery on inputs (curr_l, prev_r).
    graphs = delta_optimal_graphs(curr_l, prev_r, cfg.delta, params)
    alpha_prev_values = enumerate_gamma_from_graphs(graphs, params, cfg.t1)
    out: List[State] = []
    seen: set[State] = set()
    for alpha_prev in alpha_prev_values:
        alpha_prev_hw = hamming_weight(alpha_prev)
        if hamming_weight(curr_l) + hamming_weight(prev_r) + alpha_prev_hw > cfg.t2:
            continue
        w = xdp_add_weight(curr_l, prev_r, alpha_prev, params)
        if w is None or w > cfg.expand_threshold_weight:
            continue
        prev_l = rol(alpha_prev, params.alpha, params.word_size)
        child = (prev_l, prev_r)
        if child not in seen:
            seen.add(child)
            out.append(child)
    return out


def transition_weight(direction: str, parent: State, child: State, params: SpeckParams) -> Optional[int]:
    if direction == "forward":
        parent_l, parent_r = parent
        child_l, _ = child
        return xdp_add_weight(ror(parent_l, params.alpha, params.word_size), parent_r, child_l, params)
    if direction == "backward":
        curr_l, curr_r = parent
        prev_l, prev_r = child
        alpha_prev = ror(prev_l, params.alpha, params.word_size)
        assert prev_r == ror(curr_l ^ curr_r, params.beta_rot, params.word_size)
        return xdp_add_weight(curr_l, prev_r, alpha_prev, params)
    raise ValueError(direction)


# -----------------------------
# MCTS core
# -----------------------------


def compute_uct(round_idx: int, parent: State, child: State, store: SearchStateStore, cfg: SearchConfig) -> float:
    child_scores = store.scores.get((round_idx + 1, child))
    parent_scores = store.scores.get((round_idx, parent))
    if not child_scores or child_scores[0] == 0:
        return float("inf")
    if not parent_scores or parent_scores[0] == 0:
        return float("inf")
    term1 = child_scores[1]
    term2 = cfg.uct_c * math.sqrt(math.log(parent_scores[0]) / child_scores[0])
    term3 = math.sqrt((child_scores[2] - child_scores[0] * child_scores[1] ** 2 + cfg.uct_d) / child_scores[0])
    return term1 + term2 + term3


def sample_geometric_index(length: int, p: float, rng: random.Random) -> int:
    if length <= 1:
        return 0
    u = rng.random()
    idx = int(math.log(1 - u) / math.log(1 - p)) if p < 1 else 0
    if idx >= length:
        idx = length - 1
    return idx


def choose_seed_state(
    seed_states: Sequence[State],
    store: SearchStateStore,
    global_iteration: int,
    cfg: SearchConfig,
    rng: random.Random,
) -> State:
    if global_iteration < cfg.k:
        return seed_states[rng.randrange(len(seed_states))]
    ordered = sorted(seed_states, key=lambda st: store.scores.get((0, st), [0.0, 0.0, 0.0])[1], reverse=True)
    idx = sample_geometric_index(len(ordered), cfg.geom_p, rng)
    return ordered[idx]


def expand_children(direction: str, state: State, cfg: SearchConfig, params: SpeckParams) -> List[State]:
    if direction == "forward":
        return generate_possible_forward_children(state, cfg, params)
    if direction == "backward":
        return generate_possible_backward_children(state, cfg, params)
    raise ValueError(direction)


def mcts_iteration(
    direction: str,
    start_state: State,
    num_rounds: int,
    store: SearchStateStore,
    cfg: SearchConfig,
    params: SpeckParams,
    rng: random.Random,
) -> Tuple[List[State], List[int], int, int, int]:
    current = start_state
    path: List[State] = [current]
    path_weights: List[int] = []
    expanded_nodes = 0
    expanded_children = 0
    store.ensure_node(0, current)
    store.tree[current].visits += 1

    for round_idx in range(1, num_rounds + 1):
        node = store.tree[current]
        if node.children:
            if node.visits <= cfg.k:
                next_state = node.children[rng.randrange(len(node.children))]
            else:
                next_state = max(
                    node.children,
                    key=lambda child: compute_uct(round_idx - 1, current, child, store, cfg),
                )
        else:
            children = expand_children(direction, current, cfg, params)
            expanded_nodes += 1
            expanded_children += len(children)
            for child in children:
                if child not in node.children:
                    node.children.append(child)
                    store.ensure_node(round_idx, child)
            if not node.children:
                break
            next_state = node.children[rng.randrange(len(node.children))]

        w = transition_weight(direction, current, next_state, params)
        if w is None:
            break

        path.append(next_state)
        path_weights.append(w)
        store.tree[next_state].visits += 1
        current = next_state

    total_weight = sum(path_weights)
    for i, state in enumerate(path):
        tail_weight = sum(path_weights[i:])
        local_norm = (num_rounds - i) / num_rounds if num_rounds > 0 else 1.0
        local_score = (local_norm / tail_weight) if tail_weight > 0 else 1.0
        global_score = (1.0 / total_weight) if total_weight > 0 else 1.0
        payout = cfg.beta_mix * global_score + (1 - cfg.beta_mix) * local_score
        store.tree[state].payouts.append(payout)
        score_entry = store.scores.setdefault((i, state), [0.0, 0.0, 0.0])
        score_entry[0] = float(store.tree[state].visits)
        score_entry[1] = ((score_entry[0] - 1.0) * score_entry[1] + payout) / score_entry[0]
        score_entry[2] += payout ** 2

    return path, path_weights, total_weight, expanded_nodes, expanded_children


# -----------------------------
# Search orchestration
# -----------------------------


def combine_paths(
    split: int,
    seed_state: State,
    backward_path: List[State],
    backward_weights: List[int],
    forward_path: List[State],
    forward_weights: List[int],
) -> Characteristic:
    chronological_prefix = list(reversed(backward_path))
    chronological_prefix_weights = list(reversed(backward_weights))
    full_path = chronological_prefix + forward_path[1:]
    full_weights = chronological_prefix_weights + forward_weights
    return Characteristic(
        split=split,
        total_weight=sum(full_weights),
        full_path=full_path,
        full_path_weights=full_weights,
        seed_state=seed_state,
        forward_suffix=forward_path,
        backward_prefix_reversed=backward_path,
    )


def run_single_split(
    cfg: SearchConfig,
    split: int,
    show_progress: bool = False,
    log_interval: int = 5000,
) -> Optional[Characteristic]:
    cfg = cfg.finalize()
    params = SpeckParams.for_block_size(cfg.block_size)
    rng = random.Random(cfg.rng_seed + split)

    seed_states = resolve_seed_states(cfg, params)

    forward_rounds = cfg.total_rounds - split
    backward_rounds = split

    forward_store = SearchStateStore(seed_states)
    fw_total_expanded_nodes = 0
    fw_total_expanded_children = 0
    fw_last_percent = -1
    for it in range(cfg.forward_iterations):
        if stop_requested(cfg):
            break
        fw_last_percent = _update_progress_every_percent(
            f"[split={split}] forward ", it, cfg.forward_iterations, fw_last_percent, show_progress
        )
        seed = choose_seed_state(seed_states, forward_store, it, cfg, rng)
        path, weights, total_weight, exp_nodes, exp_children = mcts_iteration(
            "forward", seed, forward_rounds, forward_store, cfg, params, rng
        )
        fw_total_expanded_nodes += exp_nodes
        fw_total_expanded_children += exp_children
        if len(path) == forward_rounds + 1 and total_weight < forward_store.cache[seed].best_weight:
            forward_store.cache[seed].path = path
            forward_store.cache[seed].path_weights = weights
            forward_store.cache[seed].best_weight = total_weight
        if show_progress and ((it + 1) % log_interval == 0 or it + 1 == cfg.forward_iterations):
            print(
                f"[split={split}] forward iter={it + 1}/{cfg.forward_iterations} "
                f"depth={len(path) - 1}/{forward_rounds} "
                f"exp_nodes(last/total)={exp_nodes}/{fw_total_expanded_nodes} "
                f"exp_children(last/total)={exp_children}/{fw_total_expanded_children}",
                flush=True,
            )

    backward_store = SearchStateStore(seed_states)
    best: Optional[Characteristic] = None
    max_iterations = cfg.backward_iterations if cfg.backward_iterations is not None else 10**18
    bw_total_expanded_nodes = 0
    bw_total_expanded_children = 0

    bw_last_percent = -1
    for it in range(max_iterations):
        if stop_requested(cfg):
            break
        bw_last_percent = _update_progress_every_percent(
            f"[split={split}] backward", it, max_iterations, bw_last_percent, show_progress
        )
        seed = choose_seed_state(seed_states, backward_store, it, cfg, rng)
        if math.isinf(forward_store.cache[seed].best_weight):
            continue
        path, weights, total_weight, exp_nodes, exp_children = mcts_iteration(
            "backward", seed, backward_rounds, backward_store, cfg, params, rng
        )
        bw_total_expanded_nodes += exp_nodes
        bw_total_expanded_children += exp_children
        if len(path) != backward_rounds + 1:
            continue
        candidate = combine_paths(
            split,
            seed,
            path,
            weights,
            forward_store.cache[seed].path,
            forward_store.cache[seed].path_weights,
        )
        if best is None or candidate.total_weight < best.total_weight:
            best = candidate
            if cfg.target_weight is not None and best.total_weight <= cfg.target_weight:
                break
        if show_progress and ((it + 1) % log_interval == 0 or it + 1 == max_iterations):
            print(
                f"[split={split}] backward iter={it + 1}/{max_iterations} "
                f"depth={len(path) - 1}/{backward_rounds} "
                f"exp_nodes(last/total)={exp_nodes}/{bw_total_expanded_nodes} "
                f"exp_children(last/total)={exp_children}/{bw_total_expanded_children} "
                f"{_fmt_best(best)}",
                flush=True,
            )

    return best


def _run_single_split_worker(args: Tuple[SearchConfig, int, int]) -> Optional[Characteristic]:
    cfg, split, log_interval = args
    return run_single_split(cfg, split, show_progress=False, log_interval=log_interval)


def run_parallel_splits(
    cfg: SearchConfig, split_values: Sequence[int], processes: int, log_interval: int = 5000
) -> List[Optional[Characteristic]]:
    if processes <= 1 or len(split_values) <= 1:
        results: List[Optional[Characteristic]] = []
        for i, split in enumerate(split_values):
            _render_progress("split progress ", i, len(split_values))
            results.append(run_single_split(cfg, split, show_progress=True, log_interval=log_interval))
            _render_progress("split progress ", i + 1, len(split_values))
        return results
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(processes, len(split_values))) as pool:
        work_items = [(cfg, split, log_interval) for split in split_values]
        unordered_results: List[Optional[Characteristic]] = []
        for i, res in enumerate(pool.imap_unordered(_run_single_split_worker, work_items)):
            unordered_results.append(res)
            _render_progress("split progress ", i + 1, len(split_values))
        return unordered_results


def _run_single_direction_worker(args: Tuple[SearchConfig, str, int, int]) -> Optional[Characteristic]:
    cfg, direction, log_interval, restart_idx = args
    local_cfg = replace(cfg, rng_seed=cfg.rng_seed + 1_000_003 * restart_idx)
    return run_single_direction_search(local_cfg, direction, show_progress=False, log_interval=log_interval)


def run_parallel_restarts(
    cfg: SearchConfig,
    direction: str,
    processes: int,
    restarts: int,
    log_interval: int = 5000,
) -> Optional[Characteristic]:
    if restarts <= 1 or processes <= 1:
        best: Optional[Characteristic] = None
        for i in range(restarts):
            _render_progress("restart progress", i, restarts)
            candidate = _run_single_direction_worker((cfg, direction, log_interval, i))
            if candidate is not None and (best is None or candidate.total_weight < best.total_weight):
                best = candidate
            _render_progress("restart progress", i + 1, restarts)
        return best

    ctx = mp.get_context("spawn")
    work_items = [(cfg, direction, log_interval, i) for i in range(restarts)]
    best: Optional[Characteristic] = None
    with ctx.Pool(processes=min(processes, restarts)) as pool:
        for i, candidate in enumerate(pool.imap_unordered(_run_single_direction_worker, work_items)):
            if candidate is not None and (best is None or candidate.total_weight < best.total_weight):
                best = candidate
            _render_progress("restart progress", i + 1, restarts)
    return best


# -----------------------------
# Pretty printing / CLI
# -----------------------------


def fmt_state(state: State, params: SpeckParams) -> str:
    width = params.word_size // 4
    return f"({state[0]:0{width}x}, {state[1]:0{width}x})"


def print_characteristic(ch: Characteristic, cfg: SearchConfig) -> None:
    params = SpeckParams.for_block_size(cfg.block_size)
    print(f"best split s = {ch.split}")
    print(f"total weight = {ch.total_weight}")
    print(f"differential probability = 2^(-{ch.total_weight})")
    print(f"seed state   = {fmt_state(ch.seed_state, params)}")
    print("characteristic:")
    for i, state in enumerate(ch.full_path):
        if i == 0:
            print(f"  r={i:2d}  {fmt_state(state, params)}")
        else:
            print(f"  r={i:2d}  {fmt_state(state, params)}   w={ch.full_path_weights[i-1]}")


def probability_to_weight(max_diff_probability: Optional[float]) -> Optional[int]:
    if max_diff_probability is None:
        return None
    if not (0.0 < max_diff_probability <= 1.0):
        raise ValueError("MAX_DIFF_PROBABILITY must be in (0, 1].")
    return int(math.ceil(-math.log2(max_diff_probability)))


def run_single_direction_search(
    cfg: SearchConfig, direction: str, show_progress: bool = True, log_interval: int = 5000
) -> Optional[Characteristic]:
    cfg = cfg.finalize()
    params = SpeckParams.for_block_size(cfg.block_size)
    rng = random.Random(cfg.rng_seed)
    seed_states = resolve_seed_states(cfg, params)

    store = SearchStateStore(seed_states)
    best: Optional[Characteristic] = None
    iterations = cfg.forward_iterations if direction == "forward" else cfg.backward_iterations
    num_rounds = cfg.total_rounds
    total_expanded_nodes = 0
    total_expanded_children = 0

    last_percent = -1
    for it in range(iterations):
        if stop_requested(cfg):
            break
        last_percent = _update_progress_every_percent(
            f"[{direction}] iterations", it, iterations, last_percent, show_progress
        )
        seed = choose_seed_state(seed_states, store, it, cfg, rng)
        path, weights, total_weight, exp_nodes, exp_children = mcts_iteration(
            direction, seed, num_rounds, store, cfg, params, rng
        )
        total_expanded_nodes += exp_nodes
        total_expanded_children += exp_children
        if len(path) != num_rounds + 1:
            if show_progress and ((it + 1) % log_interval == 0 or it + 1 == iterations):
                print(
                    f"[{direction}] iter={it + 1}/{iterations} "
                    f"depth={len(path) - 1}/{num_rounds} "
                    f"exp_nodes(last/total)={exp_nodes}/{total_expanded_nodes} "
                    f"exp_children(last/total)={exp_children}/{total_expanded_children} "
                    f"{_fmt_best(best)}",
                    flush=True,
                )
            continue
        candidate = Characteristic(
            split=0 if direction == "forward" else cfg.total_rounds,
            total_weight=total_weight,
            full_path=path if direction == "forward" else list(reversed(path)),
            full_path_weights=weights if direction == "forward" else list(reversed(weights)),
            seed_state=seed,
            forward_suffix=path if direction == "forward" else [],
            backward_prefix_reversed=path if direction == "backward" else [],
        )
        if best is None or candidate.total_weight < best.total_weight:
            best = candidate
            if cfg.target_weight is not None and best.total_weight <= cfg.target_weight:
                break
        if show_progress and ((it + 1) % log_interval == 0 or it + 1 == iterations):
            print(
                f"[{direction}] iter={it + 1}/{iterations} "
                f"depth={len(path) - 1}/{num_rounds} "
                f"exp_nodes(last/total)={exp_nodes}/{total_expanded_nodes} "
                f"exp_children(last/total)={exp_children}/{total_expanded_children} "
                f"{_fmt_best(best)}",
                flush=True,
            )
    return best


def main() -> None:
    # ========== Directly editable settings ==========
    CIPHER_BLOCK_SIZE = 48  # 32 / 48 / 64 / 96 / 128
    SEARCH_ROUNDS = 9

    # Bellini et al. mainly describe start-in-the-middle over all split values,
    # but you can keep forward/backward-only runs for unified fixed-input comparisons.
    SEARCH_DIRECTION = "encrypt"  # "encrypt" / "decrypt" / "bidirectional"

    # Seed handling:
    #   "paper_low_weight" -> paper-style low-weight split states
    #   "fixed"            -> use INITIAL_DIFFERENCE only
    #   "mixed"            -> union of the two
    SEED_MODE = "fixed"
    INITIAL_DIFFERENCE = (0x0, 0x020000)
    SEED_STATES_OVERRIDE = None

    # The paper uses 1e5 forward iterations to build the cache and then lets the
    # backward stage continue until a satisfactory characteristic is found.
    FORWARD_ITERATIONS = 2_000
    BACKWARD_ITERATIONS = 2_000  # set to None only for bidirectional target-driven runs
    TARGET_WEIGHT = probability_to_weight(2 ** (-41))  # or None
    MAX_WALL_TIME_SECONDS = 3600  # None to disable the one-hour cap

    # Parallelism:
    #   bidirectional -> parallel over split values s
    #   encrypt/decrypt -> parallel over independent restarts
    PARALLEL_PROCESSES = 32
    PARALLEL_RESTARTS = 32

    LOG_INTERVAL = 2000
    SAVE_RESULT_FILE = "./MCTS/mcts_last_result.txt"
    # ===============================================

    search_direction = SEARCH_DIRECTION.lower()
    if search_direction not in ("encrypt", "decrypt", "bidirectional"):
        raise ValueError("SEARCH_DIRECTION must be one of: encrypt, decrypt, bidirectional")
    if PARALLEL_PROCESSES < 1:
        raise ValueError("PARALLEL_PROCESSES must be >= 1")
    if PARALLEL_RESTARTS < 1:
        raise ValueError("PARALLEL_RESTARTS must be >= 1")

    start = time.time()
    deadline_ts = None if MAX_WALL_TIME_SECONDS is None else start + MAX_WALL_TIME_SECONDS

    cfg = SearchConfig(
        block_size=CIPHER_BLOCK_SIZE,
        total_rounds=SEARCH_ROUNDS,
        forward_iterations=FORWARD_ITERATIONS,
        backward_iterations=BACKWARD_ITERATIONS,
        target_weight=TARGET_WEIGHT,
        rng_seed=0,
        initial_difference=INITIAL_DIFFERENCE,
        seed_mode=SEED_MODE,
        seed_states_override=SEED_STATES_OVERRIDE,
        deadline_ts=deadline_ts,
    ).finalize()

    if search_direction == "bidirectional":
        split_values = list(range(cfg.total_rounds + 1))
        results = run_parallel_splits(cfg, split_values, processes=PARALLEL_PROCESSES, log_interval=LOG_INTERVAL)
        valid = [r for r in results if r is not None]
        best = min(valid, key=lambda item: item.total_weight) if valid else None
    elif search_direction == "encrypt":
        best = run_parallel_restarts(cfg, "forward", PARALLEL_PROCESSES, PARALLEL_RESTARTS, LOG_INTERVAL)
    else:
        best = run_parallel_restarts(cfg, "backward", PARALLEL_PROCESSES, PARALLEL_RESTARTS, LOG_INTERVAL)
    elapsed = time.time() - start

    if best is None:
        print("No complete characteristic was found.")
        return

    print_characteristic(best, cfg)
    print(f"search direction = {search_direction}")
    print(f"seed mode = {cfg.seed_mode}")
    print(f"elapsed = {elapsed:.3f}s")
    if deadline_ts is not None and elapsed >= MAX_WALL_TIME_SECONDS:
        print("time limit reached before hitting the target; returning current best result")

    os.makedirs(os.path.dirname(SAVE_RESULT_FILE) or ".", exist_ok=True)
    with open(SAVE_RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(f"block_size = {CIPHER_BLOCK_SIZE}\n")
        f.write(f"rounds = {SEARCH_ROUNDS}\n")
        f.write(f"forward_iterations = {FORWARD_ITERATIONS}\n")
        f.write(f"backward_iterations = {BACKWARD_ITERATIONS}\n")
        f.write(f"search_direction = {search_direction}\n")
        f.write(f"seed_mode = {SEED_MODE}\n")
        f.write(f"initial_difference = {INITIAL_DIFFERENCE}\n")
        f.write(f"seed_states_override = {SEED_STATES_OVERRIDE}\n")
        f.write(f"target_weight = {TARGET_WEIGHT}\n")
        f.write(f"max_wall_time_seconds = {MAX_WALL_TIME_SECONDS}\n")
        f.write(f"parallel_processes = {PARALLEL_PROCESSES}\n")
        f.write(f"parallel_restarts = {PARALLEL_RESTARTS}\n")
        f.write(f"log_interval = {LOG_INTERVAL}\n")
        f.write(f"total_weight = {best.total_weight}\n")
        f.write(f"differential_probability = 2^(-{best.total_weight})\n")
        f.write(f"elapsed_seconds = {elapsed:.6f}\n")


if __name__ == "__main__":
    main()

