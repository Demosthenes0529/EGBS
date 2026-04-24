import contextlib
import io
import os
import sys
import time
from typing import Dict, List, Tuple


# ========== Directly editable settings ==========
CORE_MODULE_FILENAME = "mcts_paper_aligned_timecap_stopflag.py"
OUTPUT_TXT = "./MCTS/mcts_mixed_onehour_cases_results.txt"
STOP_FLAG_DIR = "./MCTS/.stop_flags_mixed_cases"

MAX_WALL_TIME_SECONDS = 3600
PARALLEL_PROCESSES = 32
PARALLEL_RESTARTS = 32
LOG_INTERVAL = 2000
RNG_SEED = 0
SEED_MODE = "fixed"
MAX_ITERATIONS = 10**18   # rely on wall-clock cap / target hit to stop

CASES: List[Dict] = [
    {
        "name": "speck32_backward_r5",
        "block_size": 32,
        "initial_difference": (0x0040, 0x0000),
        "direction": "backward",
        "rounds": 5,
        "target_weight": 23,
    },
    {
        "name": "speck32_forward_r4",
        "block_size": 32,
        "initial_difference": (0x0040, 0x0000),
        "direction": "forward",
        "rounds": 4,
        "target_weight": 7,
    },
    {
        "name": "speck48_forward_r9",
        "block_size": 48,
        "initial_difference": (0x000000, 0x020000),
        "direction": "forward",
        "rounds": 9,
        "target_weight": 41,
    },
    {
        "name": "speck48_backward_r2",
        "block_size": 48,
        "initial_difference": (0x000000, 0x020000),
        "direction": "backward",
        "rounds": 2,
        "target_weight": 4,
    },
    {
        "name": "speck64_forward_r12",
        "block_size": 64,
        "initial_difference": (0x08000000, 0x00000000),
        "direction": "forward",
        "rounds": 12,
        "target_weight": 50,
    },
    {
        "name": "speck64_backward_r3",
        "block_size": 64,
        "initial_difference": (0x08000000, 0x00000000),
        "direction": "backward",
        "rounds": 3,
        "target_weight": 12,
    },
    {
        "name": "speck96_forward_r13",
        "block_size": 96,
        "initial_difference": (0x000008000000, 0x000000000000),
        "direction": "forward",
        "rounds": 13,
        "target_weight": 75,
    },
    {
        "name": "speck96_backward_r3",
        "block_size": 96,
        "initial_difference": (0x000008000000, 0x000000000000),
        "direction": "backward",
        "rounds": 3,
        "target_weight": 12,
    },
]
# ===============================================


def load_core_module():
    module_dir = os.path.dirname(__file__)
    module_path = os.path.join(module_dir, CORE_MODULE_FILENAME)
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Core module not found: {module_path}")

    module_name, ext = os.path.splitext(os.path.basename(CORE_MODULE_FILENAME))
    if ext.lower() != ".py":
        raise ImportError(f"CORE_MODULE_FILENAME must point to a .py file, got: {CORE_MODULE_FILENAME}")

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    return __import__(module_name)


def format_state_hex(state: Tuple[int, int], word_hex_digits: int) -> str:
    return f"({state[0]:0{word_hex_digits}x}, {state[1]:0{word_hex_digits}x})"


def characteristic_to_text(mod, best, cfg) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.print_characteristic(best, cfg)
    return buf.getvalue().rstrip()


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def append_result(path: str, text: str) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
        f.flush()


def remove_file_if_exists(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def main() -> None:
    mod = load_core_module()

    header = [
        "=== MCTS mixed one-hour cases ===",
        f"max_wall_time_seconds = {MAX_WALL_TIME_SECONDS}",
        f"parallel_processes = {PARALLEL_PROCESSES}",
        f"parallel_restarts = {PARALLEL_RESTARTS}",
        f"log_interval = {LOG_INTERVAL}",
        f"seed_mode = {SEED_MODE}",
        f"max_iterations = {MAX_ITERATIONS}",
        "",
    ]
    ensure_parent_dir(OUTPUT_TXT)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(header))

    total_batch_start = time.time()

    for idx, case in enumerate(CASES, start=1):
        params = mod.SpeckParams.for_block_size(case["block_size"])
        word_hex_digits = (params.word_size + 3) // 4
        init_diff = case["initial_difference"]
        direction = case["direction"]
        experiment_start = time.time()
        deadline_ts = experiment_start + MAX_WALL_TIME_SECONDS if MAX_WALL_TIME_SECONDS is not None else None
        stop_flag_file = os.path.join(STOP_FLAG_DIR, f"{case['name']}.flag")
        remove_file_if_exists(stop_flag_file)

        print(
            f"[{idx}/{len(CASES)}] {case['name']}: "
            f"block={case['block_size']} rounds={case['rounds']} direction={direction} "
            f"init_diff={format_state_hex(init_diff, word_hex_digits)} target={case['target_weight']}"
        )

        cfg = mod.SearchConfig(
            block_size=case["block_size"],
            total_rounds=case["rounds"],
            forward_iterations=MAX_ITERATIONS if direction == "forward" else 0,
            backward_iterations=MAX_ITERATIONS if direction == "backward" else 0,
            target_weight=case["target_weight"],
            rng_seed=RNG_SEED,
            initial_difference=init_diff,
            seed_mode=SEED_MODE,
            deadline_ts=deadline_ts,
            stop_flag_file=stop_flag_file,
        ).finalize()

        best = mod.run_parallel_restarts(
            cfg,
            direction,
            processes=PARALLEL_PROCESSES,
            restarts=PARALLEL_RESTARTS,
            log_interval=LOG_INTERVAL,
        )
        elapsed = time.time() - experiment_start

        if best is None:
            status = "no_complete_characteristic"
            body = [
                f"=== {case['name']} ===",
                f"block_size = {case['block_size']}",
                f"rounds = {case['rounds']}",
                f"direction = {direction}",
                f"initial_difference = {format_state_hex(init_diff, word_hex_digits)}",
                f"target_weight = {case['target_weight']}",
                f"status = {status}",
                f"elapsed_seconds = {elapsed:.6f}",
                "",
            ]
        else:
            status = "target_reached" if best.total_weight <= case["target_weight"] else "time_limit_or_budget_exhausted"
            char_text = characteristic_to_text(mod, best, cfg)
            body = [
                f"=== {case['name']} ===",
                f"block_size = {case['block_size']}",
                f"rounds = {case['rounds']}",
                f"direction = {direction}",
                f"initial_difference = {format_state_hex(init_diff, word_hex_digits)}",
                f"target_weight = {case['target_weight']}",
                f"status = {status}",
                f"best_total_weight = {best.total_weight}",
                f"differential_probability = 2^(-{best.total_weight})",
                f"elapsed_seconds = {elapsed:.6f}",
                char_text,
                "",
            ]

        append_result(OUTPUT_TXT, "\n".join(body))
        print(f"  status={status}, elapsed={elapsed:.2f}s")
        remove_file_if_exists(stop_flag_file)

    total_batch_elapsed = time.time() - total_batch_start
    append_result(OUTPUT_TXT, f"total_batch_elapsed_seconds = {total_batch_elapsed:.6f}\n")
    print(f"All done. Results written to: {OUTPUT_TXT}")
    print(f"Total batch elapsed = {total_batch_elapsed:.2f}s")


if __name__ == "__main__":
    main()

