import contextlib
import importlib.util
import io
import os
import sys
import time
from typing import Optional, Tuple


# ========== Directly editable settings ==========
CORE_MODULE_FILENAME = "mcts_paper_aligned_timecap_stopflag.py"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_TXT = os.path.join(PROJECT_ROOT, "MCTS", "speck96_r11_forward_batch_results.txt")
STOP_FLAG_DIR = os.path.join(PROJECT_ROOT, "MCTS", ".stop_flags")

BLOCK_SIZE = 96
TOTAL_ROUNDS = 11
SEARCH_DIRECTION = "forward"
SEED_MODE = "fixed"
TARGET_WEIGHT = 64
MAX_WALL_TIME_SECONDS = 3600
PARALLEL_PROCESSES = 32
PARALLEL_RESTARTS = 32
FORWARD_ITERATIONS = 10**18   # rely on wall-clock cap / target hit to stop
BACKWARD_ITERATIONS = 0       # unused for forward search
LOG_INTERVAL = 2000
RNG_SEED = 0

BIT_POSITIONS = [12, 14, 16, 18, 20, 21, 23, 25, 27, 29, 31, 33, 36, 38]
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
    params = mod.SpeckParams.for_block_size(BLOCK_SIZE)
    word_hex_digits = (params.word_size + 3) // 4

    header = [
        "=== SPECK96 11-round forward batch search ===",
        f"block_size = {BLOCK_SIZE}",
        f"rounds = {TOTAL_ROUNDS}",
        f"search_direction = {SEARCH_DIRECTION}",
        f"seed_mode = {SEED_MODE}",
        f"target_weight = {TARGET_WEIGHT}",
        f"max_wall_time_seconds = {MAX_WALL_TIME_SECONDS}",
        f"parallel_processes = {PARALLEL_PROCESSES}",
        f"parallel_restarts = {PARALLEL_RESTARTS}",
        f"forward_iterations = {FORWARD_ITERATIONS}",
        f"log_interval = {LOG_INTERVAL}",
        f"bit_positions = {BIT_POSITIONS}",
        "",
    ]
    ensure_parent_dir(OUTPUT_TXT)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(header))

    total_batch_start = time.time()

    for idx, bit_pos in enumerate(BIT_POSITIONS, start=1):
        diff_l = 1 << bit_pos
        diff_r = 0x0
        init_diff = (diff_l, diff_r)
        experiment_start = time.time()
        deadline_ts = experiment_start + MAX_WALL_TIME_SECONDS if MAX_WALL_TIME_SECONDS is not None else None
        stop_flag_file = os.path.join(STOP_FLAG_DIR, f"speck96_r11_bit{bit_pos}.flag")
        remove_file_if_exists(stop_flag_file)

        print(f"[{idx}/{len(BIT_POSITIONS)}] bit={bit_pos}, init_diff={format_state_hex(init_diff, word_hex_digits)}")

        cfg = mod.SearchConfig(
            block_size=BLOCK_SIZE,
            total_rounds=TOTAL_ROUNDS,
            forward_iterations=FORWARD_ITERATIONS,
            backward_iterations=BACKWARD_ITERATIONS,
            target_weight=TARGET_WEIGHT,
            rng_seed=RNG_SEED,
            initial_difference=init_diff,
            seed_mode=SEED_MODE,
            deadline_ts=deadline_ts,
            stop_flag_file=stop_flag_file,
        ).finalize()

        best = mod.run_parallel_restarts(
            cfg,
            SEARCH_DIRECTION,
            processes=PARALLEL_PROCESSES,
            restarts=PARALLEL_RESTARTS,
            log_interval=LOG_INTERVAL,
        )
        elapsed = time.time() - experiment_start

        if best is None:
            status = "no_complete_characteristic"
            body = [
                f"=== bit {bit_pos} ===",
                f"initial_difference = {format_state_hex(init_diff, word_hex_digits)}",
                f"status = {status}",
                f"elapsed_seconds = {elapsed:.6f}",
                "",
            ]
        else:
            status = "target_reached" if best.total_weight <= TARGET_WEIGHT else "time_limit_or_budget_exhausted"
            char_text = characteristic_to_text(mod, best, cfg)
            body = [
                f"=== bit {bit_pos} ===",
                f"initial_difference = {format_state_hex(init_diff, word_hex_digits)}",
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

