#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze round-wise diversity from SPECK96 result text files.

Supported filename styles:
- egbs:    speck96_result_egbs_14round_(diff_l,0).txt
           speck96_result_egbs_14round_diff_l.txt   # also tolerated
- ga:      speck96_result_ga_14round_diff_l.txt
- sp:      speck96_result_ea_14round_diff_l_single-parent.txt
- mp:      speck96_result_ea_14round_diff_l.txt

Output:
1) per_diff_round_metrics.csv
   columns:
   method, diff_l, file_name, total_paths, round, states_count,
   unique_states, unique_ratio,
   avg_pairwise_hd_raw, avg_pairwise_hd_norm,
   entropy_bits, eff_states, eff_ratio

2) aggregate_round_metrics.csv
   columns:
   method, round, n_diffs,
   <metric>_mean, <metric>_median, <metric>_q25, <metric>_q75, <metric>_min, <metric>_max

Usage:
python3 analyze_speck96_diversity.py \
    --input-dir ./saved_ea_txt/speck96 \
    --output-dir ./diversity_out \
    --round-tag 14round \
    --total-bits 96
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple, Iterable, Optional


# -----------------------------
# Config
# -----------------------------

def build_method_configs(round_tag: str):
    return {
        "egbs": {
            "globs": [
                f"speck96_result_egbs_{round_tag}_*.txt",
            ],
            "diff_regexes": [
                rf"speck96_result_egbs_{round_tag}_\((\d+),0\)\.txt$",
                rf"speck96_result_egbs_{round_tag}_(\d+)\.txt$",
            ],
        },
        "ga": {
            "globs": [
                f"speck96_result_ga_{round_tag}_*.txt",
            ],
            "diff_regexes": [
                rf"speck96_result_ga_{round_tag}_(\d+)\.txt$",
            ],
        },
        "sp": {
            "globs": [
                f"speck96_result_ea_{round_tag}_*_single-parent.txt",
            ],
            "diff_regexes": [
                rf"speck96_result_ea_{round_tag}_(\d+)_single-parent\.txt$",
            ],
        },
        "mp": {
            "globs": [
                f"speck96_result_ea_{round_tag}_*_5.txt",
            ],
            "diff_regexes": [
                rf"speck96_result_ea_{round_tag}_(\d+)_5\.txt$",
            ],
        },
    }


# -----------------------------
# Parsing
# -----------------------------

PATH_BLOCK_RE = re.compile(
    r"----\s*路径\s+\d+\s*\(总分:\s*([0-9.]+)\)\s*----(.*?)(?=\n----\s*路径\s+\d+\s*\(总分:|\Z)",
    re.S,
)

# score suffix is optional because your last layer line may omit it
LAYER_RE = re.compile(
    r"层\s*(\d+)\s*:\s*\(\s*(0x[0-9a-fA-F]+)\s*,\s*(0x[0-9a-fA-F]+)\s*\)"
    r"(?:\s*\|\s*分数:\s*([0-9.]+))?"
)

State = Tuple[int, int]
PathLayers = Dict[int, State]


def extract_diff_from_name(file_name: str, diff_regexes: List[str]) -> Optional[int]:
    for pat in diff_regexes:
        m = re.search(pat, file_name)
        if m:
            return int(m.group(1))
    return None


def scan_input_files(input_dir: Path, round_tag: str) -> Dict[str, Dict[int, Path]]:
    """
    Return:
        {
            "egbs": {diff_l: path, ...},
            "ga":   {diff_l: path, ...},
            ...
        }
    """
    configs = build_method_configs(round_tag)
    found: Dict[str, Dict[int, Path]] = {k: {} for k in configs.keys()}

    for method, cfg in configs.items():
        for glob_pat in cfg["globs"]:
            for path in sorted(input_dir.glob(glob_pat)):
                file_name = path.name

                # avoid SP files being double-counted as MP
                if method == "mp" and file_name.endswith("_single-parent.txt"):
                    continue

                diff_l = extract_diff_from_name(file_name, cfg["diff_regexes"])
                if diff_l is None:
                    continue

                found[method][diff_l] = path

    return found


def parse_result_file(path: Path) -> List[PathLayers]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    paths: List[PathLayers] = []

    for block_match in PATH_BLOCK_RE.finditer(text):
        block_text = block_match.group(2)
        layers: PathLayers = {}

        for m in LAYER_RE.finditer(block_text):
            layer = int(m.group(1))
            left = int(m.group(2), 16)
            right = int(m.group(3), 16)
            layers[layer] = (left, right)

        if layers:
            paths.append(layers)

    return paths


# -----------------------------
# Metrics
# -----------------------------

def quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def avg_pairwise_hamming(states: List[State], total_bits: int, half_bits: int) -> Tuple[float, float]:
    """
    Return:
        raw average pairwise Hamming distance in bits,
        normalized average pairwise Hamming distance in [0,1].
    """
    k = len(states)
    if k <= 1:
        return 0.0, 0.0

    combined = [(l << half_bits) | r for (l, r) in states]
    disagree_pairs = 0

    for bit in range(total_bits):
        ones = 0
        mask = 1 << bit
        for x in combined:
            if x & mask:
                ones += 1
        disagree_pairs += ones * (k - ones)

    raw = (2.0 * disagree_pairs) / (k * (k - 1))
    norm = raw / total_bits
    return raw, norm


def entropy_and_effective_states(states: List[State]) -> Tuple[float, float]:
    """
    Return:
        entropy in bits,
        effective number of states = 2^H
    """
    k = len(states)
    if k == 0:
        return 0.0, 0.0

    cnt = Counter(states)
    h = 0.0
    for c in cnt.values():
        p = c / k
        h -= p * math.log2(p)

    eff = 2.0 ** h
    return h, eff


def compute_metrics_for_paths(
    paths: List[PathLayers],
    total_bits: int,
) -> List[Dict[str, float]]:
    """
    Compute per-round metrics for one file (one method, one diff_l).
    """
    if not paths:
        return []

    half_bits = total_bits // 2
    all_rounds = sorted(set().union(*(p.keys() for p in paths)))

    rows = []
    for rnd in all_rounds:
        states = [p[rnd] for p in paths if rnd in p]
        k = len(states)

        unique_states = len(set(states))
        unique_ratio = unique_states / k if k else 0.0

        avg_hd_raw, avg_hd_norm = avg_pairwise_hamming(states, total_bits=total_bits, half_bits=half_bits)
        entropy_bits, eff_states = entropy_and_effective_states(states)
        eff_ratio = eff_states / k if k else 0.0

        rows.append({
            "round": rnd,
            "states_count": k,
            "unique_states": unique_states,
            "unique_ratio": unique_ratio,
            "avg_pairwise_hd_raw": avg_hd_raw,
            "avg_pairwise_hd_norm": avg_hd_norm,
            "entropy_bits": entropy_bits,
            "eff_states": eff_states,
            "eff_ratio": eff_ratio,
        })

    return rows


# -----------------------------
# CSV helpers
# -----------------------------

def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(per_rows: List[Dict]) -> List[Dict]:
    """
    Aggregate over diff_l for each method x round.
    """
    metrics = [
        "unique_states",
        "unique_ratio",
        "avg_pairwise_hd_raw",
        "avg_pairwise_hd_norm",
        "entropy_bits",
        "eff_states",
        "eff_ratio",
    ]

    bucket = defaultdict(list)
    for row in per_rows:
        key = (row["method"], row["round"])
        bucket[key].append(row)

    agg_rows = []
    for (method, rnd), rows in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1])):
        out = {
            "method": method,
            "round": rnd,
            "n_diffs": len(rows),
        }

        for metric in metrics:
            vals = sorted(float(r[metric]) for r in rows)
            out[f"{metric}_mean"] = mean(vals)
            out[f"{metric}_median"] = median(vals)
            out[f"{metric}_q25"] = quantile(vals, 0.25)
            out[f"{metric}_q75"] = quantile(vals, 0.75)
            out[f"{metric}_min"] = vals[0]
            out[f"{metric}_max"] = vals[-1]

        agg_rows.append(out)

    return agg_rows


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze SPECK96 diversity metrics from result txt files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing result txt files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save CSV outputs")
    parser.add_argument("--round-tag", type=str, default="14round", help="e.g. 14round or 11round")
    parser.add_argument("--total-bits", type=int, default=96, help="Total state bits, SPECK96 => 96")
    parser.add_argument("--methods", type=str, default="egbs,ga,sp,mp",
                        help="Comma-separated subset of methods to include, e.g. egbs,sp,mp")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    selected_methods = {x.strip() for x in args.methods.split(",") if x.strip()}

    file_map = scan_input_files(input_dir=input_dir, round_tag=args.round_tag)

    # Filter methods
    file_map = {m: d for m, d in file_map.items() if m in selected_methods}

    # Report discovered files
    print("=== Discovered files ===")
    for method, diff_map in file_map.items():
        print(f"{method}: {len(diff_map)} files")
        if diff_map:
            sample_diffs = sorted(diff_map.keys())[:8]
            print(f"  sample diff_l: {sample_diffs}")

    per_rows: List[Dict] = []
    file_inventory_rows: List[Dict] = []

    for method, diff_map in sorted(file_map.items()):
        for diff_l, path in sorted(diff_map.items()):
            print(f"[{method}] parsing diff_l={diff_l} -> {path.name}")

            try:
                paths = parse_result_file(path)
            except Exception as e:
                print(f"  !! parse failed: {e}")
                continue

            file_inventory_rows.append({
                "method": method,
                "diff_l": diff_l,
                "file_name": path.name,
                "parsed_paths": len(paths),
            })

            if not paths:
                print("  !! no paths parsed")
                continue

            metrics_rows = compute_metrics_for_paths(paths, total_bits=args.total_bits)

            for row in metrics_rows:
                row_out = {
                    "method": method,
                    "diff_l": diff_l,
                    "file_name": path.name,
                    "total_paths": len(paths),
                    **row,
                }
                per_rows.append(row_out)

    # Save per-diff rows
    per_fieldnames = [
        "method", "diff_l", "file_name", "total_paths",
        "round", "states_count",
        "unique_states", "unique_ratio",
        "avg_pairwise_hd_raw", "avg_pairwise_hd_norm",
        "entropy_bits", "eff_states", "eff_ratio",
    ]
    write_csv(output_dir / "per_diff_round_metrics.csv", per_rows, per_fieldnames)

    # Save file inventory
    inv_fieldnames = ["method", "diff_l", "file_name", "parsed_paths"]
    write_csv(output_dir / "file_inventory.csv", file_inventory_rows, inv_fieldnames)

    # Save aggregate rows
    agg_rows = aggregate_rows(per_rows)
    agg_fieldnames = ["method", "round", "n_diffs"]
    metric_prefixes = [
        "unique_states",
        "unique_ratio",
        "avg_pairwise_hd_raw",
        "avg_pairwise_hd_norm",
        "entropy_bits",
        "eff_states",
        "eff_ratio",
    ]
    for p in metric_prefixes:
        agg_fieldnames.extend([
            f"{p}_mean",
            f"{p}_median",
            f"{p}_q25",
            f"{p}_q75",
            f"{p}_min",
            f"{p}_max",
        ])
    write_csv(output_dir / "aggregate_round_metrics.csv", agg_rows, agg_fieldnames)

    print("\n=== Done ===")
    print(f"Saved: {output_dir / 'per_diff_round_metrics.csv'}")
    print(f"Saved: {output_dir / 'aggregate_round_metrics.csv'}")
    print(f"Saved: {output_dir / 'file_inventory.csv'}")


if __name__ == "__main__":
    main()
