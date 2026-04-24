import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# =========================
# 基本配置
# =========================
DIFF_L = 0x8000000
NR_TOTAL = 14
SPECK_BITS = 96
ROUND_MIN = 5
ROUND_MAX = 14

BASE_DIR = "./saved_metrics"
OUT_DIR = os.path.join(
    BASE_DIR,
    f"speck{SPECK_BITS}",
    "diversity_plots",
    f"r{NR_TOTAL}",
    f"diff_{DIFF_L}"
)
os.makedirs(OUT_DIR, exist_ok=True)

# PDF 矢量字体
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# =========================
# 路径配置
# =========================
JSONL_PATHS = {
    "egbs": os.path.join(
        BASE_DIR, f"speck{SPECK_BITS}", "egbs", f"r{NR_TOTAL}",
        f"diff_{DIFF_L}_round_metrics.jsonl"
    ),
    "egbs_sp": os.path.join(
        BASE_DIR, f"speck{SPECK_BITS}", "egbs_sp", f"r{NR_TOTAL}",
        f"diff_{DIFF_L}_round_metrics.jsonl"
    ),
    "egbs_mp": os.path.join(
        BASE_DIR, f"speck{SPECK_BITS}", "egbs_mp", f"r{NR_TOTAL}",
        f"diff_{DIFF_L}_round_metrics.jsonl"
    ),
}

PKL_PATHS = {
    "egbs": os.path.join(
        BASE_DIR, f"speck{SPECK_BITS}", "egbs", f"r{NR_TOTAL}",
        f"diff_{DIFF_L}_round_metrics.pkl"
    ),
    "egbs_sp": os.path.join(
        BASE_DIR, f"speck{SPECK_BITS}", "egbs_sp", f"r{NR_TOTAL}",
        f"diff_{DIFF_L}_round_metrics.pkl"
    ),
    "egbs_mp": os.path.join(
        BASE_DIR, f"speck{SPECK_BITS}", "egbs_mp", f"r{NR_TOTAL}",
        f"diff_{DIFF_L}_round_metrics.pkl"
    ),
}

LABEL_MAP = {
    "egbs": "EGBS",
    "egbs_sp": "EGBS+SP",
    "egbs_mp": "EGBS+MP",
}


# =========================
# 工具函数
# =========================
def check_files_exist(paths_dict, file_type_name):
    for name, path in paths_dict.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"[{file_type_name}] missing for {name}: {path}")


def load_round_metrics(jsonl_path, method_label, round_min=5, round_max=14):
    """
    读取一个 method 的 round_metrics.jsonl
    同一个 (round, phase) 若重复，只保留最后一条
    默认取 post_selection 指标
    """
    seen = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj["round"], obj["phase"])
            seen[key] = obj  # 后面的覆盖前面的，达到“清洗去重”的效果

    rows = []
    for (round_id, phase), obj in sorted(seen.items(), key=lambda x: x[0][0]):
        if not (round_min <= round_id <= round_max):
            continue
        post = obj["post_selection"]
        rows.append({
            "method": method_label,
            "round": round_id,
            "phase": phase,
            "path_count": post["path_count"],
            "unique_state_count": post["unique_state_count"],
            "avg_pairwise_hamming": post["avg_pairwise_hamming"],
            "topk_entropy_norm": post["topk_entropy_norm"],
            "topk_unique_states": post.get("topk_unique_states", None),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No usable rows loaded from {jsonl_path}")
    return df


def plot_metric_pdf(df, metric, ylabel, out_path):
    plt.figure(figsize=(6.8, 4.2))

    for method in ["egbs", "egbs_sp", "egbs_mp"]:
        sub = df[df["method"] == method].sort_values("round")
        plt.plot(
            sub["round"],
            sub[metric],
            marker="o",
            linewidth=2,
            label=LABEL_MAP[method]
        )

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.xticks(list(range(ROUND_MIN, ROUND_MAX + 1)))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def load_payload_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def import_bridge_compare_function():
    """
    尝试导入你原工程里的 compare_bridge_states_against_baseline
    """
    candidate_dirs = [
        ".",
        "./",
        "./",
    ]
    for d in candidate_dirs:
        abs_d = os.path.abspath(d)
        if abs_d not in sys.path:
            sys.path.insert(0, abs_d)

    try:
        from speck_diversity_bridge_instrumentation import compare_bridge_states_against_baseline
        return compare_bridge_states_against_baseline
    except Exception as e:
        raise ImportError(
            "Cannot import compare_bridge_states_against_baseline from "
            "speck_diversity_bridge_instrumentation.py\n"
            f"Original error: {e}"
        )


def build_bridge_df(compare_rows, method_name, round_min=5, round_max=14):
    """
    compare_rows 应该是 compare_bridge_states_against_baseline(...) 的输出
    预期包含至少：
      - round
      - rescued_vs_baseline
    """
    df = pd.DataFrame(compare_rows)
    if "round" not in df.columns or "rescued_vs_baseline" not in df.columns:
        raise ValueError(
            f"Bridge comparison rows do not contain expected columns. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[(df["round"] >= round_min) & (df["round"] <= round_max)].copy()
    df["method"] = method_name
    return df[["round", "method", "rescued_vs_baseline"]]


def plot_bridge_bar_pdf(df_bridge, out_path):
    """
    画 bridge states rescued bar:
      - EGBS+SP
      - EGBS+MP
    """
    rounds = list(range(ROUND_MIN, ROUND_MAX + 1))
    x = np.arange(len(rounds))
    width = 0.35

    df_sp = df_bridge[df_bridge["method"] == "EGBS+SP"].set_index("round")
    df_mp = df_bridge[df_bridge["method"] == "EGBS+MP"].set_index("round")

    y_sp = [int(df_sp.loc[r, "rescued_vs_baseline"]) if r in df_sp.index else 0 for r in rounds]
    y_mp = [int(df_mp.loc[r, "rescued_vs_baseline"]) if r in df_mp.index else 0 for r in rounds]

    plt.figure(figsize=(7.0, 4.4))
    plt.bar(x - width / 2, y_sp, width=width, label="EGBS+SP")
    plt.bar(x + width / 2, y_mp, width=width, label="EGBS+MP")

    plt.xlabel("Round")
    plt.ylabel("Rescued bridge states")
    plt.xticks(x, rounds)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    # 1) 检查输入文件
    check_files_exist(JSONL_PATHS, "jsonl")
    check_files_exist(PKL_PATHS, "pkl")

    # 2) 读取三种方法的 diversity 数据
    df_egbs = load_round_metrics(JSONL_PATHS["egbs"], "egbs", ROUND_MIN, ROUND_MAX)
    df_sp   = load_round_metrics(JSONL_PATHS["egbs_sp"], "egbs_sp", ROUND_MIN, ROUND_MAX)
    df_mp   = load_round_metrics(JSONL_PATHS["egbs_mp"], "egbs_mp", ROUND_MIN, ROUND_MAX)

    df_all = pd.concat([df_egbs, df_sp, df_mp], ignore_index=True)
    df_all = df_all.sort_values(["method", "round"]).reset_index(drop=True)

    # 3) 导出明细表
    df_all.to_csv(
        os.path.join(OUT_DIR, f"diff_{DIFF_L}_diversity_summary_round{ROUND_MIN}_{ROUND_MAX}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 4) 画三张 diversity 曲线
    plot_metric_pdf(
        df_all,
        metric="unique_state_count",
        ylabel="Unique states",
        out_path=os.path.join(OUT_DIR, "unique_states_round5_14.pdf"),
    )

    plot_metric_pdf(
        df_all,
        metric="topk_entropy_norm",
        ylabel="Normalized entropy",
        out_path=os.path.join(OUT_DIR, "topk_entropy_round5_14.pdf"),
    )

    plot_metric_pdf(
        df_all,
        metric="avg_pairwise_hamming",
        ylabel="Average pairwise Hamming distance",
        out_path=os.path.join(OUT_DIR, "avg_pairwise_hamming_round5_14.pdf"),
    )

    # 5) 读取 pkl，计算 bridge-state 对比
    compare_bridge_states_against_baseline = import_bridge_compare_function()

    egbs_payload = load_payload_pkl(PKL_PATHS["egbs"])
    sp_payload   = load_payload_pkl(PKL_PATHS["egbs_sp"])
    mp_payload   = load_payload_pkl(PKL_PATHS["egbs_mp"])

    if "round_logs" not in egbs_payload or "round_logs" not in sp_payload or "round_logs" not in mp_payload:
        raise KeyError("Loaded payload .pkl does not contain 'round_logs'.")

    rows_sp = compare_bridge_states_against_baseline(
        egbs_payload["round_logs"],
        sp_payload["round_logs"]
    )
    rows_mp = compare_bridge_states_against_baseline(
        egbs_payload["round_logs"],
        mp_payload["round_logs"]
    )

    df_bridge_sp = build_bridge_df(rows_sp, "EGBS+SP", ROUND_MIN, ROUND_MAX)
    df_bridge_mp = build_bridge_df(rows_mp, "EGBS+MP", ROUND_MIN, ROUND_MAX)
    df_bridge = pd.concat([df_bridge_sp, df_bridge_mp], ignore_index=True)
    df_bridge = df_bridge.sort_values(["round", "method"]).reset_index(drop=True)

    # 导出 bridge 明细
    df_bridge.to_csv(
        os.path.join(OUT_DIR, f"diff_{DIFF_L}_bridge_states_round{ROUND_MIN}_{ROUND_MAX}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 6) 画 bridge states 柱状图
    plot_bridge_bar_pdf(
        df_bridge,
        out_path=os.path.join(OUT_DIR, "bridge_states_rescued_bar.pdf"),
    )

    print("\n=== quick summary: diversity ===")
    print(df_all.pivot(index="round", columns="method", values="unique_state_count"))
    print(df_all.pivot(index="round", columns="method", values="topk_entropy_norm"))
    print(df_all.pivot(index="round", columns="method", values="avg_pairwise_hamming"))

    print("\n=== quick summary: bridge states ===")
    print(df_bridge)

    print(f"\nAll PDFs have been saved to:\n{OUT_DIR}")
