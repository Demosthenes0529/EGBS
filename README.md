# EGBS: Evolution-Guided Beam Search for Differential Trail Search

本仓库实现了论文 **"Evolution-Guided Beam Search for Differential Trail Search of Block Ciphers"** 的主要实验代码与分析脚本。

- 论文源码（LaTeX）：`IEEE-Transactions-LaTeX2e-templates-and-instructions/`
- 其余目录：与论文实验对应的代码、结果与分析工具

项目目标是面向区分路径搜索（differential trail search）构建一个统一框架，在**有界束搜索（Beam Search）**基础上加入**进化式多样性增强（EA stage）**，用于 ARX 与 SPN 两类分组密码（主要覆盖 SPECK 与 PRESENT）。

---

## 1. 论文与代码对应关系

根据论文内容，核心思想包括：

1. `δ`-optimal 转移生成（控制分支爆炸）
2. 结构感知评分（ARX 采用前瞻评分，SPN 结合活跃 S-box）
3. 演化增强（扰动 + 多父节点过滤）

代码中主要落地为：

- **搜索核心**：`core_search/`
- **实验脚本**（含 MCTS/EGBS/EA 对比与批处理）：`experiments/`
- **结果统计与绘图**：`analysis/`
- **MILP 对照工具**：`milp_tools/`
- **SAT 对照工具**：`sat_tools/`

---

## 2. 目录结构说明

```text
EGBS/
├─ IEEE-Transactions-LaTeX2e-templates-and-instructions/  # 论文 LaTeX 源码
├─ core_search/                                            # EGBS/BS 核心搜索实现
│  ├─ ea_initial_population.py                             # 初始种群/路径搜索（SPECK）
│  ├─ ea_variation.py                                      # 变异与多父过滤相关流程
│  └─ present.py                                           # PRESENT 搜索逻辑
├─ experiments/                                            # 主实验与参数设置脚本
├─ analysis/                                               # 统计、可视化、表格生成
├─ milp_tools/                                             # MILP 建模与路径恢复工具
├─ sat_tools/                                              # SAT CNF 建模与求解/解码工具
├─ saved_initial_paths/                                    # 中间结果
├─ saved_trail/                                            # 路径结果（pkl）
├─ saved_ea_txt/                                           # 路径结果（txt）
├─ saved_metrics/                                          # 指标数据
├─ milp_output/                                            # MILP 输出
├─ sat_output/                                             # SAT 输出
└─ param_ablation/                                         # 参数消融汇总
```

---

## 3. 运行环境

建议环境：

- Python 3.9+（Windows / Linux 均可）
- 推荐 Conda 环境（你当前已在 `pytorch` 环境中工作）

常用依赖（按代码导入统计）：

- `numpy`
- `tqdm`
- `pandas`
- `matplotlib`

安装示例：

```bash
pip install numpy tqdm pandas matplotlib
```

可选外部工具（用于基线对照）：

- **Gurobi**（MILP）
- **CryptoMiniSat**（SAT）

---

## 4. 快速开始（建议顺序）

> 注意：当前代码里有部分历史硬编码路径（如 `./...`）。开源前建议统一改为相对路径或配置文件。

### 4.1 运行核心搜索（SPECK）

- 初始搜索：`core_search/ea_initial_population.py`
- 演化增强：`core_search/ea_variation.py`

这两个脚本的 `if __name__ == "__main__":` 中提供了可直接修改的参数示例（如 `word_size`、`nr`、`delta`、`population_size`、`n_jobs`）。

### 4.2 运行主实验脚本（论文对齐）

- `experiments/mcts_paper_aligned_timecap_stopflag.py`

在 `main()` 顶部可直接设置：

- `CIPHER_BLOCK_SIZE`（32/48/64/96/128）
- `SEARCH_ROUNDS`
- `SEARCH_DIRECTION`（encrypt/decrypt/bidirectional）
- `SEED_MODE`
- `FORWARD_ITERATIONS` / `BACKWARD_ITERATIONS`
- `TARGET_WEIGHT`
- `PARALLEL_PROCESSES`

### 4.3 参数消融

- `experiments/ea_variation_param_ablation.py`

运行后汇总结果会输出到 `param_ablation/`（csv / txt / pkl）。

### 4.4 PRESENT 实验

- 搜索脚本：`core_search/present.py`
- 统计脚本：`analysis/present_diff.py`、`analysis/present_count_diff.py`

### 4.5 多样性分析与绘图

- 指标统计：`analysis/analyze_speck96_diversity.py`
- 曲线绘图：`analysis/plot_diversity_curves_pdf.py`

---

## 5. 基线方法（MILP / SAT）

### MILP

- 目录：`milp_tools/`
- 主要脚本：`speck_diff_find.py`、`get_route_from_sol.py`
- 说明文档：`milp_tools/README.md`

### SAT

- 目录：`sat_tools/`
- 主要脚本：`speck_diff_cnf.py`
- 功能：生成 CNF、可调用求解器、支持解码模型

---

## 6. 结果文件说明

- `saved_ea_txt/`：可读文本路径结果
- `saved_trail/`：序列化路径/树结构（`.pkl`）
- `saved_metrics/`：轮次统计指标
- `param_ablation/`：消融汇总表（csv/txt/pkl）

