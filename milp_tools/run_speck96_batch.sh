#!/usr/bin/env bash
set -u
set -o pipefail

# 批量运行 Speck96 MILP 搜索：
#   speck96diff-12-round-11-fixed.lp
#   speck96diff-14-round-11-fixed.lp
#   ...
#   speck96diff-38-round-11-fixed.lp
#
# 特性：
# 1. 终端实时显示 Gurobi 搜索过程
# 2. 同时把每次运行的完整输出保存到日志文件
# 3. 把每个任务的耗时、退出码、状态写入 CSV
#
# 用法：
#   chmod +x /home/user/speck_differential/run_speck96_batch.sh
#   /home/user/speck_differential/run_speck96_batch.sh

GUROBI_BIN="gurobi_cl"
THREADS=32
ROUND=11
DIFF_START=21
DIFF_END=37
DIFF_STEP=2

BASE_DIR="/home/user/speck_output/speck96"
LP_PREFIX="speck96diff"
LP_SUFFIX="-round-${ROUND}-fixed.lp"
SOL_PREFIX="speck96round${ROUND}"

TIME_LOG="${BASE_DIR}/speck96_round${ROUND}_batch_times.csv"
RUN_LOG_DIR="${BASE_DIR}/logs_round${ROUND}"
mkdir -p "${RUN_LOG_DIR}"
mkdir -p "${BASE_DIR}"

if ! command -v "${GUROBI_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] 未找到 ${GUROBI_BIN}，请先确认 Gurobi 已加入 PATH。" >&2
    exit 1
fi

if [[ ! -f "${TIME_LOG}" || ! -s "${TIME_LOG}" ]]; then
    echo "diff,lp_file,sol_file,start_time,end_time,elapsed_seconds,exit_code,status" > "${TIME_LOG}"
fi

echo "========== Speck96 batch MILP start =========="
echo "BASE_DIR   = ${BASE_DIR}"
echo "ROUND      = ${ROUND}"
echo "THREADS    = ${THREADS}"
echo "DIFF RANGE = ${DIFF_START}..${DIFF_END} step ${DIFF_STEP}"
echo "TIME_LOG   = ${TIME_LOG}"
echo

for ((diff=${DIFF_START}; diff<=${DIFF_END}; diff+=${DIFF_STEP})); do
    lp_file="${BASE_DIR}/${LP_PREFIX}-${diff}${LP_SUFFIX}"
    sol_file="${BASE_DIR}/${SOL_PREFIX}-${diff}.sol"
    run_log="${RUN_LOG_DIR}/gurobi_diff_${diff}.log"

    if [[ ! -f "${lp_file}" ]]; then
        echo "[WARN] 跳过：文件不存在 -> ${lp_file}"
        now="$(date '+%F %T')"
        echo "${diff},${lp_file},${sol_file},${now},${now},0,127,LP_NOT_FOUND" >> "${TIME_LOG}"
        continue
    fi

    echo "============================================================"
    echo "[INFO] 开始求解 diff=${diff}"
    echo "[INFO] lp  = ${lp_file}"
    echo "[INFO] sol = ${sol_file}"
    echo "[INFO] log = ${run_log}"
    echo "============================================================"

    start_time="$(date '+%F %T')"
    start_ts=$(date +%s)

    # 既实时显示到终端，也完整写入日志
    # stdbuf 用于尽量减少因为管道造成的缓冲延迟
    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf -oL -eL "${GUROBI_BIN}" \
            Threads="${THREADS}" \
            ResultFile="${sol_file}" \
            "${lp_file}" 2>&1 | tee "${run_log}"
        exit_code=${PIPESTATUS[0]}
    else
        "${GUROBI_BIN}" \
            Threads="${THREADS}" \
            ResultFile="${sol_file}" \
            "${lp_file}" 2>&1 | tee "${run_log}"
        exit_code=${PIPESTATUS[0]}
    fi

    end_time="$(date '+%F %T')"
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    status="UNKNOWN"
    if grep -q "Optimal solution found" "${run_log}"; then
        status="OPTIMAL"
    elif grep -q "Model is infeasible" "${run_log}"; then
        status="INFEASIBLE"
    elif grep -q "Time limit reached" "${run_log}"; then
        status="TIME_LIMIT"
    elif grep -q "Interrupted" "${run_log}"; then
        status="INTERRUPTED"
    elif [[ ${exit_code} -eq 0 ]]; then
        status="FINISHED"
    else
        status="FAILED"
    fi

    echo "${diff},${lp_file},${sol_file},${start_time},${end_time},${elapsed},${exit_code},${status}" >> "${TIME_LOG}"

    echo
    echo "[INFO] 结束 diff=${diff} | elapsed=${elapsed}s | exit_code=${exit_code} | status=${status}"
    echo

done

echo "========== Speck96 batch MILP finished =========="
echo "耗时日志已写入: ${TIME_LOG}"
echo "单次运行日志目录: ${RUN_LOG_DIR}"
