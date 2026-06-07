#!/bin/bash
# AutoRAG 정석 사용 — vllm module 직접 (API server X)
# 모델별 config 교체 + autorag 실행 + cleanup
set -u

REPO="/home/jinwoolee/surion"
VALID_FILE="${VALID_FILE:?VALID_FILE required}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
GPU="${GPU:-0}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-$REPO/autorag/config.server.vllm.yaml}"
STREAM_TAG="${STREAM_TAG:-vllm}"
SUMMARY_LOG="$REPO/run_summary_${STREAM_TAG}.log"
DISK_LIMIT_PCT="${DISK_LIMIT_PCT:-95}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY_LOG"; }

check_disk() {
    local pct=$(df --output=pcent / | tail -1 | tr -d ' %')
    if (( pct >= DISK_LIMIT_PCT )); then
        log "[disk] ${pct}% used >= ${DISK_LIMIT_PCT}% — SKIP"
        return 1
    fi
    return 0
}

ensure_gpu_free() {
    local free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
    if (( free_mb < 20000 )); then
        log "[gpu] GPU $GPU free ${free_mb}MB < 20GB — 강제 정리"
        nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | xargs -r kill -9 2>/dev/null
        sleep 5
    fi
}

log "===== AutoRAG vllm module 평가 시작 (stream=$STREAM_TAG, GPU=$GPU, PYTHON=$PYTHON) ====="
TOTAL=0; SUCCESS=0; FAILED=0; SKIPPED=0

while IFS='|' read -r MODEL QUANT DTYPE UTIL MML; do
    [[ -z "$MODEL" || "${MODEL:0:1}" == "#" ]] && continue
    TOTAL=$((TOTAL+1))
    SAFE_NAME=$(echo "$MODEL" | tr '/' '_')
    PROJECT_DIR="$REPO/autorag/benchmark_${SAFE_NAME}"
    CONFIG_RUN="$REPO/autorag/_run.${SAFE_NAME}.yaml"
    LOG_FILE="$REPO/eval_${SAFE_NAME}.log"

    log "=============================================="
    log "[$TOTAL] MODEL: $MODEL"

    check_disk || { SKIPPED=$((SKIPPED+1)); continue; }
    ensure_gpu_free

    # config: __MODEL_PLACEHOLDER__ 교체 + 모델별 옵션 (UTIL, MML)
    sed "s|__MODEL_PLACEHOLDER__|$MODEL|g" "$CONFIG_TEMPLATE" > "$CONFIG_RUN"
    if [[ -n "${UTIL:-}" ]]; then
        sed -i "s|gpu_memory_utilization: 0.7|gpu_memory_utilization: $UTIL|g" "$CONFIG_RUN"
    fi
    if [[ -n "${MML:-}" ]]; then
        sed -i "s|max_model_len: 7000|max_model_len: $MML|g" "$CONFIG_RUN"
    fi
    if [[ "$QUANT" != "none" ]]; then
        # quantization 추가 (vllm.LLM 파라미터)
        sed -i "/limit_mm_per_prompt:/i\            quantization: $QUANT" "$CONFIG_RUN"
    fi

    # cleanup
    rm -rf "$PROJECT_DIR" "$REPO"/autorag_chroma_*_vllm

    log "[eval start] $MODEL"
    start_ts=$(date +%s)
    CUDA_VISIBLE_DEVICES="$GPU" TORCH_CUDA_ARCH_LIST="12.0+PTX" VLLM_USE_FLASHINFER_SAMPLER=0 \
        timeout 1800 "$PYTHON" "$REPO/autorag/run_autorag.py" \
            --config "$CONFIG_RUN" \
            --project-dir "$PROJECT_DIR" \
            > "$LOG_FILE" 2>&1
    RC=$?
    dur=$(( $(date +%s) - start_ts ))

    if [[ $RC -eq 0 ]]; then
        log "[OK] $MODEL — ${dur}s"
        SUCCESS=$((SUCCESS+1))
    elif [[ $RC -eq 124 || $RC -eq 137 ]]; then
        log "[TIMEOUT] $MODEL — ${dur}s"
        FAILED=$((FAILED+1))
    else
        log "[FAIL] $MODEL — rc=$RC, ${dur}s, log=$LOG_FILE"
        FAILED=$((FAILED+1))
    fi

    rm -f "$CONFIG_RUN"
    # GPU 정리 (vllm.LLM destructor가 cleanup 하지만 안전 차원)
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 5
done < "$VALID_FILE"

log "===== 종료: 총 $TOTAL — 성공 $SUCCESS / 실패 $FAILED / 스킵 $SKIPPED ====="
