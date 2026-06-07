#!/bin/bash
# Direct eval: vLLM 띄움 → evaluate_direct.py 실행 → vLLM kill, 모델별 순차
set -u

REPO="/home/jinwoolee/surion"
VALID_FILE="${VALID_FILE:?VALID_FILE required}"
VLLM_PY="${VLLM_PY:-$REPO/.venv/bin/python}"
EVAL_PY="${EVAL_PY:-$REPO/.venv/bin/python}"
VLLM_GPU="${VLLM_GPU:-2}"
EVAL_GPU="${EVAL_GPU:-3}"
PORT="${PORT:-8001}"
EMBED_MODEL="${EMBED_MODEL:-nlpai-lab/KURE-v1}"
EMB_CACHE="${EMB_CACHE:-/tmp/corpus_emb_kure.npy}"
OUT_DIR="${OUT_DIR:-$REPO/eval_direct_results}"
mkdir -p "$OUT_DIR"

LOG="$REPO/run_eval_direct.log"
echo "[$(date '+%F %T')] === DIRECT EVAL START ===" | tee -a "$LOG"

while IFS='|' read -r MODEL QUANT DTYPE UTIL MML; do
    [[ -z "$MODEL" || "${MODEL:0:1}" == "#" ]] && continue
    SAFE_NAME=$(echo "$MODEL" | tr '/' '_')
    OUT_FILE="$OUT_DIR/${SAFE_NAME}.parquet"
    if [[ -f "$OUT_FILE" ]]; then
        echo "[$(date '+%F %T')] [skip] $MODEL (already done)" | tee -a "$LOG"
        continue
    fi

    echo "[$(date '+%F %T')] === $MODEL ===" | tee -a "$LOG"

    # vLLM 시작
    quant_arg=()
    [[ "$QUANT" != "none" ]] && quant_arg=( --quantization "$QUANT" )
    mml_arg=()
    [[ -n "${MML:-}" ]] && mml_arg=( --max-model-len "$MML" )

    CUDA_VISIBLE_DEVICES="$VLLM_GPU" TORCH_CUDA_ARCH_LIST="12.0+PTX" VLLM_USE_FLASHINFER_SAMPLER=0 \
        nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --served-model-name "$MODEL" \
            --dtype "$DTYPE" --tensor-parallel-size 1 \
            --gpu-memory-utilization "$UTIL" \
            --enforce-eager --trust-remote-code \
            --max-num-batched-tokens 8192 \
            --limit-mm-per-prompt '{"image":0}' \
            --port "$PORT" \
            "${mml_arg[@]}" "${quant_arg[@]}" \
            > "$REPO/vllm_direct.log" 2>&1 < /dev/null &
    VLLM_PID=$!

    # ready 대기 (max 300s) — VLLM_READY 플래그로 명시
    VLLM_READY=0
    for i in $(seq 1 60); do
        if curl -fsS "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
            echo "[$(date '+%F %T')] vLLM ready" | tee -a "$LOG"
            VLLM_READY=1
            break
        fi
        if grep -qE "raise ValueError|Engine core initialization failed" "$REPO/vllm_direct.log" 2>/dev/null; then
            echo "[$(date '+%F %T')] vLLM error - kill" | tee -a "$LOG"
            kill -9 $VLLM_PID 2>/dev/null
            pkill -9 -P $VLLM_PID 2>/dev/null
            break
        fi
        sleep 5
    done

    # eval 실행 (VLLM_READY 플래그 기반)
    if [[ $VLLM_READY -eq 1 ]]; then
        echo "[$(date '+%F %T')] eval start" | tee -a "$LOG"
        CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
            timeout 1800 "$EVAL_PY" "$REPO/autorag/evaluate_direct.py" \
                --llm-model "$MODEL" \
                --port "$PORT" \
                --embed-model "$EMBED_MODEL" \
                --corpus-emb-cache "$EMB_CACHE" \
                --out "$OUT_FILE" \
                >> "$LOG" 2>&1
        RC=$?
        if [[ $RC -eq 0 ]]; then
            echo "[$(date '+%F %T')] [OK] $MODEL" | tee -a "$LOG"
        else
            echo "[$(date '+%F %T')] [FAIL] $MODEL rc=$RC" | tee -a "$LOG"
        fi
    else
        echo "[$(date '+%F %T')] [SKIP] $MODEL — vLLM not ready" | tee -a "$LOG"
    fi

    # vLLM kill
    kill -9 $VLLM_PID 2>/dev/null
    pkill -9 -f "vllm.entrypoints.*--port $PORT" 2>/dev/null
    sleep 5
    # GPU memory clear
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$VLLM_GPU" 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 5

done < "$VALID_FILE"

echo "[$(date '+%F %T')] === DIRECT EVAL DONE ===" | tee -a "$LOG"
