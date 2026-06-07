#!/usr/bin/env bash
# =============================================================================
# 서버용 multi-model 직렬 평가 wrapper (예외처리 강화)
# =============================================================================
# 흐름: 각 모델 → vLLM 기동 → AutoRAG 평가 → vLLM 종료 → 다음
# 결과: autorag/benchmark_<safe_name>/  +  autorag/PROD_RESULT_FINAL.md (자동 생성)
#
# 예외처리:
#   - HF API 401/404 사전 체크 (gated/missing 모델 즉시 skip)
#   - vLLM 기동 1회 retry 후 실패 시 skip
#   - 평가 timeout 60분 (hang 방지)
#   - 디스크 90% 도달 시 모든 잔여 모델 skip (디스크 보호)
#   - SIGHUP/INT/TERM trap (SSH 끊겨도 정리)
#   - 모든 결과 SUMMARY_LOG 에 모델별 성공/실패/시간 기록
#   - 끝나면 collect_results.py 자동 실행 → PROD_RESULT_FINAL.md
#
# 백그라운드 안전 기동 (SSH 끊겨도 살아있음):
#   nohup setsid bash autorag/run_server_multi.sh > run_all.log 2>&1 < /dev/null &
#   disown
# =============================================================================
set -uo pipefail   # set -e 빼기: 한 모델 실패해도 다음 모델 진행
shopt -s nullglob

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
CONFIG_TEMPLATE="$REPO/autorag/config.server.yaml"
COLLECT_SCRIPT="$REPO/autorag/collect_results.py"

VLLM_GPU="${VLLM_GPU:-2}"
EVAL_GPU="${EVAL_GPU:-3}"
PORT="${PORT:-8000}"
DISK_LIMIT_PCT="${DISK_LIMIT_PCT:-90}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-3600}"      # 60분
VLLM_TIMEOUT="${VLLM_TIMEOUT:-1500}"      # 25분 (큰 모델 다운로드 대비)
STREAM_TAG="${STREAM_TAG:-}"               # 병렬 실행 시 stream 식별

SUMMARY_LOG="${SUMMARY_LOG:-$REPO/run_summary${STREAM_TAG:+_$STREAM_TAG}.log}"
VLLM_LOG="${VLLM_LOG:-$REPO/vllm${STREAM_TAG:+_$STREAM_TAG}.log}"
echo "===== run started: $(date) (stream=${STREAM_TAG:-default}) =====" >> "$SUMMARY_LOG"

VLLM_PID=""

log_summary() {
    local msg
    msg="$(date '+%Y-%m-%d %H:%M:%S') $*"
    echo "$msg"
    echo "$msg" >> "$SUMMARY_LOG"
}

stop_vllm() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill -9 "$VLLM_PID" 2>/dev/null || true
        # process group 까지 (setsid 안 썼으니 같은 pg 안일 수 있음)
        kill -9 -- "-$VLLM_PID" 2>/dev/null || true
    fi
    sleep 3
    # 내 GPU(VLLM_GPU) 만 점유한 PID 찾아 kill — 다른 stream 영향 X
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$VLLM_GPU" 2>/dev/null | tr -d ' ')
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
        sleep 3
    fi
    VLLM_PID=""
}

ensure_gpu_free() {
    local free_mb
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$VLLM_GPU" 2>/dev/null | tr -d ' ')
    if [[ -z "$free_mb" ]]; then return 0; fi
    if (( free_mb < 20000 )); then
        log_summary "[gpu] GPU $VLLM_GPU free ${free_mb}MB < 20GB. 강제 정리"
        nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$VLLM_GPU" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        sleep 5
    fi
}

trap 'log_summary "===== INTERRUPTED ====="; stop_vllm; exit 130' INT TERM HUP

# ---- 평가할 모델 리스트 ----
DEFAULT_MODELS=(
    # ---- 매우 큰 (32B 클래스, AWQ 양자화로 한 장에 fit) ----
    "Qwen/Qwen3-32B-AWQ|awq_marlin|auto|0.7"
    "Qwen/Qwen2.5-32B-Instruct-AWQ|awq_marlin|auto|0.7"
    "Qwen/QwQ-32B-AWQ|awq_marlin|auto|0.7"
    "cyankiwi/gemma-4-31B-it-AWQ-4bit|awq_marlin|auto|0.7"
    "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit|awq_marlin|auto|0.65"
    "casperhansen/deepseek-r1-distill-qwen-32b-awq|awq_marlin|auto|0.7"

    # ---- 큰 (12–14B) ----
    "Qwen/Qwen2.5-14B-Instruct-AWQ|awq_marlin|auto|0.6"
    "Qwen/Qwen3-14B-AWQ|awq_marlin|auto|0.6"
    "mistralai/Mistral-Nemo-Instruct-2407|none|bfloat16|0.75"

    # ---- 한국어 특화 (8B) ----
    "MLP-KTLim/llama-3-Korean-Bllossom-8B|none|bfloat16|0.6"
    "NCSOFT/Llama-VARCO-8B-Instruct|none|bfloat16|0.6"

    # ---- 다국어 (9–12B) ----
    "01-ai/Yi-1.5-9B-Chat|none|bfloat16|0.6"
    "google/gemma-3-12b-it|none|bfloat16|0.75"
    "google/gemma-3-4b-it|none|bfloat16|0.4"

    # ---- Gemma 4 작은 ----
    "google/gemma-4-E4B-it|none|bfloat16|0.4"
    "google/gemma-4-E2B-it|none|bfloat16|0.3"

    # ---- 7–8B 클래스 ----
    "Qwen/Qwen3-8B|none|bfloat16|0.55"
    "Qwen/Qwen2.5-7B-Instruct-AWQ|awq_marlin|auto|0.45"
    "mistralai/Mistral-7B-Instruct-v0.3|none|bfloat16|0.5"
    "HuggingFaceH4/zephyr-7b-beta|none|bfloat16|0.5"
    "01-ai/Yi-1.5-6B-Chat|none|bfloat16|0.45"

    # ---- 작은 (1.5–4B) ----
    "Qwen/Qwen2.5-3B-Instruct|none|bfloat16|0.4"
    "Bllossom/llama-3.2-Korean-Bllossom-3B|none|bfloat16|0.4"
    "microsoft/Phi-3.5-mini-instruct|none|bfloat16|0.4"
    "google/gemma-3-1b-it|none|bfloat16|0.25"
    "Qwen/Qwen2.5-1.5B-Instruct|none|bfloat16|0.3"
)

if [[ -n "${VALID_FILE:-}" && -f "$VALID_FILE" ]]; then
    # precheck 의 valid_models.txt (spec 그대로 = "model|quant|dtype|util")
    MODEL_SPECS=()
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        MODEL_SPECS+=("$line")
    done < "$VALID_FILE"
elif [[ -n "${MODELS:-}" ]]; then
    IFS=' ' read -ra MODEL_LIST <<< "$MODELS"
    MODEL_SPECS=()
    for m in "${MODEL_LIST[@]}"; do
        if [[ "$m" == *"AWQ"* ]]; then
            MODEL_SPECS+=("$m|awq_marlin|auto|0.6")
        else
            MODEL_SPECS+=("$m|none|bfloat16|0.6")
        fi
    done
else
    MODEL_SPECS=("${DEFAULT_MODELS[@]}")
fi

# ---- helper ----
check_disk() {
    local pct
    pct=$(df --output=pcent "$REPO" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [[ -z "$pct" ]]; then return 0; fi
    if (( pct >= DISK_LIMIT_PCT )); then
        log_summary "[disk] ${pct}% used >= ${DISK_LIMIT_PCT}% limit. SKIP remaining models."
        return 1
    fi
    return 0
}

start_vllm() {
    local model="$1" quant="$2" dtype="$3" util="$4" mml="${5:-8192}"
    local quant_arg=()
    [[ "$quant" != "none" ]] && quant_arg=( --quantization "$quant" )

    nohup env CUDA_VISIBLE_DEVICES="$VLLM_GPU" "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --served-model-name "$model" \
        --dtype "$dtype" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "$util" \
        --max-model-len "$mml" \
        --enforce-eager \
        --trust-remote-code \
        --port "$PORT" \
        "${quant_arg[@]}" \
        > "$VLLM_LOG" 2>&1 < /dev/null &
    VLLM_PID=$!

    local waited=0
    while true; do
        if curl -fsS "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
            log_summary "[vllm] ready (${waited}s)"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            log_summary "[vllm] process died for $model after ${waited}s"
            return 1
        fi
        if (( waited > VLLM_TIMEOUT )); then
            log_summary "[vllm] ${VLLM_TIMEOUT}s timeout for $model"
            kill -9 "$VLLM_PID" 2>/dev/null || true
            return 1
        fi
        sleep 10
        waited=$((waited+10))
    done
}

# ---- main loop ----
TOTAL=${#MODEL_SPECS[@]}
IDX=0
SUCCESS=0
SKIPPED=0
FAILED=0

log_summary "총 ${TOTAL}개 모델 평가 시작 (vLLM GPU=$VLLM_GPU, eval GPU=$EVAL_GPU)"

for spec in "${MODEL_SPECS[@]}"; do
    IDX=$((IDX+1))
    IFS='|' read -r MODEL QUANT DTYPE UTIL MML <<< "$spec"
    MML=${MML:-8192}
    SAFE_NAME="$(echo "$MODEL" | tr '/' '_' | tr ':' '_')"
    PROJECT_DIR="$REPO/autorag/benchmark_$SAFE_NAME"
    CONFIG_RUN="$REPO/autorag/_config.run.$SAFE_NAME.yaml"

    log_summary "============================================="
    log_summary "[$IDX/$TOTAL] MODEL: $MODEL"

    # 1) 디스크 체크
    if ! check_disk; then
        SKIPPED=$((SKIPPED+1))
        break
    fi

    # 1b) GPU 메모리 사전 확인 — 이전 모델 leak 정리
    ensure_gpu_free

    # 2) HF 모델 ID 사전 체크
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "https://huggingface.co/api/models/$MODEL" 2>/dev/null || echo 000)
    if [[ "$code" != "200" ]]; then
        log_summary "[skip] $MODEL — HF API ${code} (gated/missing/network)"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    # 3) vLLM 기동 (최대 2회 시도)
    attempt=0
    success=0
    while (( attempt < 2 )); do
        attempt=$((attempt+1))
        log_summary "[vllm start attempt $attempt] $MODEL"
        if start_vllm "$MODEL" "$QUANT" "$DTYPE" "$UTIL" "$MML"; then
            success=1
            break
        fi
        log_summary "[vllm fail attempt $attempt] cleanup + sleep 15s"
        stop_vllm
        sleep 15
    done

    if (( success == 0 )); then
        log_summary "[SKIP] $MODEL — vLLM 기동 2회 실패"
        SKIPPED=$((SKIPPED+1))
        stop_vllm
        continue
    fi

    # 4) config 치환
    sed "s|__MODEL_PLACEHOLDER__|$MODEL|g" "$CONFIG_TEMPLATE" > "$CONFIG_RUN"

    # 5) 평가 (timeout 적용)
    rm -rf "$PROJECT_DIR"
    eval_start=$(date +%s)
    log_summary "[eval start] $MODEL"

    timeout --signal=KILL "$EVAL_TIMEOUT" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
        OPENAI_API_KEY=EMPTY \
        OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" \
        "$PY" "$REPO/autorag/run_autorag.py" \
        --config "$CONFIG_RUN" \
        --project-dir "$PROJECT_DIR" \
        > "$REPO/eval_$SAFE_NAME.log" 2>&1
    rc=$?
    eval_dur=$(( $(date +%s) - eval_start ))

    if [[ $rc -eq 0 ]]; then
        log_summary "[OK] $MODEL — ${eval_dur}s"
        SUCCESS=$((SUCCESS+1))
    elif [[ $rc -eq 124 || $rc -eq 137 ]]; then
        log_summary "[TIMEOUT] $MODEL — ${EVAL_TIMEOUT}s exceeded"
        FAILED=$((FAILED+1))
    else
        log_summary "[FAIL] $MODEL — rc=$rc, ${eval_dur}s, log=$REPO/eval_$SAFE_NAME.log"
        FAILED=$((FAILED+1))
    fi

    rm -f "$CONFIG_RUN"

    # 6) vLLM 정리 (다음 모델 위해)
    stop_vllm

    # 6b) GPU 회수 검증 — 내 vLLM GPU 만 점검 (다른 stream 영향 X)
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$VLLM_GPU" 2>/dev/null | tr -d ' ')
    if [[ -n "$free_mb" ]] && (( free_mb < 30000 )); then
        log_summary "[gpu cleanup] GPU $VLLM_GPU free=${free_mb}MB. 추가 정리"
        nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$VLLM_GPU" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        sleep 5
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$VLLM_GPU" 2>/dev/null | tr -d ' ')
        log_summary "[gpu] 재정리 후 GPU $VLLM_GPU free=${free_mb}MB"
    fi
done

log_summary "===== run completed: $(date) ====="
log_summary "총 ${TOTAL}개 — 성공 $SUCCESS / 실패 $FAILED / 스킵 $SKIPPED"

# ---- 자동 결과 정리 ----
if [[ -f "$COLLECT_SCRIPT" ]]; then
    log_summary "[collect] PROD_RESULT_FINAL.md 작성 시작"
    "$PY" "$COLLECT_SCRIPT" >> "$SUMMARY_LOG" 2>&1 && \
        log_summary "[collect] OK $REPO/autorag/PROD_RESULT_FINAL.md" || \
        log_summary "[collect] failed (보존된 csv 로 수동 수집 가능)"
fi

cat <<EOF

==========================================
완료. 결과 요약:
  성공:  $SUCCESS
  실패:  $FAILED
  스킵:  $SKIPPED
  로그:  $SUMMARY_LOG
  결과:  $REPO/autorag/PROD_RESULT_FINAL.md
==========================================
EOF
