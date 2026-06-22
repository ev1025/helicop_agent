#!/bin/bash
# 재평가 자동화: envA retry (3) + envB (5) + 72B Judge 재실행
# - GPU 2,3 만 사용
# - 본인 process만 kill (다른 사용자 영향 0)
# - envA retry: shortgen config (max_tokens 200)
# - envB: .venv_tf5 정확한 경로 (/home/jinwoolee/.venv_tf5)
set -u

REPO="/home/jinwoolee/surion"
LOG="$REPO/auto_retry.log"
exec > "$LOG" 2>&1

echo "[$(date '+%F %T')] ============================================"
echo "[$(date '+%F %T')] === AUTO RETRY 시작 ==="
echo "[$(date '+%F %T')] ============================================"

# ===========================================================
# STEP 0: 다른 사용자 GPU 점유 확인 (안전 가드)
# ===========================================================
echo "[$(date '+%F %T')] ===== STEP 0: 사전 GPU 점유 확인 ====="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo

# ===========================================================
# STEP 1: envA retry (Yi-1.5-9B, Yi-1.5-6B, Mistral-Nemo-AWQ)
#   - .venv (vLLM 0.20.1, transformers 4.57)
#   - shortgen config (max_tokens 200, 4K context 모델 안전)
# ===========================================================
echo "[$(date '+%F %T')] ===== STEP 1: envA retry 시작 (3 모델) ====="
DISK_LIMIT_PCT=98 \
  VLLM_PY="$REPO/.venv/bin/python" \
  VLLM_GPU=2 EVAL_GPU=3 \
  STREAM_TAG=retryA PORT=8000 \
  VALID_FILE=/tmp/valid_prodA_retry.txt \
  CONFIG_FILE="$REPO/autorag/config.server.api.prod.shortgen.yaml" \
  TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  bash "$REPO/autorag/run_server_multi.sh"
echo "[$(date '+%F %T')] envA retry 완료"

# ===========================================================
# STEP 2: vLLM kill (본인 process만)
# ===========================================================
echo "[$(date '+%F %T')] ===== STEP 2: vLLM kill ====="
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -u "$USER" -f "run_autorag.py" 2>/dev/null || true
sleep 10

# ===========================================================
# STEP 3: envB (gemma 시리즈 5개 — RedHatAI 제외)
#   - .venv_tf5 (vLLM 0.21.0, transformers 5.8.1) -- 정확한 경로!
#   - prod config (gemma는 충분한 context 지원)
# ===========================================================
echo "[$(date '+%F %T')] ===== STEP 3: envB 시작 (5 모델) ====="
DISK_LIMIT_PCT=98 \
  VLLM_PY="/home/jinwoolee/.venv_tf5/bin/python" \
  VLLM_GPU=2 EVAL_GPU=3 \
  STREAM_TAG=retryB PORT=8000 \
  VALID_FILE=/tmp/valid_prodB_retry.txt \
  CONFIG_FILE="$REPO/autorag/config.server.api.prod.yaml" \
  TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  bash "$REPO/autorag/run_server_multi.sh"
echo "[$(date '+%F %T')] envB 완료"

# ===========================================================
# STEP 4: vLLM kill (72B 위해)
# ===========================================================
echo "[$(date '+%F %T')] ===== STEP 4: vLLM kill ====="
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -u "$USER" -f "run_autorag.py" 2>/dev/null || true
sleep 10

# ===========================================================
# STEP 5: 72B vLLM 시작 (TP=2, GPU 2,3)
# ===========================================================
echo "[$(date '+%F %T')] ===== STEP 5: 72B vLLM 시작 ====="
CUDA_VISIBLE_DEVICES=2,3 \
  TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  nohup "$REPO/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --served-model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --dtype auto --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 16384 \
    --enforce-eager --trust-remote-code \
    --quantization awq_marlin \
    --port 8000 \
    > /tmp/vllm_72b_judge_retry.log 2>&1 &
VLLM72_PID=$!
echo "[$(date '+%F %T')] 72B PID=$VLLM72_PID"

# ===========================================================
# STEP 6: 72B ready 대기 (10분 max) + Judge 실행
#   - judge_all.py는 benchmark_* 디렉토리 자동 스캔
#   - 결과: judge_avg.csv (전체 모델 일관 score)
# ===========================================================
echo "[$(date '+%F %T')] ===== STEP 6: 72B ready 대기 ====="
READY=0
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
    READY=1
    echo "[$(date '+%F %T')] 72B ready"
    break
  fi
  sleep 10
done

if [[ $READY -eq 1 ]]; then
  echo "[$(date '+%F %T')] ===== Judge 실행 ====="
  "$REPO/.venv/bin/python" "$REPO/autorag/judge_all.py" \
    --benchmark-root "$REPO/autorag" \
    --llm-model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --port 8000 \
    --out-csv "$REPO/autorag/judge_scores.csv" \
    --avg-csv "$REPO/autorag/judge_avg.csv" \
    2>&1 | tee "$REPO/autorag/judge_retry.log"
  echo "[$(date '+%F %T')] Judge 완료"
else
  echo "[$(date '+%F %T')] 72B ready timeout — judge 스킵"
fi

# ===========================================================
# STEP 7: 본인 process 전체 KILL (다른 사용자 GPU 사용 가능)
# ===========================================================
echo "[$(date '+%F %T')] ===== STEP 7: 본인 process KILL ====="
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -u "$USER" -f "run_server_multi" 2>/dev/null || true
pkill -9 -u "$USER" -f "run_autorag.py" 2>/dev/null || true
sleep 10

# ===========================================================
# STEP 8: 최종 GPU 상태
# ===========================================================
echo "[$(date '+%F %T')] ===== STEP 8: 최종 GPU 상태 ====="
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader
echo
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo "[$(date '+%F %T')] ============================================"
echo "[$(date '+%F %T')] === AUTO RETRY 완료 ==="
echo "[$(date '+%F %T')] ============================================"
