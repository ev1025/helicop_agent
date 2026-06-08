#!/bin/bash
# 추가 재평가: RedHatAI/gemma-3-12b-it-quantized.w4a16 (단일 모델)
#   - envB (.venv_tf5, vLLM 0.21.0)
#   - prod config (max_tokens 256)
#   - 평가 후 72B Judge 재실행 (judge_avg.csv 갱신)
set -u

REPO="/home/jinwoolee/surion"
LOG="$REPO/auto_retry_g12b.log"
exec > "$LOG" 2>&1

echo "[$(date '+%F %T')] ============================================"
echo "[$(date '+%F %T')] === AUTO RETRY g12b 시작 ==="
echo "[$(date '+%F %T')] ============================================"

# STEP 0: 사전 GPU 점유 확인
echo "[$(date '+%F %T')] ===== STEP 0: GPU 점유 확인 ====="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# STEP 1: RedHatAI/gemma-3-12b 평가 (.venv_tf5)
echo "[$(date '+%F %T')] ===== STEP 1: gemma-3-12b w4a16 평가 ====="
DISK_LIMIT_PCT=98 \
  VLLM_PY="/home/jinwoolee/.venv_tf5/bin/python" \
  VLLM_GPU=2 EVAL_GPU=3 \
  STREAM_TAG=retryG12b PORT=8000 \
  VALID_FILE=/tmp/valid_g12b.txt \
  CONFIG_FILE="$REPO/autorag/config.server.api.prod.yaml" \
  TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  bash "$REPO/autorag/run_server_multi.sh"

# STEP 2: vLLM kill
echo "[$(date '+%F %T')] ===== STEP 2: vLLM kill ====="
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -u "$USER" -f "run_autorag.py" 2>/dev/null || true
sleep 10

# STEP 3: 72B vLLM 시작
echo "[$(date '+%F %T')] ===== STEP 3: 72B vLLM 시작 ====="
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
    > /tmp/vllm_72b_g12b.log 2>&1 &
VLLM72_PID=$!
echo "[$(date '+%F %T')] 72B PID=$VLLM72_PID"

# STEP 4: 72B ready 대기 + Judge
echo "[$(date '+%F %T')] ===== STEP 4: 72B ready 대기 ====="
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
    2>&1 | tee "$REPO/autorag/judge_g12b.log"
  echo "[$(date '+%F %T')] Judge 완료"
else
  echo "[$(date '+%F %T')] 72B ready timeout — judge 스킵"
fi

# STEP 5: 본인 process 전체 KILL
echo "[$(date '+%F %T')] ===== STEP 5: 본인 process KILL ====="
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -u "$USER" -f "run_server_multi" 2>/dev/null || true
pkill -9 -u "$USER" -f "run_autorag.py" 2>/dev/null || true
sleep 10

# STEP 6: 최종 GPU
echo "[$(date '+%F %T')] ===== STEP 6: 최종 GPU 상태 ====="
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo "[$(date '+%F %T')] === AUTO RETRY g12b 완료 ==="
