#!/bin/bash
# Overnight 본 평가 자동화: envA → envB → 72B Judge → GPU KILL
# GPU 2,3 만 사용. fail 시 SKIP. 컴퓨터 종료해도 진행.
set -u

REPO="/home/jinwoolee/surion"
LOG="$REPO/auto_prod.log"
exec > "$LOG" 2>&1

echo "[$(date '+%F %T')] ============================================"
echo "[$(date '+%F %T')] === AUTO PROD 시작 ==="
echo "[$(date '+%F %T')] ============================================"

# 1) envA (vLLM=GPU2, eval=GPU3, .venv)
echo "[$(date '+%F %T')] ===== STEP 1: envA 평가 시작 ====="
DISK_LIMIT_PCT=98 \
  VLLM_PY="$REPO/.venv/bin/python" \
  VLLM_GPU=2 EVAL_GPU=3 \
  STREAM_TAG=prodApi_A PORT=8000 \
  VALID_FILE=/tmp/valid_prodA.txt \
  CONFIG_FILE="$REPO/autorag/config.server.api.prod.yaml" \
  TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  bash "$REPO/autorag/run_server_multi.sh"
echo "[$(date '+%F %T')] envA 완료"

# 2) 잔여 vLLM kill (envB 위해)
echo "[$(date '+%F %T')] ===== STEP 2: vLLM 정리 ====="
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 5

# 3) envB (vLLM=GPU2, eval=GPU3, .venv_tf5)
echo "[$(date '+%F %T')] ===== STEP 3: envB 평가 시작 ====="
DISK_LIMIT_PCT=98 \
  VLLM_PY="$REPO/.venv_tf5/bin/python" \
  VLLM_GPU=2 EVAL_GPU=3 \
  STREAM_TAG=prodApi_B PORT=8000 \
  VALID_FILE=/tmp/valid_prodB.txt \
  CONFIG_FILE="$REPO/autorag/config.server.api.prod.yaml" \
  TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  bash "$REPO/autorag/run_server_multi.sh"
echo "[$(date '+%F %T')] envB 완료"

# 4) 잔여 vLLM kill (72B 위해)
echo "[$(date '+%F %T')] ===== STEP 4: vLLM 정리 ====="
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 5

# 5) Qwen 72B 시작 (GPU 2,3 TP=2)
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
    > /tmp/vllm_72b_judge.log 2>&1 &
VLLM72_PID=$!
echo "[$(date '+%F %T')] 72B PID=$VLLM72_PID"

# 6) 72B ready 대기 (10분 max)
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
  # 7) Judge 실행
  echo "[$(date '+%F %T')] ===== STEP 7: 72B Judge 실행 ====="
  "$REPO/.venv/bin/python" "$REPO/autorag/judge_all.py" \
    --benchmark-root "$REPO/autorag" \
    --llm-model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --port 8000 \
    --out-csv "$REPO/autorag/judge_scores.csv" \
    --avg-csv "$REPO/autorag/judge_avg.csv" \
    2>&1 | tee "$REPO/autorag/judge.log"
  echo "[$(date '+%F %T')] Judge 완료"
else
  echo "[$(date '+%F %T')] 72B ready timeout — judge 스킵"
fi

# 8) 모든 사용자 GPU process 종료
echo "[$(date '+%F %T')] ===== STEP 8: 잔여 process KILL ====="
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -u "$USER" -f "run_server_multi" 2>/dev/null || true
pkill -9 -u "$USER" -f "run_autorag.py" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 5

# 9) 최종 GPU 상태
echo "[$(date '+%F %T')] ===== STEP 9: 최종 GPU 상태 ====="
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader

echo "[$(date '+%F %T')] ============================================"
echo "[$(date '+%F %T')] === AUTO PROD 완료 ==="
echo "[$(date '+%F %T')] ============================================"
