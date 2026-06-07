#!/bin/bash
# Overnight auto: env_A 21모델 평가 → 72B vLLM 띄움 → judge → 모든 vLLM kill
set -u
cd /home/jinwoolee/surion

LOG=/home/jinwoolee/surion/auto_overnight.log
exec > "$LOG" 2>&1

echo "[$(date '+%F %T')] === AUTO OVERNIGHT START ==="

# 1) env_A 21모델 평가 (GPU 0+1, port 8000)
echo "[$(date '+%F %T')] === STEP 1: env_A 21모델 평가 시작 ==="
DISK_LIMIT_PCT=98 VLLM_GPU=0 EVAL_GPU=1 STREAM_TAG=envA PORT=8000 \
  CONFIG_FILE=/home/jinwoolee/surion/autorag/config.server.A.yaml \
  TORCH_CUDA_ARCH_LIST="12.0+PTX" VLLM_USE_FLASHINFER_SAMPLER=0 \
  bash autorag/run_server_multi.sh
ENVA_RC=$?
echo "[$(date '+%F %T')] env_A 완료 (rc=$ENVA_RC)"

# 2) 잔여 vLLM 정리
echo "[$(date '+%F %T')] === STEP 2: vLLM 정리 ==="
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

# 3) 72B vLLM 시작 (TP=2, GPU 0+1)
echo "[$(date '+%F %T')] === STEP 3: 72B vLLM 시작 ==="
CUDA_VISIBLE_DEVICES=0,1 TORCH_CUDA_ARCH_LIST="12.0+PTX" VLLM_USE_FLASHINFER_SAMPLER=0 \
  nohup /home/jinwoolee/surion/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --served-model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --dtype auto --tensor-parallel-size 2 --gpu-memory-utilization 0.92 \
    --max-model-len 16384 --enforce-eager --trust-remote-code --quantization awq_marlin \
    --port 8000 > /tmp/vllm_72b_judge.log 2>&1 &
VLLM72_PID=$!
echo "[$(date '+%F %T')] 72B PID=$VLLM72_PID"

# 4) 72B ready 대기 (최대 10분)
echo "[$(date '+%F %T')] === STEP 4: 72B ready 대기 ==="
READY=0
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
    READY=1
    echo "[$(date '+%F %T')] 72B ready (${i}회 시도)"
    break
  fi
  sleep 10
done
if [[ $READY -eq 0 ]]; then
  echo "[$(date '+%F %T')] 72B ready timeout — judge 스킵"
else
  # 5) judge 스크립트 실행
  echo "[$(date '+%F %T')] === STEP 5: judge 시작 ==="
  /home/jinwoolee/surion/.venv/bin/python \
    /home/jinwoolee/surion/autorag/judge_all.py \
    --benchmark-root /home/jinwoolee/surion/autorag \
    --llm-model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --port 8000 \
    --out-csv /home/jinwoolee/surion/autorag/judge_scores.csv \
    --avg-csv /home/jinwoolee/surion/autorag/judge_avg.csv \
    2>&1 | tee /home/jinwoolee/surion/autorag/judge.log
  echo "[$(date '+%F %T')] judge 완료"
fi

# 6) 모든 사용자 vLLM/wrapper 종료
echo "[$(date '+%F %T')] === STEP 6: 잔여 프로세스 종료 ==="
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -u "$USER" -f "run_server_multi" 2>/dev/null || true
pkill -9 -u "$USER" -f "run_autorag.py" 2>/dev/null || true
sleep 5

# 7) 최종 GPU 상태
echo "[$(date '+%F %T')] === STEP 7: 최종 GPU 상태 ==="
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader

echo "[$(date '+%F %T')] === AUTO OVERNIGHT DONE ==="
