"""72B-as-judge — 모든 evaluated model의 generation 결과에 g_eval/faithfulness 점수 매김.

전제: vLLM Qwen2.5-72B-AWQ가 http://127.0.0.1:8000/v1 에 떠 있음.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI

JUDGE_PROMPT = """다음 RAG 시스템 평가용 질문-답변 쌍을 평가해주세요.

[질문]
{query}

[정답 (ground truth)]
{gen_gt}

[모델 답변]
{answer}

다음 3가지 기준으로 각각 0-10점으로 평가해주세요:
1. faithfulness: 모델 답변이 정답에 충실한가? (할루시네이션 없음, 정답 의미 포함)
2. relevance: 모델 답변이 질문에 적절히 답하는가?
3. naturalness: 한국어 표현이 자연스러운가?

반드시 다음 JSON 형식으로만 출력하세요 (설명 없이):
{{"faithfulness": <0-10>, "relevance": <0-10>, "naturalness": <0-10>}}
"""


def parse_scores(text: str) -> dict | None:
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return None
    try:
        d = json.loads(match.group(0))
        for k in ("faithfulness", "relevance", "naturalness"):
            if k not in d:
                return None
            d[k] = float(d[k])
        return d
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="/home/jinwoolee/surion/autorag")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-csv", default="/home/jinwoolee/surion/autorag/judge_scores.csv")
    ap.add_argument("--avg-csv", default="/home/jinwoolee/surion/autorag/judge_avg.csv")
    args = ap.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{args.port}/v1")

    bench_dirs = sorted(glob.glob(f"{args.benchmark_root}/benchmark_*"))
    print(f"benchmark dirs: {len(bench_dirs)}")

    results = []
    for bench_dir in bench_dirs:
        model_name = Path(bench_dir).name.replace("benchmark_", "")
        gen_files = glob.glob(f"{bench_dir}/0/generate_line/generator/*.parquet")
        if not gen_files:
            print(f"[skip] {model_name}: no generator output")
            continue
        df = pd.read_parquet(gen_files[0])
        print(f"[start] {model_name}: {len(df)} rows")

        for i, row in df.iterrows():
            gen_gt = row.get("generation_gt")
            if isinstance(gen_gt, (list, tuple)) and len(gen_gt) > 0:
                gen_gt = gen_gt[0]
            answer = row.get("generated_texts") or row.get("answer") or ""
            if isinstance(answer, (list, tuple)) and len(answer) > 0:
                answer = answer[0]
            query = row.get("query", "")

            prompt = JUDGE_PROMPT.format(
                query=str(query)[:1500],
                gen_gt=str(gen_gt)[:1500],
                answer=str(answer)[:2000],
            )
            try:
                resp = client.chat.completions.create(
                    model=args.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200,
                    timeout=60,
                )
                text = resp.choices[0].message.content
                scores = parse_scores(text)
                if scores is None:
                    print(f"  [parse fail] {model_name} row {i}: {text[:100]}")
                    continue
                results.append({"model": model_name, "qid": row.get("qid"), **scores})
            except Exception as e:
                print(f"  [error] {model_name} row {i}: {e}")
                continue

        # 중간 저장
        if results:
            pd.DataFrame(results).to_csv(args.out_csv, index=False)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_csv, index=False)
    print(f"saved: {args.out_csv} ({len(out_df)} rows)")

    if len(out_df):
        agg = out_df.groupby("model")[["faithfulness", "relevance", "naturalness"]].mean().round(3)
        agg["overall"] = agg.mean(axis=1).round(3)
        agg = agg.sort_values("overall", ascending=False)
        agg.to_csv(args.avg_csv)
        print(f"saved: {args.avg_csv}")
        print(agg.to_string())


if __name__ == "__main__":
    main()
