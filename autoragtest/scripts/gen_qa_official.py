"""
AutoRAG 정식 QA 생성 — Raw 없이 기존 corpus.parquet 에서.
- Sample N 청크 → query 생성 → answer 생성 → filter → 저장
- LLM: vLLM (Qwen2.5-32B-AWQ) on port 8000
"""
import argparse
import pandas as pd

from autorag.data.qa.schema import Corpus
from autorag.data.qa.sample import random_single_hop
from autorag.data.qa.query.llama_gen_query import factoid_query_gen, concept_completion_query_gen
from autorag.data.qa.generation_gt.llama_index_gen_gt import make_basic_gen_gt
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.filter.passage_dependency import passage_dependency_filter_llama_index
from llama_index.llms.openai_like import OpenAILike


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/jinwoolee/surion/autorag/data/corpus.parquet")
    ap.add_argument("--out", default="/home/jinwoolee/surion/autorag/data/qa_v2.parquet")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n-sample", type=int, default=100, help="처음 샘플링할 청크 수")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lang", default="ko")
    args = ap.parse_args()

    corpus_df = pd.read_parquet(args.corpus)
    print(f"corpus: {corpus_df.shape}")
    corpus = Corpus(corpus_df)

    llm = OpenAILike(
        model=args.llm_model,
        api_base=f"http://127.0.0.1:{args.port}/v1",
        api_key="EMPTY",
        is_chat_model=True,
        max_tokens=512,
        context_window=8192,
        temperature=0.0,
        timeout=60.0,
    )

    print(f"\n[1] Sample n={args.n_sample} (single-hop)")
    qa = corpus.sample(random_single_hop, n=args.n_sample, random_state=42)
    print(f"  qa columns: {list(qa.data.columns)}, rows: {len(qa.data)}")

    print("\n[2] make_retrieval_gt_contents")
    qa = qa.make_retrieval_gt_contents()
    print(f"  qa columns: {list(qa.data.columns)}")

    print("\n[3] Query 생성 (factoid_query_gen, lang={})".format(args.lang))
    qa = qa.batch_apply(factoid_query_gen, llm=llm, lang=args.lang, batch_size=args.batch)
    sample_q = qa.data["query"].iloc[0] if len(qa.data) else ""
    print(f"  생성된 query 샘플: {str(sample_q)[:200]}")

    print("\n[4] generation_gt 생성 (make_basic_gen_gt)")
    qa = qa.batch_apply(make_basic_gen_gt, llm=llm, lang=args.lang, batch_size=args.batch)
    sample_a = qa.data["generation_gt"].iloc[0] if len(qa.data) else ""
    print(f"  생성된 answer 샘플: {str(sample_a)[:200]}")

    print(f"\n[5] Filter — dontknow_rule_based (lang={args.lang})")
    before = len(qa.data)
    qa = qa.filter(dontknow_filter_rule_based, lang=args.lang)
    print(f"  {before} → {len(qa.data)} (rule_based dontknow 제거)")

    print(f"\n[6] Filter — passage_dependency (LLM)")
    before = len(qa.data)
    qa = qa.batch_filter(passage_dependency_filter_llama_index, llm=llm, lang=args.lang, batch_size=args.batch)
    print(f"  {before} → {len(qa.data)} (passage_dependent 만 남김)")

    print(f"\n[7] 저장 → {args.out}")
    qa.data.reset_index(drop=True).to_parquet(args.out)
    print(f"  최종 row: {len(qa.data)}")
    print(f"  컬럼: {list(qa.data.columns)}")
    if len(qa.data):
        print(f"\n--- 샘플 row ---")
        r = qa.data.iloc[0]
        print(f"  qid: {r['qid']}")
        print(f"  query: {str(r['query'])[:200]}")
        print(f"  generation_gt: {str(r['generation_gt'])[:200]}")
        print(f"  retrieval_gt: {r['retrieval_gt']}")

if __name__ == "__main__":
    main()
