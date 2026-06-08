"""
AutoRAG multi-hop QA 생성 (two_hop_incremental + same-source chunk pair).
- Sample N pair → query 생성 → answer 생성 → filter → 저장
- LLM: vLLM (Qwen2.5-72B-AWQ TP=2) on port 8000
"""
import argparse
import uuid
import numpy as np
import pandas as pd

from autorag.data.qa.schema import Corpus
from autorag.data.qa.query.prompt import QUERY_GEN_PROMPT
from autorag.data.qa.generation_gt.llama_index_gen_gt import make_basic_gen_gt
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.filter.passage_dependency import passage_dependency_filter_llama_index
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike


KO_MULTIHOP_MESSAGES = [
    ChatMessage(role=MessageRole.SYSTEM,
                content="주어진 두 문서를 모두 참조해야만 답할 수 있는 멀티홉(multi-hop) 질문을 한국어로 생성하세요. 질문은 반드시 한국어로 작성해야 합니다."),
    ChatMessage(role=MessageRole.USER,
                content=("Document 1: 누에보 라레도 시는 멕시코의 타마울리파스 주에 위치해 있다.\n"
                         "Document 2: 시우다드 데포르티바(스포츠 시티)는 멕시코 누에보 라레도에 있는 종합 스포츠 단지로, "
                         "멕시칸 야구 리그의 테콜로테스 데 누에보 라레도 팀의 홈구장이다.")),
    ChatMessage(role=MessageRole.ASSISTANT,
                content=("Answer: 타마울리파스\n"
                         "One-hop question (using Document 1): 누에보 라레도 시는 멕시코의 어느 주에 위치해 있는가?\n"
                         "Two-hop question (using Document 2): 테콜로테스 데 누에보 라레도의 홈구장인 시우다드 데포르티바는 멕시코의 어느 주에 있는가?")),
]


async def two_hop_incremental_patched(row, llm, lang="ko"):
    """AutoRAG two_hop_incremental 버그 패치 — messages list copy + 한국어 예시."""
    if lang == "ko":
        messages = list(KO_MULTIHOP_MESSAGES)
    else:
        messages = list(QUERY_GEN_PROMPT["two_hop_incremental"][lang])
    passages = row["retrieval_gt_contents"]
    assert len(passages) >= 2
    context_str = f"Document 1: {passages[0][0]}\nDocument 2: {passages[1][0]}"
    user_prompt = f"{context_str}\n\nGenerated two-hop Question from two Documents:\n"
    messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))
    chat_response = await llm.achat(messages=messages)
    response = chat_response.message.content
    row["query"] = response.split(":")[-1].strip()
    return row


def random_two_hop_same_source(corpus_df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """같은 source 내 2개 child chunk pair sampling.
    retrieval_gt = [[c1], [c2]] (2개 outer group, AND 관계 = multi-hop)
    """
    child_df = corpus_df[corpus_df["doc_id"].str.startswith("c-")].copy()
    child_df["source"] = child_df["metadata"].apply(lambda m: m.get("source", ""))
    counts = child_df["source"].value_counts()
    eligible_sources = counts[counts >= 2].index.tolist()
    assert len(eligible_sources) > 0, "2개 이상 chunk 가진 source가 없습니다."

    rng = np.random.default_rng(random_state)
    rows = []
    for _ in range(n):
        src = eligible_sources[rng.integers(0, len(eligible_sources))]
        candidates = child_df[child_df["source"] == src]
        idx = rng.choice(len(candidates), size=2, replace=False)
        c1, c2 = candidates.iloc[idx[0]]["doc_id"], candidates.iloc[idx[1]]["doc_id"]
        rows.append({
            "qid": str(uuid.uuid4()),
            "retrieval_gt": [[c1], [c2]],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/jinwoolee/surion/autorag/data/corpus.parquet")
    ap.add_argument("--out", default="/home/jinwoolee/surion/autorag/data/qa_multihop.parquet")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n-sample", type=int, default=60, help="처음 샘플링할 pair 수 (필터 후 30개 목표)")
    ap.add_argument("--batch", type=int, default=4)
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
        max_tokens=1024,
        context_window=16384,
        temperature=0.0,
        timeout=180.0,
    )

    print(f"\n[1] Sample n={args.n_sample} (two-hop, same source)")
    qa = corpus.sample(random_two_hop_same_source, n=args.n_sample, random_state=42)
    print(f"  qa columns: {list(qa.data.columns)}, rows: {len(qa.data)}")
    print(f"  retrieval_gt sample: {qa.data['retrieval_gt'].iloc[0]}")

    print("\n[2] make_retrieval_gt_contents")
    qa = qa.make_retrieval_gt_contents()
    print(f"  qa columns: {list(qa.data.columns)}")
    print(f"  contents groups: {len(qa.data['retrieval_gt_contents'].iloc[0])}")

    print(f"\n[3] Query 생성 (two_hop_incremental_patched, lang={args.lang})")
    qa = qa.batch_apply(two_hop_incremental_patched, llm=llm, lang=args.lang, batch_size=args.batch)
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
        r = qa.data.iloc[0]
        print(f"\n--- 샘플 row ---")
        print(f"  qid: {r['qid']}")
        print(f"  query: {str(r['query'])[:200]}")
        print(f"  generation_gt: {str(r['generation_gt'])[:200]}")
        print(f"  retrieval_gt: {r['retrieval_gt']}")


if __name__ == "__main__":
    main()
