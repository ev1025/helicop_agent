# ============================================================
# Phase 6 Main Test Loop - FIXED VERSION (Cell 14 교체용)
# ============================================================
# 이 코드를 phase6_best_combinations_optimization.ipynb의 Cell 14에 복사하세요
# ============================================================

print(f"\n{'='*80}")
print(f"Starting Phase 6 Tests")
print(f"{'='*80}")
print(f"Total configs: {len(phase6_configs)}")
print(f"Expected time: ~{len(phase6_configs) * 20 / 60:.1f} hours")
print(f"Results dir: {results_dir}")
print(f"{'='*80}\n")

results_summary = []

for idx, config_item in enumerate(phase6_configs, 1):
    print("=" * 80)
    print(f"[{idx}/{len(phase6_configs)}] {config_item['name']}")
    print(f"Priority: {config_item['priority']}")
    print(f"Description: {config_item['description']}")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:
        # 1. Config 적용 (매번 backup에서 로드)
        apply_config(config_item)

        # 2. 벤치마크 실행
        print("\n🔄 Running benchmark...")
        result_path = run_bench(
            cases_path=DEFAULT_CASES_PATH,
            limit=None,
            use_llm=True,
            max_turns=5,
            verbose=False
        )

        # ============================================================
        # 🔍 Rule 3: Config/Model 일치성 검증 (추가됨!)
        # ============================================================
        print("\n🔍 Validating config/model consistency...")

        with open(result_path, 'r', encoding='utf-8') as f:
            bench_data = json.load(f)

        validation_passed = True
        validation_errors = []

        model_info = bench_data.get('model_info', {})
        rag_info = bench_data.get('config_info', {}).get('rag', {})

        # Rule 3-1: Embedding model 검증
        config_embedding = config_item['embedding_model']
        actual_embedding = model_info.get('actual_embedding_model', 'N/A')
        if config_embedding != actual_embedding:
            validation_passed = False
            validation_errors.append(f"Embedding mismatch: Expected {config_embedding}, Got {actual_embedding}")

        # Rule 3-2: Reranker model 검증
        config_reranker = config_item.get('reranker_model', None)
        actual_reranker = model_info.get('actual_reranker_model', None)
        if config_item.get('reranker_model'):
            if config_reranker != actual_reranker:
                validation_passed = False
                validation_errors.append(f"Reranker mismatch: Expected {config_reranker}, Got {actual_reranker}")
        else:
            if actual_reranker is not None:
                validation_passed = False
                validation_errors.append(f"Reranker should be None, Got {actual_reranker}")

        # Rule 3-3: RAG config 검증
        expected_top_k = PHASE2_BEST_RETRIEVAL['vector_search_top_k']
        actual_top_k = rag_info.get('vector_search_top_k')
        if expected_top_k != actual_top_k:
            validation_passed = False
            validation_errors.append(f"vector_search_top_k mismatch: Expected {expected_top_k}, Got {actual_top_k}")

        if config_item.get('reranker_model'):
            expected_rerank_k = config_item.get('rerank_top_k', PHASE2_BEST_RETRIEVAL['rerank_top_k'])
            actual_rerank_k = rag_info.get('rerank_top_k')
            if expected_rerank_k != actual_rerank_k:
                validation_passed = False
                validation_errors.append(f"rerank_top_k mismatch: Expected {expected_rerank_k}, Got {actual_rerank_k}")

        expected_hybrid = config_item.get('use_hybrid', False)
        actual_hybrid = rag_info.get('use_hybrid_search', False)
        if expected_hybrid != actual_hybrid:
            validation_passed = False
            validation_errors.append(f"use_hybrid mismatch: Expected {expected_hybrid}, Got {actual_hybrid}")

        if expected_hybrid:
            expected_weight = config_item.get('hybrid_bm25_weight', 0.5)
            actual_weight = rag_info.get('hybrid_bm25_weight')
            if abs(expected_weight - actual_weight) > 0.01:
                validation_passed = False
                validation_errors.append(f"hybrid_bm25_weight mismatch: Expected {expected_weight}, Got {actual_weight}")

        # Rule 3-4: Prompt variant 검증 (간소화)
        if config_item.get('prompt_variant'):
            actual_prompt = bench_data.get('config_info', {}).get('prompts', {}).get('rag_answer_prompt', '')
            if not actual_prompt:
                validation_passed = False
                validation_errors.append(f"Prompt variant {config_item['prompt_variant']} not applied")

        # 검증 실패 시 즉시 중단
        if not validation_passed:
            print(f"\n{'🛑'*40}")
            print(f"🛑 CRITICAL: Config/Model validation FAILED!")
            for error in validation_errors:
                print(f"🛑 {error}")
            print(f"🛑 Stopping immediately to prevent wasted time!")
            print(f"{'🛑'*40}\n")

            # 에러 정보를 summary에 저장
            summary = {
                'idx': idx,
                'name': config_item["name"],
                'priority': config_item["priority"],
                'description': config_item["description"],
                'error': f"Validation failed: {'; '.join(validation_errors)}",
                'duration_min': (time.time() - start_time) / 60
            }
            results_summary.append(summary)

            break  # 즉시 중단!

        print(f"✅ Validation passed!")
        # ============================================================

        # 3. 평가
        evaluation = evaluate_phase2_bench(Path(result_path))

        # 4. 결과 저장
        result_file = results_dir / f"{idx:02d}_{config_item['name']}.json"
        result_data = {
            'index': idx,
            'config': config_item,
            'evaluation': evaluation,
            'bench_file': str(result_path),
            'duration_sec': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        # 5. 요약 저장
        summary = {
            'idx': idx,
            'name': config_item["name"],
            'priority': config_item["priority"],
            'embedding_model': config_item["embedding_model"],
            'reranker_model': config_item.get("reranker_model", ""),
            'rerank_top_k': config_item.get("rerank_top_k", ""),
            'use_hybrid': config_item.get("use_hybrid", False),
            'hybrid_bm25_weight': config_item.get("hybrid_bm25_weight", ""),
            'prompt_variant': config_item.get("prompt_variant", ""),
            'description': config_item["description"],
            'comprehensive': evaluation['avg_scores']['comprehensive'],
            'rag_quality': evaluation['avg_scores']['rag_quality'],
            'exact_repetition': evaluation['avg_scores']['exact_repetition'],
            'semantic_redundancy': evaluation['avg_scores']['semantic_redundancy'],
            'verbosity': evaluation['avg_scores']['verbosity'],
            'grammatical_particles': evaluation['avg_scores']['grammatical_particles'],
            'duration_min': (time.time() - start_time) / 60
        }
        results_summary.append(summary)

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"\n✅ Complete! Comprehensive: {summary['comprehensive']:.2f}, RAG: {summary['rag_quality']:.2f}")
        print(f"   Duration: {summary['duration_min']:.1f}min")
        print(f"   Result saved: {result_file.name}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        summary = {
            'idx': idx,
            'name': config_item["name"],
            'priority': config_item["priority"],
            'description': config_item["description"],
            'error': str(e),
            'duration_min': (time.time() - start_time) / 60
        }
        results_summary.append(summary)

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\nProgress: {idx}/{len(phase6_configs)} ({idx/len(phase6_configs)*100:.1f}%)")
    remaining_h = (len(phase6_configs)-idx) * 20 / 60
    print(f"Estimated remaining: {remaining_h:.1f}h\n")

print("\n" + "=" * 80)
print("All Phase 6 tests complete!")
print("=" * 80)
