#!/usr/bin/env python3
"""
Cell 8의 run_single_test() 함수에 로깅 레벨 복구 추가
"""
import json
from pathlib import Path

notebook_path = Path("/workspace/jupyter_notebooks/surion_llm_isaacsim/jupyter_notebooks/phase6_comprehensive_100h_test2.ipynb")

# 노트북 로드
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 8 찾기 (run_single_test 함수가 있는 셀)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def run_single_test(' in source and 'models.load_reranker_model()' in source:
            print("✅ Found Cell 8 with run_single_test()")

            # 기존 코드에서 수정할 부분 찾기
            old_code = """        # Reranker 모델 강제 재로드
        print("🔄 Reloading reranker model...")
        models.load_reranker_model()
        print(f"   ✅ Reranker: {config.RERANKER_MODEL}")

        active_config = {"""

            new_code = """        # Reranker 모델 강제 재로드
        print("🔄 Reloading reranker model...")
        models.load_reranker_model()
        print(f"   ✅ Reranker: {config.RERANKER_MODEL}")

        # 🆕 로깅 레벨 복구 (transformers가 변경했을 수 있음)
        import logging
        logging.getLogger('app').setLevel(logging.INFO)
        logging.getLogger('app.services.rag_service').setLevel(logging.INFO)
        logging.getLogger('app.core.llm').setLevel(logging.INFO)

        active_config = {"""

            if old_code in source:
                new_source = source.replace(old_code, new_code)
                cell['source'] = new_source.split('\n')

                # 각 라인 끝에 '\n' 추가 (마지막 제외)
                for i in range(len(cell['source']) - 1):
                    if not cell['source'][i].endswith('\n'):
                        cell['source'][i] += '\n'

                print("✅ Added logging level recovery after reranker reload")
                break
            else:
                print("❌ Could not find exact match - trying alternative...")

                # 대안: load_reranker_model() 다음에 삽입
                if 'models.load_reranker_model()' in source:
                    lines = source.split('\n')
                    new_lines = []
                    for line in lines:
                        new_lines.append(line)
                        if 'print(f"   ✅ Reranker: {config.RERANKER_MODEL}")' in line:
                            # 다음 줄에 로깅 복구 코드 삽입
                            new_lines.append('')
                            new_lines.append('        # 🆕 로깅 레벨 복구 (transformers가 변경했을 수 있음)')
                            new_lines.append('        import logging')
                            new_lines.append("        logging.getLogger('app').setLevel(logging.INFO)")
                            new_lines.append("        logging.getLogger('app.services.rag_service').setLevel(logging.INFO)")
                            new_lines.append("        logging.getLogger('app.core.llm').setLevel(logging.INFO)")

                    new_source = '\n'.join(new_lines)
                    cell['source'] = new_source.split('\n')
                    for i in range(len(cell['source']) - 1):
                        if not cell['source'][i].endswith('\n'):
                            cell['source'][i] += '\n'

                    print("✅ Modified using alternative method")
                    break

# 백업
backup_path = notebook_path.with_name(notebook_path.stem + '.ipynb.backup_logging_fix')
import shutil
shutil.copy(notebook_path, backup_path)
print(f"✅ Backup created: {backup_path}")

# 저장
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✅ Modified notebook saved: {notebook_path}")
print("\n다음 단계:")
print("1. Kernel → Interrupt (현재 실행 중단)")
print("2. Kernel → Restart")
print("3. Cell 8 재실행 (함수 재정의)")
print("4. Cell 12부터 재실행 (Stage 1 시작)")
print("\n이제 INFO 로그가 정상적으로 출력됩니다!")
