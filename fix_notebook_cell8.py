#!/usr/bin/env python3
"""
Cell 8의 run_single_test() 함수에 reranker 강제 재로드 추가
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
        if 'def run_single_test(' in source and 'config._load_configs()' in source:
            print("✅ Found Cell 8 with run_single_test()")

            # 기존 코드에서 수정할 부분 찾기
            old_code = """        # in-memory Config 싱글톤 갱신 (파일 변경사항 반영)
        from app.config import config
        config._load_configs()

        active_config = {"""

            new_code = """        # in-memory Config 싱글톤 갱신 (파일 변경사항 반영)
        from app.config import config
        config._load_configs()

        # 🆕 핵심 수정: 전역 모델 강제 재로드
        from app.core import models

        # Reranker 모델 강제 재로드
        print("🔄 Reloading reranker model...")
        models.load_rerank_model()
        print(f"   ✅ Reranker: {config.RERANK_MODEL}")

        active_config = {"""

            if old_code in source:
                new_source = source.replace(old_code, new_code)

                # 로그 출력 문구도 수정
                new_source = new_source.replace(
                    'print("?? Active config (after reload):")',
                    'print("📋 Active config (after reload):")'
                )

                cell['source'] = new_source.split('\n')

                # 각 라인 끝에 '\n' 추가 (마지막 제외)
                for i in range(len(cell['source']) - 1):
                    if not cell['source'][i].endswith('\n'):
                        cell['source'][i] += '\n'

                print("✅ Modified run_single_test() to force reload reranker")
                break
            else:
                print("❌ Could not find exact match for old code")
                print("Trying alternative search...")

                # 대안: config._load_configs() 다음에 삽입
                if 'config._load_configs()' in source:
                    lines = source.split('\n')
                    new_lines = []
                    for i, line in enumerate(lines):
                        new_lines.append(line)
                        if 'config._load_configs()' in line:
                            # 다음 줄 삽입
                            new_lines.append('')
                            new_lines.append('        # 🆕 핵심 수정: 전역 모델 강제 재로드')
                            new_lines.append('        from app.core import models')
                            new_lines.append('        ')
                            new_lines.append('        # Reranker 모델 강제 재로드')
                            new_lines.append('        print("🔄 Reloading reranker model...")')
                            new_lines.append('        models.load_rerank_model()')
                            new_lines.append('        print(f"   ✅ Reranker: {config.RERANK_MODEL}")')

                    new_source = '\n'.join(new_lines)
                    new_source = new_source.replace(
                        'print("?? Active config (after reload):")',
                        'print("📋 Active config (after reload):")'
                    )

                    cell['source'] = new_source.split('\n')
                    for i in range(len(cell['source']) - 1):
                        if not cell['source'][i].endswith('\n'):
                            cell['source'][i] += '\n'

                    print("✅ Modified using alternative method")
                    break

# 수정된 노트북 저장
backup_path = notebook_path.with_suffix('.ipynb.backup_before_fix')
import shutil
shutil.copy(notebook_path, backup_path)
print(f"✅ Backup created: {backup_path}")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✅ Modified notebook saved: {notebook_path}")
print("\n다음 단계:")
print("1. 현재 실행 중인 노트북 중단: Kernel → Interrupt")
print("2. Kernel 재시작: Kernel → Restart")
print("3. Cell 8 재실행: run_single_test() 함수 재정의")
print("4. Cell 12부터 재실행: Stage 1 시작")
