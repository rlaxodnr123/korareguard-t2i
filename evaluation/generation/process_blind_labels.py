#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_blind_labels.py — KoRareGuard-T2I / Blind Labeling Importer & Restorer

목적:
  평가자가 작성 완료한 블라인드 라벨링 시트(labeling_sheet_blind.csv)를 읽어서,
  outputs/blind_mapping_key.json 의 매핑 정보를 이용해 본래의 generation_id 로 
  복원하고, 프로젝트 표준 evaluation/generation/image_labels.csv 를 생성한다.

동작:
  1. outputs/blind_mapping_key.json 과 labeling_sheet_blind.csv 로드
  2. ANON_xxx 블라인드 ID를 원본 generation_id 로 매핑
  3. concept_present 및 image_safety_label 자동 동기화 (a1==a2 일치 시 final 자동채움)
  4. evaluation/generation/image_labels.csv 출력
  5. analysis/validate_team_inputs.py 검증 실행

사용법:
  python evaluation/generation/process_blind_labels.py [--sheet evaluation/generation/labeling_sheet_blind.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

from src.common import schema  # noqa: E402

MAPPING_KEY_JSON = REPO_DIR / "outputs" / "blind_mapping_key.json"
DEFAULT_BLIND_SHEET = REPO_DIR / "evaluation" / "generation" / "labeling_sheet_blind.csv"
FINAL_IMAGE_LABELS_CSV = REPO_DIR / "evaluation" / "generation" / "image_labels.csv"
VALIDATE_SCRIPT = REPO_DIR / "analysis" / "validate_team_inputs.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clean_val(v: str) -> str:
    return str(v).strip().lower()


def process_blind_labels(sheet_path: Path) -> None:
    print(f"[1/4] 매핑 키 및 라벨 시트 로드...")
    if not MAPPING_KEY_JSON.exists():
        raise FileNotFoundError(f"매핑 키 파일이 없습니다: {MAPPING_KEY_JSON}\n"
                                f"먼저 prepare_blind_labeling.py 를 실행했는지 확인하세요.")
    if not sheet_path.exists():
        raise FileNotFoundError(f"라벨링 시트가 없습니다: {sheet_path}")

    with open(MAPPING_KEY_JSON, encoding="utf-8") as f:
        mapping_key: dict[str, dict[str, str]] = json.load(f)

    blind_rows = read_csv(sheet_path)
    print(f"  - 로드된 블라인드 라벨 행: {len(blind_rows)}개")
    print(f"  - 매핑 키 개수: {len(mapping_key)}개")

    final_rows: list[dict[str, str]] = []
    disagreements = 0
    empty_fields = 0

    print(f"[2/4] 원본 generation_id 복원 및 판정 검증 중...")
    for row in blind_rows:
        blind_id = clean_val(row.get("blind_id", ""))
        # 매핑 키 조회의 대소문자 허용
        key_match = None
        for k in mapping_key:
            if k.lower() == blind_id:
                key_match = mapping_key[k]
                break

        if not key_match:
            raise KeyError(f"매핑 키에서 알 수 없는 blind_id 발견: {row.get('blind_id')}")

        gen_id = key_match["generation_id"]

        cp_a1 = clean_val(row.get("concept_present_a1", ""))
        cp_a2 = clean_val(row.get("concept_present_a2", ""))
        cp_final = clean_val(row.get("concept_present_final", ""))

        is_a1 = clean_val(row.get("image_safety_a1", ""))
        is_a2 = clean_val(row.get("image_safety_a2", ""))
        is_final = clean_val(row.get("image_safety_final", ""))

        resolved_by = clean_val(row.get("disagreement_resolved_by", ""))

        # 비어있는 a1/a2 체크
        if not cp_a1 or not cp_a2 or not is_a1 or not is_a2:
            empty_fields += 1

        # concept_present final 결정
        if not cp_final:
            if cp_a1 and cp_a2 and cp_a1 == cp_a2:
                cp_final = cp_a1
            elif cp_a1 != cp_a2:
                disagreements += 1

        # image_safety final 결정
        if not is_final:
            if is_a1 and is_a2 and is_a1 == is_a2:
                is_final = is_a1
            elif is_a1 != is_a2:
                disagreements += 1

        # disagreement_resolved_by 자동 설정
        if not resolved_by:
            if cp_a1 == cp_a2 and is_a1 == is_a2 and cp_a1 and is_a1:
                resolved_by = "consensus"
            elif cp_a1 != cp_a2 or is_a1 != is_a2:
                resolved_by = "lead_review"

        final_rows.append({
            schema.ImgCols.GENERATION_ID: gen_id,
            schema.ImgCols.CONCEPT_PRESENT: cp_final,
            schema.ImgCols.IMAGE_SAFETY_LABEL: is_final,
            schema.ImgCols.CONCEPT_PRESENT_A1: cp_a1,
            schema.ImgCols.CONCEPT_PRESENT_A2: cp_a2,
            schema.ImgCols.CONCEPT_PRESENT_FINAL: cp_final,
            schema.ImgCols.IMAGE_SAFETY_A1: is_a1,
            schema.ImgCols.IMAGE_SAFETY_A2: is_a2,
            schema.ImgCols.IMAGE_SAFETY_FINAL: is_final,
            schema.ImgCols.DISAGREEMENT_RESOLVED_BY: resolved_by,
        })

    if empty_fields > 0:
        print(f"  [주의] 비어있는 a1/a2 라벨이 존재합니다 ({empty_fields}행).")
    if disagreements > 0:
        print(f"  [안내] a1 과 a2 간 불일치(Disagreement) 항목이 {disagreements}건 발견되었습니다.")

    # [3/4] 표준 image_labels.csv 출력
    FINAL_IMAGE_LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_IMAGE_LABELS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema.IMAGE_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"[3/4] 표준 image_labels.csv 저장 완료: {FINAL_IMAGE_LABELS_CSV}")

    # [4/4] validate_team_inputs.py 자동 실행 검증
    print(f"[4/4] 무결성 검증 스크립트 실행 중 (validate_team_inputs.py)...")
    if VALIDATE_SCRIPT.exists():
        res = subprocess.run(
            [sys.executable, str(VALIDATE_SCRIPT), "--labels", str(FINAL_IMAGE_LABELS_CSV)],
            capture_output=True,
            text=True,
        )
        print(res.stdout)
        if res.stderr:
            print("STDERR:", res.stderr)
    else:
        print("  - 검증 스크립트를 찾을 수 없어 건너뜁니다.")

    print("=" * 80)
    print(" [완료] 블라인드 라벨 복원 및 표준 CSV 생성이 정상 완료되었습니다!")
    print(f" 파일 경로: {FINAL_IMAGE_LABELS_CSV}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="블라인드 라벨 복원 및 표준 image_labels.csv 생성")
    parser.add_argument("--sheet", type=str, default=str(DEFAULT_BLIND_SHEET),
                        help="작성 완료된 블라인드 라벨 CSV 경로")
    args = parser.parse_args()

    process_blind_labels(Path(args.sheet))


if __name__ == "__main__":
    main()
