#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_blind_labeling.py — KoRareGuard-T2I / Blind Labeling Exporter

목적:
  평가자 편향(Annotator Bias / Confirmation Bias)을 방지하기 위해 
  원본 데이터셋(prompts.csv, generation_results.csv) 및 생성 이미지를 
  익명화(ANON_001 ~ ANON_432)하여 평가 전용 블라인드 라벨링 패키지를 생성한다.

동작:
  1. benchmarks/prompts/prompts.csv 와 evaluation/generation/generation_results.csv 를 읽는다.
  2. generation_id 항목을 랜덤 셔플하여 ANON_001 ~ ANON_432 블라인드 ID를 부여한다.
  3. 비밀 매핑 파일(outputs/blind_mapping_key.json)을 생성하여 안전하게 저장한다.
  4. outputs/images/ 경로의 원본 PNG 이미지들을 outputs/images_blind/ 경로로 복사(ANON_xxx.png)한다.
  5. 평가용 시트(evaluation/generation/labeling_sheet_blind.csv)를 생성한다.

사용법:
  python evaluation/generation/prepare_blind_labeling.py [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

# 레포지토리 루트 경로 설정
REPO_DIR = Path(__file__).resolve().parents[2]
PROMPTS_CSV = REPO_DIR / "benchmarks" / "prompts" / "prompts.csv"
GEN_RESULTS_CSV = REPO_DIR / "evaluation" / "generation" / "generation_results.csv"
ORIGINAL_IMAGES_DIR = REPO_DIR / "outputs" / "images"

# 출력 파일 경로
MAPPING_KEY_JSON = REPO_DIR / "outputs" / "blind_mapping_key.json"
BLIND_IMAGES_DIR = REPO_DIR / "outputs" / "images_blind"
BLIND_SHEET_CSV = REPO_DIR / "evaluation" / "generation" / "labeling_sheet_blind.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def prepare_blind_dataset(seed: int = 42) -> None:
    print(f"[1/5] 원본 데이터 검증 및 로드...")
    if not PROMPTS_CSV.exists():
        raise FileNotFoundError(f"프롬프트 파일이 없습니다: {PROMPTS_CSV}")
    if not GEN_RESULTS_CSV.exists():
        raise FileNotFoundError(f"생성 결과 파일이 없습니다: {GEN_RESULTS_CSV}")
    if not ORIGINAL_IMAGES_DIR.exists():
        raise FileNotFoundError(f"이미지 디렉토리가 없습니다: {ORIGINAL_IMAGES_DIR}")

    prompts_list = read_csv(PROMPTS_CSV)
    gen_results = read_csv(GEN_RESULTS_CSV)

    prompt_map = {p["prompt_id"]: p for p in prompts_list}

    print(f"  - 로드된 생성 결과: {len(gen_results)}개")

    # [2/5] 무작위 셔플 및 블라인드 ID 할당
    random.seed(seed)
    gen_indices = list(range(len(gen_results)))
    random.shuffle(gen_indices)

    mapping_key: dict[str, dict[str, str]] = {}
    sheet_rows: list[dict[str, str]] = []

    print(f"[2/5] 익명화 ID (ANON_001 ~ ANON_{len(gen_results):03d}) 할당 중 (Seed: {seed})...")
    for new_idx, orig_idx in enumerate(gen_indices, start=1):
        blind_id = f"ANON_{new_idx:03d}"
        gen_item = gen_results[orig_idx]
        gen_id = gen_item["generation_id"]
        prompt_id = gen_item["prompt_id"]
        
        prompt_info = prompt_map.get(prompt_id, {})
        key_expr = prompt_info.get("key_expression", "")
        raw_prompt = prompt_info.get("raw_prompt", "")

        orig_img_name = f"{gen_id}.png"
        blind_img_name = f"{blind_id}.png"

        # 매핑 정보 구성
        mapping_key[blind_id] = {
            "generation_id": gen_id,
            "prompt_id": prompt_id,
            "key_expression": key_expr,
            "raw_prompt": raw_prompt,
            "original_image": orig_img_name,
            "blind_image": blind_img_name,
        }

        # 시트용 행 구성
        sheet_rows.append({
            "blind_id": blind_id,
            "image_file": blind_img_name,
            "key_expression": key_expr,
            "raw_prompt": raw_prompt,
            "concept_present_a1": "",
            "concept_present_a2": "",
            "concept_present_final": "",
            "image_safety_a1": "",
            "image_safety_a2": "",
            "image_safety_final": "",
            "disagreement_resolved_by": "",
            "notes": "",
        })

    # [3/5] 비밀 매핑 키 JSON 저장
    MAPPING_KEY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_KEY_JSON, "w", encoding="utf-8") as f:
        json.dump(mapping_key, f, ensure_ascii=False, indent=2)
    print(f"[3/5] 매핑 키 저장 완료: {MAPPING_KEY_JSON}")

    # [4/5] 블라인드 이미지 복사 (outputs/images_blind/)
    BLIND_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[4/5] 블라인드 이미지 복사 중 ({len(mapping_key)}개)...")
    copied_count = 0
    missing_count = 0

    for blind_id, info in mapping_key.items():
        src_img = ORIGINAL_IMAGES_DIR / info["original_image"]
        dst_img = BLIND_IMAGES_DIR / info["blind_image"]

        if src_img.exists():
            shutil.copyfile(src_img, dst_img)
            copied_count += 1
        else:
            print(f"  [경고] 이미지 원본 파일 없음: {src_img.name}")
            missing_count += 1

    print(f"  - 이미지 복사 완료: {copied_count}개 성공 / {missing_count}개 누락")

    # [5/5] 평가자용 블라인드 시트 CSV 저장
    BLIND_SHEET_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "blind_id", "image_file", "key_expression", "raw_prompt",
        "concept_present_a1", "concept_present_a2", "concept_present_final",
        "image_safety_a1", "image_safety_a2", "image_safety_final",
        "disagreement_resolved_by", "notes"
    ]
    with open(BLIND_SHEET_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sheet_rows)

    print(f"[5/5] 라벨링 평가 시트 생성 완료: {BLIND_SHEET_CSV}")
    print("\n" + "=" * 80)
    print(" [안내] 블라인드 평가 패키지 준비가 완료되었습니다.")
    print(f" 1. 평가용 이미지 경로: {BLIND_IMAGES_DIR}")
    print(f" 2. 라벨링 작성 시트: {BLIND_SHEET_CSV}")
    print(f" 3. 비밀 매핑 키 (내부용): {MAPPING_KEY_JSON}")
    print(" 평가 완료 후 `python evaluation/generation/process_blind_labels.py`를 실행하여")
    print(" 원본 image_labels.csv 파일로 자동 복원하세요.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="블라인드 라벨링 데이터셋 패키지 생성")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 셔플 시드 (기본값: 42)")
    args = parser.parse_args()

    prepare_blind_dataset(seed=args.seed)


if __name__ == "__main__":
    main()
