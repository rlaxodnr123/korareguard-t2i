#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_glossary.py — KoRareGuard-T2I / Student 5

`benchmarks/prompts/prompts.csv` 에서 concept 별 (희귀 표현, 일반 표현) 쌍을 뽑아
`defense/glossary.json` 을 만든다. semantic_normalizer 가 읽는 사전이다.

================================================================================
왜 주석문을 직접 쓰지 않는가
================================================================================
방어 1 은 "희귀 표현 + 의미를 명확히 하는 표준 설명" 형태다. 그 표준 설명을
내가 손으로 쓰면 두 가지 문제가 생긴다.

1. **위험 신호 주입.** 위험 개념의 주석문을 원본보다 강하게 쓰면 정규화가 아니라
   위험 신호를 새로 넣는 것이 된다. 반대로 약하게 쓰면 방어를 스스로 무력화한다.
   어느 쪽이든 "정규화가 효과가 있었다"는 결론이 내 문장 선택의 결과가 된다.
2. **검증되지 않은 의미 동일성.** 표현쌍이 실제로 같은 뜻인지 검토하는 것은
   학생1 의 필수 업무다. 내가 새 문장을 만들면 그 검토를 우회하게 된다.

그래서 주석문으로 **벤치마크의 일반 표현을 그대로** 쓴다. 이미 학생1 이 C2 기준
(concept 별 쌍 동일성)으로 검증했고, 24/24 통과가 기록돼 있다
(`analysis/truncation/benchmark_variant_comparison.md`).

결과적으로 이 사전에는 내 주관적 판단이 들어가지 않는다. prompts.csv 에서
기계적으로 유도되며, 재생성하면 항상 같은 파일이 나온다.

================================================================================
한계 — 반드시 논문에 적을 것
================================================================================
사전이 평가 대상 벤치마크 자체에서 유도되므로 **표현 커버리지가 100% 다.**
실제 배포 시스템은 사전에 없는 희귀 표현을 만난다. 따라서 이 사전으로 측정한
정규화 효과는 실현 가능한 값이 아니라 **상한(upper bound)** 이다.

이 한계는 Phase 4 에서 커버리지 축으로 정량화한다. 커버리지 c 에서는 프롬프트의
c 비율만 정규화되고 나머지는 원본 점수를 쓰므로, 원본·정규화 두 점수가 이미 있으면
**추가 추론 없이** 커버리지-효과 곡선을 그릴 수 있다.

사용법:
    python defense/build_glossary.py            # 생성 (기존 파일 있으면 차이만 보고)
    python defense/build_glossary.py --write    # 실제로 쓰기
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.common import schema  # noqa: E402
from src.common.io import read_csv  # noqa: E402

PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"
OUT_JSON = REPO / "defense" / "glossary.json"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    rows = read_csv(str(PROMPTS_CSV))

    pairs: dict[str, dict[str, str]] = {}
    for r in rows:
        cid = r["concept_id"]
        slot = pairs.setdefault(cid, {})
        slot[r["rarity_label"]] = r["key_expression"]
        slot["safety_label"] = r["safety_label"]
        slot["hazard_category"] = r["hazard_category"]

    entries = []
    problems = []
    for cid in sorted(pairs):
        v = pairs[cid]
        rare = v.get(schema.RARITY_RARE)
        common = v.get(schema.RARITY_COMMON)
        if not rare or not common:
            problems.append(f"{cid}: 쌍이 불완전 (rare={rare!r}, common={common!r})")
            continue
        if rare == common:
            problems.append(f"{cid}: 희귀형과 일반형이 동일 — 정규화할 것이 없음")
            continue
        entries.append({
            "concept_id": cid,
            "rare_expression": rare,
            "gloss": common,          # 주석문 = 벤치마크의 일반 표현 (학생1 C2 검증본)
            "safety_label": v["safety_label"],
            "hazard_category": v["hazard_category"],
            # 학생1 검토 전이다. 검토 후 이 필드를 채운다.
            "reviewed_by": None,
            "review_note": "",
        })

    return {
        "purpose": "semantic_normalizer 가 쓰는 희귀 표현 → 표준 설명 사전",
        "gloss_source": (
            "prompts.csv 의 동일 concept_id 일반 표현. 작성자가 새로 쓴 문장이 아니며, "
            "학생1 이 C2 기준으로 의미 동일성을 검증한 쌍이다 "
            "(analysis/truncation/benchmark_variant_comparison.md, 24/24)."
        ),
        "limitation": (
            "사전이 평가 대상 벤치마크에서 유도되므로 표현 커버리지가 100% 다. "
            "측정되는 정규화 효과는 실현값이 아니라 상한이다."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": "benchmarks/prompts/prompts.csv",
        "source_csv_sha256": sha256_of(PROMPTS_CSV),
        "n_entries": len(entries),
        "problems": problems,
        "entries": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="glossary.json 을 실제로 쓴다")
    args = ap.parse_args()

    if not PROMPTS_CSV.exists():
        print(f"프롬프트 파일이 없습니다: {PROMPTS_CSV}")
        return 1

    g = build()

    print("=" * 92)
    print(f"  희귀 표현 사전 — {g['n_entries']} 항목")
    print("=" * 92)
    print(f"  {'concept':<18} {'라벨':<7} {'희귀 표현':<20} → 주석문")
    print("-" * 92)
    for e in g["entries"]:
        print(f"  {e['concept_id']:<18} {e['safety_label']:<7} {e['rare_expression']:<20} → {e['gloss']}")
    if g["problems"]:
        print("-" * 92)
        print("  문제:")
        for p in g["problems"]:
            print(f"    - {p}")
    print("=" * 92)

    if not args.write:
        print("  (--write 를 주면 defense/glossary.json 에 씁니다)")
        return 0

    OUT_JSON.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  기록: {OUT_JSON}")
    print("  ! reviewed_by 가 전부 null 입니다 — 학생1 검토 후 채우세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
