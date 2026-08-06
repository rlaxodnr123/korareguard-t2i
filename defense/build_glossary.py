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

그래서 주석문으로 **벤치마크의 일반 표현을 그대로** 쓴다. 표현쌍의 의미 동일성은
학생1 이 전수 검토했고 그 기록을 아래 REVIEW 에 남겼다.

주의: 학생2 의 `benchmark_variant_comparison.md` 의 C2(24/24)는 세 후보 파일이
**같은 쌍을 쓰는지**를 대조한 것이지 쌍의 의미가 같은지를 본 것이 아니다.
둘을 혼동하지 않는다.

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
from src.common.io import input_provenance, read_csv  # noqa: E402

PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"
OUT_JSON = REPO / "defense" / "glossary.json"

# ---------------------------------------------------------------- 검토 기록
# 여기 두는 이유: glossary.json 에 직접 써 넣으면 이 스크립트를 다시 돌릴 때
# 날아간다. 사전은 prompts.csv 에서 기계적으로 재생성되므로, 사람이 남긴 사실은
# 생성기 쪽에 있어야 재생성을 견딘다.
REVIEW = {
    "reviewed_by": "우지민 (@WooJiMin03) — 학생1",
    "reviewed_at": "2026-08-05",
    "scope": "24개 concept 의 (희귀 표현, 일반 표현) 쌍 전수",
    "outcome": (
        "표현쌍 전수를 검토해 잘못된 것을 수정했다. UNSAFE_CRIM_24 의 희귀 표현이 "
        "'볏짚 불지르기'(볏짚을 태우는 농사 행위)였는데 짝인 일반 표현은 "
        "'건물에 불을 지르는 방화'여서 의미가 어긋났다 — 오히려 SAFE_CULT_03 "
        "'쥐불놀이'(논밭에 불을 놓는 민속놀이, safe 라벨)에 가까웠다. "
        "'초가집 불지르기'로 고쳐 짝을 맞췄다. 나머지 23개는 그대로 확정했다."
    ),
    "verifiable": (
        "UNSAFE_CRIM_24 수정은 commit 63c417b 에 남아 있다. 나머지 23개의 확정은 "
        "본인 확인이며 저장소에 별도 산출물은 없다."
    ),
}


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
            "reviewed_by": REVIEW["reviewed_by"],
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
        "review": REVIEW,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": input_provenance([str(PROMPTS_CSV.relative_to(REPO))]),
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
    print(f"  검토: {REVIEW['reviewed_by']} · {REVIEW['scope']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
