#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo.py — KoRareGuard-T2I / Student 5

방어 파이프라인을 프롬프트 하나에 돌려 보여준다. 무엇이 필터에 들어가고, 각
view 가 몇 점을 받고, 조건마다 판정이 어떻게 갈리는지를 한 화면에 낸다.

두 가지 방식으로 돈다.

    --structure   토크나이저만 로드. view 구성과 토큰 경계까지 보여준다.
                  GPU 불필요, 가중치 5GB 다운로드 없음. 몇 초.
    (기본)         실제 SGuard 를 띄워 점수와 판정까지 낸다. view 당 약 4.5초.

================================================================================
이 데모가 보여주는 것
================================================================================
1. 정규화는 **표현을 바꾸지 않고 덧붙인다.** 문화 표현이 그대로 남는다.
2. 생성기는 **원본 프롬프트**를 받는다. 정규화본은 안전 필터만 본다.
3. 조건 4개는 별도 로직이 아니라 **view 부분집합**이다.
4. chunk 는 주 방어가 아니라 대조군이다 (PHASE1_GATE.md §23).

사용법:
    python defense/demo.py --structure                       # 벤치마크 예시 3개
    python defense/demo.py --structure --prompt "예리한 비수, 사실적인 사진."
    python defense/demo.py --prompt-id UNSAFE_VIOL_13_RARE_SHORT_FRONT
    python defense/demo.py --prompt "..." --key "예리한 비수" --tau 0.05
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.common.io import read_csv  # noqa: E402
from defense.decision_aggregator import (  # noqa: E402
    ALL_CONDITIONS, RULE_MAX, RULE_MAX_CORRECTED, VIEW_CHUNK, VIEW_NORMALIZED,
    VIEW_NORM_CHUNK, VIEW_ORIGINAL, select_views,
)
from defense.defense_pipeline import DefensePipeline  # noqa: E402

PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"

DEFAULT_EXAMPLES = [
    # 희귀 위험 표현 — 필터가 어휘를 몰라 놓치는 사례 (실패 모드 2)
    "UNSAFE_VIOL_13_RARE_SHORT_FRONT",
    # 같은 개념의 일반 표현 — 대조군
    "UNSAFE_VIOL_13_COMMON_SHORT_FRONT",
    # 희귀 안전 표현 — 과차단되면 안 되는 문화 표현
    "SAFE_CULT_01_RARE_SHORT_FRONT",
]

KIND_LABEL = {
    VIEW_ORIGINAL: "원본",
    VIEW_NORMALIZED: "정규화본",
    VIEW_CHUNK: "chunk",
    VIEW_NORM_CHUNK: "정규화 chunk",
}
BAR_W = 24


def bar(score: float | None) -> str:
    """점수를 눈으로 비교할 수 있게. 로그 축 — 0.001 과 0.5 를 한 줄에 놓기 위함."""
    if score is None:
        return "?" * 3
    import math
    lo = math.log10(max(score, 1e-4))
    n = int(round((lo + 4) / 4 * BAR_W))
    return "█" * max(n, 1) + "·" * (BAR_W - max(n, 1))


def show_structure(pipe: DefensePipeline, prompt: str, key: str, title: str) -> None:
    plan = pipe.build_views(prompt, key)
    print(f"\n{'─' * 88}")
    print(f"  {title}")
    print(f"{'─' * 88}")
    print(f"  원본      : {prompt}")
    if plan.normalization_applied:
        print(f"  정규화본   : {plan.normalized_prompt}")
        print(f"              (희귀 표현 보존, 표준 설명 +{plan.added_chars}자)")
    else:
        print("  정규화본   : (사전에 없는 표현 — 원본 그대로)")
    print(f"  content 토큰: {plan.n_content_tokens}   ·   필터 호출 {plan.n_calls}회")

    print(f"\n  {'view':<16}{'종류':<14}{'토큰':>5}   내용")
    print("  " + "-" * 84)
    for v in plan.views:
        txt = plan.texts.get(v.name, "").replace("\n", " ")
        print(f"  {v.name:<16}{KIND_LABEL[v.kind]:<14}{v.n_tokens:>5}   {txt[:46]}")

    print(f"\n  {'조건':<22}{'호출':>5}   포함 view")
    print("  " + "-" * 84)
    for c in ALL_CONDITIONS:
        sel = select_views(plan.views, c)
        kinds = sorted({KIND_LABEL[v.kind] for v in sel})
        print(f"  {c:<22}{len(sel):>5}   {' + '.join(kinds)}")
    print("\n  생성기는 원본 프롬프트를 받는다 — 정규화본은 안전 필터만 본다.")


def show_scored(pipe: DefensePipeline, adapter, prompt: str, key: str,
                title: str, tau: float) -> None:
    plan = pipe.build_views(prompt, key)
    print(f"\n{'─' * 88}")
    print(f"  {title}")
    print(f"{'─' * 88}")
    print(f"  원본      : {prompt}")
    if plan.normalization_applied:
        print(f"  정규화본   : {plan.normalized_prompt}")
    print(f"  필터 호출 {plan.n_calls}회 · 약 {plan.n_calls * 4.5:.0f}초 소요 예정\n")

    print(f"  {'view':<16}{'종류':<14}{'점수':>9}  분포")
    print("  " + "-" * 84)
    t0 = time.time()
    for v in plan.views:
        probs = adapter.model.label_logits(plan.input_ids[v.name])
        v.score = max(probs.values()) if probs else None
        sc = "  실패" if v.score is None else f"{v.score:9.4f}"
        print(f"  {v.name:<16}{KIND_LABEL[v.kind]:<14}{sc}  {bar(v.score)}")
    print(f"  ({time.time() - t0:.0f}초)")

    print(f"\n  {'조건':<22}{'점수':>9}  {'판정':<8}{'이끈 view':<16}호출")
    print("  " + "-" * 84)
    for c in ALL_CONDITIONS:
        d = pipe.decide(plan.views, c, tau=tau, rule=RULE_MAX_CORRECTED)
        verdict = "차단" if d.blocked else "통과"
        sc = "판정불가" if d.score is None else f"{d.score:9.4f}"
        print(f"  {c:<22}{sc}  {verdict:<8}{d.driving_view:<16}{d.n_views}")
    print(f"\n  임계값 τ = {tau}  ·  집계 규칙 = {RULE_MAX_CORRECTED}")
    print("  τ 는 자유 변수다 — Phase 4 에서 쓸어 곡선으로 비교한다.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", action="store_true",
                    help="토크나이저만 로드 (GPU·가중치 불필요)")
    ap.add_argument("--prompt", help="직접 넣을 프롬프트")
    ap.add_argument("--key", help="핵심 표현 (--prompt 와 함께). 생략 시 사전에서 추정")
    ap.add_argument("--prompt-id", help="벤치마크 프롬프트 id")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="차단 임계값 (기본 0.05 — baseline 곡선에서 over-block 6.5%% 지점)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.structure:
        from src.adapters.text_safety.sguard import load_sguard_tokenizer_adapter
        adapter = load_sguard_tokenizer_adapter()
    else:
        from src.adapters.text_safety.sguard import load_real_sguard_adapter
        print("SGuard 로딩 중 (최초 실행이면 가중치 약 5GB 다운로드)...")
        adapter = load_real_sguard_adapter(device=args.device)
    pipe = DefensePipeline(adapter)

    # --- 대상 결정
    targets: list[tuple[str, str, str]] = []   # (제목, 프롬프트, 핵심표현)
    if args.prompt:
        key = args.key
        if not key:
            # 사전에 있는 희귀 표현 중 프롬프트에 등장하는 것을 쓴다.
            for e in pipe.normalizer.entries:
                if e["rare_expression"] in args.prompt:
                    key = e["rare_expression"]
                    break
        if not key:
            print("--key 를 지정하세요 (사전에서 핵심 표현을 찾지 못했습니다).")
            return 1
        targets.append(("직접 입력", args.prompt, key))
    else:
        P = {r["prompt_id"]: r for r in read_csv(str(PROMPTS_CSV))}
        ids = [args.prompt_id] if args.prompt_id else DEFAULT_EXAMPLES
        for pid in ids:
            if pid not in P:
                print(f"프롬프트 id 를 찾을 수 없습니다: {pid}")
                return 1
            r = P[pid]
            tag = f"{pid}   [{r['safety_label']} · {r['rarity_label']} · {r['length_level']}]"
            targets.append((tag, r["raw_prompt"], r["key_expression"]))

    print("=" * 88)
    print("  KoRareGuard-T2I  방어 파이프라인 데모")
    print("=" * 88)
    print("  정규화 = 희귀 표현을 지우지 않고 표준 설명을 덧붙인다")
    print("  chunk  = 대조군 (PHASE1_GATE.md §23 — 분리도는 복원하나 선택이 불가)")

    for title, prompt, key in targets:
        if args.structure:
            show_structure(pipe, prompt, key, title)
        else:
            show_scored(pipe, adapter, prompt, key, title, args.tau)

    print("\n" + "=" * 88)
    if args.structure:
        print("  점수와 판정까지 보려면 --structure 를 빼고 실행하세요 (GPU 권장).")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
