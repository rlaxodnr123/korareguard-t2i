#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
defense_pipeline.py — KoRareGuard-T2I / Student 5

프롬프트 하나를 받아 안전 필터에 넣을 **view 집합**을 만든다. 점수 계산과 판정은
분리돼 있다 — 이 모듈은 "무엇을 필터에 먹일 것인가"만 정한다.

    build_views(prompt)  →  [원본, 정규화본, chunk…, 정규화 chunk…]
       ↓  Phase 3: view 마다 SGuard 를 한 번씩 (GPU, 1회 배치)
    decide(views, condition, tau, rule)  →  차단 / 통과   (CPU, 무한 반복)

이 분리가 핵심이다. tau 와 조건과 집계 규칙은 전부 사후 자유 변수이므로,
추론은 view 당 딱 한 번만 하고 분석 단계에서 얼마든지 다시 계산한다.

================================================================================
게이트 결과가 반영된 구성
================================================================================
PHASE1_GATE.md §23·§24 에 따라 **정규화가 주 방어이고 chunk 는 대조군**이다.

chunk 는 분리도를 실제로 복원하지만(ORACLE 이 baseline 을 최대 33pp 앞섬),
어느 chunk 가 위험을 담았는지를 점수만으로 고를 수 없다. 4개 크기 전부에서
단순 max 가 baseline 을 못 이겼다. 그래서 파이프라인에 남기되 주 방어로 두지
않는다 — ablation 4조건 중 `chunk_only` 와 `combined` 가 그 대조를 담당한다.

chunk budget 은 77 로 고정한다. 2차 게이트에서 32/48/64 를 모두 시험했고
77 이 그중 가장 나았다 (분리비 5.28 대 2.77/0.95/0.95).

================================================================================
비용 — cascade
================================================================================
전수 fan-out 은 프롬프트당 view 가 최대 (1 + 1 + k + k) 개다. 대부분의 프롬프트는
그럴 필요가 없다.

    1. 원본 1회
    2. 점수가 충분히 높으면            → 즉시 차단, 종료
    3. 점수가 충분히 낮고 프롬프트가 짧으면 → 통과, 종료
    4. 그 외에만 나머지 view 를 만든다

명세의 "긴 prompt 에서만 적용했을 때의 효과" 항목이 여기에 대응한다.
Phase 4 에서 cascade 문턱을 스윕해 호출 수 ↔ 성능 곡선을 그린다.

사용법:
    python defense/defense_pipeline.py --selftest
    python defense/defense_pipeline.py --demo      # 실제 토크나이저로 view 구성 보기
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from defense.decision_aggregator import (  # noqa: E402
    VIEW_CHUNK, VIEW_NORMALIZED, VIEW_NORM_CHUNK, VIEW_ORIGINAL,
    Decision, View, decide,
)
from defense.semantic_normalizer import SemanticNormalizer  # noqa: E402
from defense.token_chunk_checker import (  # noqa: E402
    DEFAULT_BUDGET, DEFAULT_STRIDE, build_chunks,
)

# 2차 게이트에서 4개 크기를 비교해 확정. 32/48/64 는 모두 더 나빴다.
PIPELINE_BUDGET = DEFAULT_BUDGET      # 77
PIPELINE_STRIDE = DEFAULT_STRIDE      # 62

# cascade 기본 문턱. Phase 4 에서 스윕한다.
CASCADE_HIGH = 0.5      # 원본 점수가 이보다 높으면 더 볼 것 없이 차단
CASCADE_LOW = 0.001     # 이보다 낮고 짧으면 통과
CASCADE_SHORT_TOKENS = 77   # "짧다"의 기준 = chunk 하나에 다 들어감


@dataclass
class ViewPlan:
    """한 프롬프트에 대해 만들어진 view 목록과 그 입력들.

    input_ids 는 모델에 그대로 먹이는 토큰열이다. 문자열을 재토큰화하지 않는다
    (sguard.py #3 과 같은 이유).
    """
    prompt_id: str = ""
    views: list[View] = field(default_factory=list)
    input_ids: dict[str, list[int]] = field(default_factory=dict)   # view 이름 → 토큰열
    texts: dict[str, str] = field(default_factory=dict)             # 표시/로그 전용
    normalized_prompt: str = ""
    normalization_applied: bool = False
    added_chars: int = 0
    n_content_tokens: int = 0

    @property
    def n_calls(self) -> int:
        return len(self.views)


class DefensePipeline:
    def __init__(self, adapter, normalizer: SemanticNormalizer | None = None,
                 budget: int = PIPELINE_BUDGET, stride: int = PIPELINE_STRIDE):
        self.adapter = adapter
        self.normalizer = normalizer or SemanticNormalizer.from_json()
        self.budget = budget
        self.stride = stride

    # ------------------------------------------------------------ view 구성
    def build_views(self, prompt: str, key_expression: str,
                    prompt_id: str = "",
                    include: tuple[str, ...] = (VIEW_ORIGINAL, VIEW_NORMALIZED,
                                                VIEW_CHUNK, VIEW_NORM_CHUNK)) -> ViewPlan:
        """필터에 먹일 view 를 전부 만든다. 점수는 아직 매기지 않는다.

        include 를 좁히면 그만큼만 만든다 — cascade 나 조건별 부분 실행용이다.
        단, ablation 4조건은 여기서 거르지 않고 **전부 만든 뒤 사후에 고른다.**
        조건마다 따로 추론하면 view 가 겹쳐 중복 호출이 된다.
        """
        plan = ViewPlan(prompt_id=prompt_id)

        enc = self.adapter.tok.encode_chat_template(prompt_text=prompt, response_text="")
        plan.n_content_tokens = len(enc.content_ids)

        if VIEW_ORIGINAL in include:
            ids = enc.prefix_ids + list(enc.content_ids) + enc.suffix_ids
            plan.views.append(View(VIEW_ORIGINAL, "original", n_tokens=len(enc.content_ids)))
            plan.input_ids["original"] = ids
            plan.texts["original"] = prompt

        nr = self.normalizer.normalize(prompt)
        plan.normalized_prompt = nr.normalized
        plan.normalization_applied = nr.applied
        plan.added_chars = nr.added_chars

        need_norm_views = VIEW_NORMALIZED in include or VIEW_NORM_CHUNK in include
        nenc = None
        if need_norm_views and nr.applied:
            nenc = self.adapter.tok.encode_chat_template(
                prompt_text=nr.normalized, response_text="")

        if VIEW_NORMALIZED in include and nr.applied:
            # 정규화가 적용되지 않았으면 원본과 같은 입력이므로 view 를 만들지 않는다.
            # 만들면 같은 추론을 두 번 하게 되고, 일반 표현 프롬프트에서 호출 수가
            # 실제보다 부풀려져 비용 비교가 왜곡된다.
            ids = nenc.prefix_ids + list(nenc.content_ids) + nenc.suffix_ids
            plan.views.append(View(VIEW_NORMALIZED, "normalized",
                                   n_tokens=len(nenc.content_ids)))
            plan.input_ids["normalized"] = ids
            plan.texts["normalized"] = nr.normalized

        if VIEW_CHUNK in include:
            for c in build_chunks(self.adapter, prompt, key_expression,
                                  budget=self.budget, stride=self.stride):
                name = f"chunk:{c.index}"
                plan.views.append(View(VIEW_CHUNK, name, n_tokens=c.n_tokens))
                plan.input_ids[name] = c.input_ids
                plan.texts[name] = c.text

        if VIEW_NORM_CHUNK in include and nr.applied:
            for c in build_chunks(self.adapter, nr.normalized, key_expression,
                                  budget=self.budget, stride=self.stride):
                name = f"norm_chunk:{c.index}"
                plan.views.append(View(VIEW_NORM_CHUNK, name, n_tokens=c.n_tokens))
                plan.input_ids[name] = c.input_ids
                plan.texts[name] = c.text

        return plan

    # ------------------------------------------------------------ cascade
    def cascade_plan(self, prompt: str, key_expression: str, original_score: float,
                     prompt_id: str = "",
                     high: float = CASCADE_HIGH, low: float = CASCADE_LOW,
                     short_tokens: int = CASCADE_SHORT_TOKENS) -> tuple[ViewPlan | None, str]:
        """원본 점수를 보고 나머지 view 를 만들지 말지 정한다.

        Returns:
            (추가로 만들 ViewPlan 또는 None, 사유 문자열)
            None 이면 원본 1회로 판정이 끝났다는 뜻이다.
        """
        if original_score > high:
            return None, "원본 점수가 높음 — 추가 검사 없이 차단"
        enc = self.adapter.tok.encode_chat_template(prompt_text=prompt, response_text="")
        n = len(enc.content_ids)
        if original_score < low and n <= short_tokens:
            return None, "원본 점수가 낮고 프롬프트가 짧음 — 추가 검사 불필요"
        plan = self.build_views(prompt, key_expression, prompt_id,
                                include=(VIEW_NORMALIZED, VIEW_CHUNK, VIEW_NORM_CHUNK))
        return plan, f"fan-out {plan.n_calls}회 (content {n} 토큰)"

    # ------------------------------------------------------------ 판정
    @staticmethod
    def decide(views: list[View], condition: str, tau: float, rule: str | None = None) -> Decision:
        """점수가 채워진 view 로 판정한다. 얇은 위임 — 규칙은 aggregator 가 갖는다."""
        from defense.decision_aggregator import RULE_MAX_CORRECTED
        return decide(views, condition, tau, rule or RULE_MAX_CORRECTED)


# ================================================================ self-test
def _selftest() -> int:
    from src.common.io import read_csv
    from src.adapters.text_safety.sguard import load_sguard_tokenizer_adapter
    from defense.decision_aggregator import (
        COND_BASELINE, COND_CHUNK_ONLY, COND_COMBINED, COND_NORM_ONLY, select_views,
    )

    PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
    res: list[bool] = []

    def check(name, cond, detail=""):
        res.append(bool(cond))
        print(f"  [{PASS if cond else FAIL}] {name}" + (f"  — {detail}" if detail else ""))

    print("=" * 78)
    print("  defense_pipeline  (실제 토크나이저, 모델 가중치 없음)")
    print("=" * 78)

    ad = load_sguard_tokenizer_adapter()
    pipe = DefensePipeline(ad)
    P = {r["prompt_id"]: r for r in read_csv(str(REPO / "benchmarks" / "prompts" / "prompts.csv"))}

    # --- 희귀 × 긴 프롬프트: 네 종류가 전부 나와야 한다
    r = P["UNSAFE_VIOL_13_RARE_OVER_LIMIT_BACK"]
    plan = pipe.build_views(r["raw_prompt"], r["key_expression"], r["prompt_id"])
    kinds = {v.kind for v in plan.views}
    check("희귀·장문은 view 4종류 전부", kinds == {VIEW_ORIGINAL, VIEW_NORMALIZED,
                                             VIEW_CHUNK, VIEW_NORM_CHUNK}, str(sorted(kinds)))
    check("정규화 적용됨", plan.normalization_applied and plan.added_chars > 0,
          f"+{plan.added_chars}자")
    check("모든 view 가 input_ids 를 가짐",
          all(v.name in plan.input_ids for v in plan.views), f"{plan.n_calls} view")
    from src.common import config
    check("모든 입력이 template overhead 보존",
          all(len(ids) >= config.SGUARD_TEMPLATE_OVERHEAD_TOKENS
              for ids in plan.input_ids.values()))
    check("정규화본이 원본보다 chunk 가 많거나 같음",
          sum(1 for v in plan.views if v.kind == VIEW_NORM_CHUNK)
          >= sum(1 for v in plan.views if v.kind == VIEW_CHUNK))

    # --- 일반 표현: 정규화 view 가 생기면 안 된다 (같은 입력 중복 추론 방지)
    r2 = P["UNSAFE_VIOL_13_COMMON_OVER_LIMIT_BACK"]
    plan2 = pipe.build_views(r2["raw_prompt"], r2["key_expression"], r2["prompt_id"])
    check("일반 표현엔 정규화 view 없음",
          not any(v.kind in (VIEW_NORMALIZED, VIEW_NORM_CHUNK) for v in plan2.views),
          f"{plan2.n_calls} view")
    check("일반 표현도 원본+chunk 는 있음",
          {v.kind for v in plan2.views} == {VIEW_ORIGINAL, VIEW_CHUNK})

    # --- 짧은 프롬프트: chunk 가 1개 (원본과 같은 범위)
    r3 = P["SAFE_CULT_01_RARE_SHORT_FRONT"]
    plan3 = pipe.build_views(r3["raw_prompt"], r3["key_expression"], r3["prompt_id"])
    n_ch = sum(1 for v in plan3.views if v.kind == VIEW_CHUNK)
    check("짧은 프롬프트는 chunk 1개", n_ch == 1, f"{n_ch}개")

    # --- 조건별 view 선택이 실제 plan 에서 동작하는가
    got = {c: len(select_views(plan.views, c))
           for c in (COND_BASELINE, COND_NORM_ONLY, COND_CHUNK_ONLY, COND_COMBINED)}
    check("baseline 은 1 view", got[COND_BASELINE] == 1)
    check("정규화만은 2 view", got[COND_NORM_ONLY] == 2)
    check("combined 가 가장 많음", got[COND_COMBINED] == plan.n_calls, str(got))

    # --- cascade
    p, why = pipe.cascade_plan(r["raw_prompt"], r["key_expression"], original_score=0.9)
    check("원본 점수 높으면 fan-out 안 함", p is None, why)
    p, why = pipe.cascade_plan(r3["raw_prompt"], r3["key_expression"], original_score=0.0001)
    check("짧고 점수 낮으면 fan-out 안 함", p is None, why)
    p, why = pipe.cascade_plan(r["raw_prompt"], r["key_expression"], original_score=0.01)
    check("애매하면 fan-out 함", p is not None and p.n_calls > 0, why)

    # --- 판정 위임
    for v, s in zip(plan.views, [0.01, 0.72] + [0.003] * (plan.n_calls - 2)):
        v.score = s
    d = pipe.decide(plan.views, COND_NORM_ONLY, tau=0.5)
    check("정규화 조건에서 정규화본이 판정을 이끔",
          d.blocked and d.driving_view == "normalized", f"{d.score:.3f} / {d.driving_view}")
    d0 = pipe.decide(plan.views, COND_BASELINE, tau=0.5)
    check("baseline 조건에서는 통과", not d0.blocked and d0.score == 0.01)

    n = sum(res)
    print("=" * 78)
    print(f"  {n}/{len(res)} CHECK 통과")
    print("=" * 78)
    return 0 if n == len(res) else 1


def _demo() -> int:
    from src.common.io import read_csv
    from src.adapters.text_safety.sguard import load_sguard_tokenizer_adapter
    ad = load_sguard_tokenizer_adapter()
    pipe = DefensePipeline(ad)
    P = {r["prompt_id"]: r for r in read_csv(str(REPO / "benchmarks" / "prompts" / "prompts.csv"))}

    print("=" * 92)
    print("  프롬프트별 view 구성 — Phase 3 배치 규모의 근거")
    print("=" * 92)
    print(f"  {'prompt_id':<44}{'토큰':>6}{'view':>6}   구성")
    print("-" * 92)
    total = 0
    for pid in ["SAFE_CULT_01_COMMON_SHORT_FRONT", "SAFE_CULT_01_RARE_SHORT_FRONT",
                "UNSAFE_VIOL_13_COMMON_NEAR_LIMIT_BACK", "UNSAFE_VIOL_13_RARE_NEAR_LIMIT_BACK",
                "UNSAFE_VIOL_13_COMMON_OVER_LIMIT_BACK", "UNSAFE_VIOL_13_RARE_OVER_LIMIT_BACK"]:
        r = P[pid]
        pl = pipe.build_views(r["raw_prompt"], r["key_expression"], pid)
        from collections import Counter
        c = Counter(v.kind for v in pl.views)
        comp = " ".join(f"{k}×{n}" for k, n in sorted(c.items()))
        print(f"  {pid:<44}{pl.n_content_tokens:>6}{pl.n_calls:>6}   {comp}")

    print("-" * 92)
    for r in P.values():
        pl = pipe.build_views(r["raw_prompt"], r["key_expression"])
        total += pl.n_calls
    print(f"  432 프롬프트 전체 view 합계: {total:,}회")
    print(f"  4.49초/회 기준 예상 소요:    {total * 4.49 / 3600:.1f}시간")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.demo:
        sys.exit(_demo())
    print(__doc__)
