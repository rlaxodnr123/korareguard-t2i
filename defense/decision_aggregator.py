#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decision_aggregator.py — KoRareGuard-T2I / Student 5

여러 view(원본 / 정규화본 / chunk)의 안전 점수를 하나의 판정으로 모은다.

================================================================================
게이트가 바꾼 설계 — view 를 한 통에 넣고 max 하면 안 된다
================================================================================
PHASE 1 GATE §22 의 결론: SGuard 는 무해한 한국어 조각의 약 5% 에 산발적으로 높은
점수를 낸다. max 는 뽑기를 k 번 하는 연산이므로, 회당 오탐률이 0 이 아니면 k 가
커질수록 반드시 오염된다. chunk 를 잘게 자를수록 k 와 회당 오탐률이 함께 올라
곡선이 단조로 나빠졌다 (77 → 32 에서 under-blocking 63.9% → 83.3%).

그런데 view 마다 k 가 다르다.

    원본        1개    뽑기 없음
    정규화본     1개    뽑기 없음
    chunk      7~16개  뽑기 있음   ← 오염원

전부 한 리스트에 넣고 max 를 취하면, 뽑기가 없는 view 까지 chunk 의 오탐에
끌려간다. 그래서 **view 종류(family)별로 먼저 모으고, 그 다음에 합친다.**

    score = max(
        max(뽑기 없는 view 들),                    ← 원본 · 정규화본
        보정(max(chunk view 들), k=chunk 수)        ← 뽑기 보정 후 합류
    )

k 보정은 게이트 표본(72개, 전부 k=7)에서 chunk_only 를 63.9% → 52.8% 로
개선했지만, 그때 "같은 표본에 8번째로 대본 규칙이므로 가설"이라고 기록했다
(PHASE1_GATE §12). **전체 432개에서는 반대로 나빴다** — combined 0% 예산 기준
max 73.1% 대 max_corrected 81.9% (RESULTS.md §3). 그래서 기본은 `max` 다.
보정 규칙은 대조용으로 남긴다.

================================================================================
조건(condition) = view 집합의 선택
================================================================================
ablation 4조건은 별도 로직이 아니라 **어떤 view 를 넣느냐**로만 갈린다. 점수는
Phase 3 에서 view 마다 한 번씩만 계산해 두고, 여기서는 고르기만 한다.

게이트 결과에 따라 chunk 는 주 방어가 아니라 **대조군**이다 (PHASE1_GATE.md §24).

사용법:
    python defense/decision_aggregator.py --selftest
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------- view 종류
VIEW_ORIGINAL = "original"
VIEW_NORMALIZED = "normalized"
VIEW_CHUNK = "chunk"
VIEW_NORM_CHUNK = "norm_chunk"

# 뽑기(다중 검정)가 일어나는 family. 여기에 속한 view 만 k 보정 대상이다.
LOTTERY_KINDS = frozenset({VIEW_CHUNK, VIEW_NORM_CHUNK})

# ---------------------------------------------------------------- 조건
COND_BASELINE = "baseline"
COND_NORM_ONLY = "normalization_only"
COND_CHUNK_ONLY = "chunk_only"
COND_COMBINED = "combined"

CONDITION_VIEWS: dict[str, tuple[str, ...]] = {
    COND_BASELINE:   (VIEW_ORIGINAL,),
    COND_NORM_ONLY:  (VIEW_ORIGINAL, VIEW_NORMALIZED),
    COND_CHUNK_ONLY: (VIEW_ORIGINAL, VIEW_CHUNK),
    COND_COMBINED:   (VIEW_ORIGINAL, VIEW_NORMALIZED, VIEW_CHUNK, VIEW_NORM_CHUNK),
}
ALL_CONDITIONS = tuple(CONDITION_VIEWS)

# ---------------------------------------------------------------- 집계 규칙
RULE_MAX = "max"                      # 전부 한 통에 넣고 max (명세의 "하나라도 위험")
RULE_MAX_FAMILY = "max_family"        # family 별로 모은 뒤 합류 (k 보정 없음)
RULE_MAX_CORRECTED = "max_corrected"  # family 별 + 뽑기 family 에 k 보정
RULE_SECOND = "second_highest"        # 상위 2번째 (게이트에서 최악, 대조용)
RULE_MEAN = "mean"                    # 평균 (게이트에서 열등, 대조용)
ALL_RULES = (RULE_MAX, RULE_MAX_FAMILY, RULE_MAX_CORRECTED, RULE_SECOND, RULE_MEAN)


@dataclass
class View:
    """한 번의 안전 필터 호출에 대응. score 가 None 이면 실패한 호출이다."""
    kind: str
    name: str = ""
    score: float | None = None
    n_tokens: int = 0

    def __post_init__(self) -> None:
        if self.kind not in (VIEW_ORIGINAL, VIEW_NORMALIZED, VIEW_CHUNK, VIEW_NORM_CHUNK):
            raise ValueError(f"unknown view kind: {self.kind!r}")
        if not self.name:
            self.name = self.kind


@dataclass
class Decision:
    """집계 결과. 어느 view 가 판정을 이끌었는지까지 남긴다 (Phase 4 분해용)."""
    score: float | None
    blocked: bool
    condition: str
    rule: str
    tau: float
    driving_view: str = ""
    n_views: int = 0
    n_lottery_views: int = 0
    n_failed_views: int = 0
    notes: list[str] = field(default_factory=list)


def k_correct(p_max: float, k: int) -> float:
    """max-of-k 를 1회 추출 상당으로 되돌린다.

    chunk 가 독립 추출이라고 보면 "하나라도 위험" 확률은 1-(1-p)^k 이므로,
    역변환은 p_single = 1-(1-p_max)^(1/k) 다. k=1 이면 항등이다.

    주의: 고정된 k 안에서는 단조 변환이라 곡선(순서)을 바꾸지 못한다.
    프롬프트마다 k 가 다를 때만 순위를 실제로 바꾼다 — 전체 432개가 그 경우다
    (short 는 chunk 1개, over_limit 은 7개).
    """
    if k <= 1:
        return p_max
    p = min(max(p_max, 0.0), 1.0)
    return 1.0 - (1.0 - p) ** (1.0 / k)


def select_views(views: list[View], condition: str) -> list[View]:
    """조건이 허용하는 종류의 view 만 남긴다."""
    if condition not in CONDITION_VIEWS:
        raise ValueError(f"unknown condition: {condition!r}")
    allowed = CONDITION_VIEWS[condition]
    return [v for v in views if v.kind in allowed]


def _scored(views: list[View]) -> list[View]:
    """점수가 있는 view 만. 실패한 호출을 0 으로 채우면 '안전하다' 쪽으로 편향된다."""
    return [v for v in views if v.score is not None]


def aggregate(views: list[View], rule: str = RULE_MAX) -> tuple[float | None, str]:
    """view 목록 → (점수, 그 점수를 낸 view 이름). 점수 있는 view 가 없으면 (None, "")."""
    if rule not in ALL_RULES:
        raise ValueError(f"unknown rule: {rule!r}")
    ok = _scored(views)
    if not ok:
        return None, ""

    if rule == RULE_MAX:
        best = max(ok, key=lambda v: v.score)
        return best.score, best.name

    if rule == RULE_SECOND:
        if len(ok) == 1:
            return ok[0].score, ok[0].name
        s = sorted(ok, key=lambda v: v.score, reverse=True)
        return s[1].score, s[1].name

    if rule == RULE_MEAN:
        return statistics.fmean(v.score for v in ok), "mean"

    # --- family 별로 모은 뒤 합류 (RULE_MAX_FAMILY / RULE_MAX_CORRECTED)
    lottery = [v for v in ok if v.kind in LOTTERY_KINDS]
    single = [v for v in ok if v.kind not in LOTTERY_KINDS]

    candidates: list[tuple[float, str]] = []
    if single:
        b = max(single, key=lambda v: v.score)
        candidates.append((b.score, b.name))
    if lottery:
        # 뽑기 family 는 종류별로 따로 모은다. chunk 와 norm_chunk 는 서로 다른
        # 추출 집합이므로 k 를 합쳐 세면 보정이 과해진다.
        by_kind: dict[str, list[View]] = {}
        for v in lottery:
            by_kind.setdefault(v.kind, []).append(v)
        for kind, group in by_kind.items():
            b = max(group, key=lambda v: v.score)
            sc = b.score
            if rule == RULE_MAX_CORRECTED:
                sc = k_correct(sc, len(group))
            candidates.append((sc, b.name))

    best = max(candidates)
    return best[0], best[1]


def decide(views: list[View], condition: str, tau: float,
           rule: str = RULE_MAX) -> Decision:
    """조건·규칙·임계값을 적용해 차단 여부를 낸다.

    tau 는 사후 스윕용 자유 변수다. 점수는 view 마다 한 번만 계산해 두고
    (Phase 3), tau 는 분석 단계에서 쓸어본다 (Phase 4).
    """
    sel = select_views(views, condition)
    score, driver = aggregate(sel, rule)
    n_fail = sum(1 for v in sel if v.score is None)
    d = Decision(
        score=score, blocked=(score is not None and score > tau),
        condition=condition, rule=rule, tau=tau, driving_view=driver,
        n_views=len(sel),
        n_lottery_views=sum(1 for v in sel if v.kind in LOTTERY_KINDS),
        n_failed_views=n_fail,
    )
    if score is None:
        # 판정 불가. 차단도 통과도 아닌 상태로 남긴다 — 임의로 채우면
        # 실패가 많을수록 결론이 한쪽으로 기운다 (root_cause.py 의 undecided 와 같은 원칙).
        d.notes.append("점수를 낸 view 가 하나도 없음 — 판정 불가")
    if n_fail:
        d.notes.append(f"실패한 view {n_fail}개는 집계에서 제외됨")
    return d


# ================================================================ self-test
def _selftest() -> int:
    PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
    res: list[bool] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        res.append(bool(cond))
        print(f"  [{PASS if cond else FAIL}] {name}" + (f"  — {detail}" if detail else ""))

    print("=" * 78)
    print("  decision_aggregator")
    print("=" * 78)

    def mk(orig, norm=None, chunks=(), nchunks=()):
        vs = [View(VIEW_ORIGINAL, "original", orig)]
        if norm is not None:
            vs.append(View(VIEW_NORMALIZED, "normalized", norm))
        vs += [View(VIEW_CHUNK, f"chunk:{i}", s) for i, s in enumerate(chunks)]
        vs += [View(VIEW_NORM_CHUNK, f"norm_chunk:{i}", s) for i, s in enumerate(nchunks)]
        return vs

    # --- 조건이 view 를 제대로 고르는가
    v = mk(0.01, 0.5, (0.3, 0.4), (0.2,))
    check("baseline 은 원본만", [x.kind for x in select_views(v, COND_BASELINE)] == [VIEW_ORIGINAL])
    check("정규화만 조건은 2개", len(select_views(v, COND_NORM_ONLY)) == 2)
    check("chunk만 조건은 원본+chunk", len(select_views(v, COND_CHUNK_ONLY)) == 3)
    check("combined 는 전부", len(select_views(v, COND_COMBINED)) == 5)

    # --- k 보정
    check("k=1 이면 항등", k_correct(0.4, 1) == 0.4)
    check("k>1 이면 낮아짐", k_correct(0.4, 7) < 0.4, f"{k_correct(0.4,7):.4f}")
    check("k 가 클수록 더 낮아짐", k_correct(0.4, 16) < k_correct(0.4, 7))
    check("1-(1-p)^k 의 역변환", abs(1 - (1 - k_correct(0.4, 7)) ** 7 - 0.4) < 1e-9)

    # --- 게이트가 드러낸 문제를 실제로 막는가
    # 안전 프롬프트: 원본·정규화본은 낮은데 chunk 하나가 튐 (오탐률 5% 상황)
    safe = mk(0.008, 0.010, (0.003, 0.002, 0.31, 0.004, 0.003, 0.002, 0.003))
    s_max, _ = aggregate(safe, RULE_MAX)
    s_cor, _ = aggregate(safe, RULE_MAX_CORRECTED)
    check("한 통 max 는 튄 chunk 에 끌려감", abs(s_max - 0.31) < 1e-9, f"{s_max:.4f}")
    check("family+보정은 끌려감이 줄어듦", s_cor < s_max, f"{s_cor:.4f} < {s_max:.4f}")
    check("보정해도 원본 점수 아래로는 안 내려감", s_cor >= 0.010 - 1e-9, f"{s_cor:.4f}")

    # 위험 프롬프트: 정규화본이 신호를 실어 나름 (뽑기 없음 → 보정 대상 아님)
    unsafe = mk(0.005, 0.72, (0.003, 0.12, 0.004))
    u_cor, u_drv = aggregate(unsafe, RULE_MAX_CORRECTED)
    check("정규화본은 보정되지 않고 그대로", abs(u_cor - 0.72) < 1e-9, f"{u_cor:.4f}")
    check("판정을 이끈 view 를 기록", u_drv == "normalized", u_drv)

    # --- chunk 와 norm_chunk 는 k 를 따로 센다
    a = mk(0.0, None, (0.4,) * 4, (0.4,) * 4)
    b = mk(0.0, None, (0.4,) * 8)
    sa, _ = aggregate(a, RULE_MAX_CORRECTED)
    sb, _ = aggregate(b, RULE_MAX_CORRECTED)
    check("chunk/norm_chunk 의 k 를 합쳐 세지 않음", sa > sb, f"{sa:.4f} > {sb:.4f}")

    # --- 실패한 view 처리
    part = [View(VIEW_ORIGINAL, "original", None), View(VIEW_NORMALIZED, "normalized", 0.6)]
    d = decide(part, COND_NORM_ONLY, tau=0.5)
    check("실패 view 는 0 으로 채우지 않고 제외", d.score == 0.6 and d.n_failed_views == 1)
    allfail = [View(VIEW_ORIGINAL, "original", None)]
    d2 = decide(allfail, COND_BASELINE, tau=0.5)
    check("전부 실패면 판정 불가 (차단도 통과도 아님)",
          d2.score is None and d2.blocked is False and d2.notes)

    # --- 임계값
    d3 = decide(mk(0.6), COND_BASELINE, tau=0.5)
    d4 = decide(mk(0.4), COND_BASELINE, tau=0.5)
    check("tau 초과면 차단", d3.blocked and not d4.blocked)
    check("baseline 은 원본 점수 그대로", d3.score == 0.6)

    # --- 실측 데이터 재현: 게이트 1차의 chunk max 가 그대로 나오는가
    import csv
    from collections import defaultdict
    gate = REPO / "defense" / "gate" / "phase1_gate_chunks.csv"
    if gate.exists():
        ch = defaultdict(list)
        for r in csv.DictReader(open(gate, encoding="utf-8-sig")):
            ch[r["prompt_id"]].append(float(r["unsafe_score"]))
        pid = sorted(ch)[0]
        views = [View(VIEW_ORIGINAL, "original", 0.0)] + \
                [View(VIEW_CHUNK, f"chunk:{i}", s) for i, s in enumerate(ch[pid])]
        got, _ = aggregate(views, RULE_MAX)
        check("게이트 CSV 로 chunk max 재현", abs(got - max(ch[pid])) < 1e-12, f"{got:.6f}")
        check("보정 시 k=chunk 수가 반영됨",
              abs(aggregate(views, RULE_MAX_CORRECTED)[0]
                  - k_correct(max(ch[pid]), len(ch[pid]))) < 1e-12)
    else:
        print("  [skip] 게이트 CSV 없음 — 실측 재현 검사 생략")

    n = sum(res)
    print("=" * 78)
    print(f"  {n}/{len(res)} CHECK 통과")
    print("=" * 78)
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    print(__doc__)
