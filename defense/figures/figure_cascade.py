#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_cascade.py — KoRareGuard-T2I / Student 5

호출 수 ↔ 성능 곡선 (명세의 "긴 prompt 에서만 적용했을 때의 효과" 항목).

출력: defense/figures/figure_cascade.{png,pdf}

================================================================================
이 그림이 답하는 것
================================================================================
방어는 안전 필터를 더 부르는 대가로 성능을 얻는다. 그 대가를 줄이는 두 가지
방법을 비교한다.

A. **길이 게이트** — 긴 프롬프트에만 방어를 건다. 설계 문서가 제안한 절약 방식이다.
B. **조기 차단** — 원본 점수만으로 판정이 확실하면 나머지 view 를 부르지 않는다.

결론이 갈린다. 길이 게이트는 절약분에 비해 성능을 너무 많이 버린다 — 정규화가
고치는 것이 길이가 아니라 **어휘**이기 때문이다. 희귀 표현은 짧은 프롬프트에서도
안 보이므로(단문 89% → 61%), 짧은 것을 건너뛰면 이득의 대부분이 사라진다.

설계 문서가 길이 게이트를 제안한 것은 원인을 절단·희석으로 봤기 때문이다.
실측된 원인은 어휘였고, 그래서 절약 손잡이도 달라진다.

사용법:
    python defense/figures/figure_cascade.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402

from defense.analyze_phase4 import (  # noqa: E402
    SCORES_CSV, cascade_curve, load_views,
)
from defense.figures._style import (  # noqa: E402
    BASELINE, C_ALTDIFF, C_DEFENSE, C_SGUARD, GRIDLINE, INK_MUTED, INK_PRIMARY,
    INK_SECONDARY, save, setup, suptitle,
)

BUDGET = 5.6


def _frame(ax, xlabel):
    ax.set_xlabel(xlabel, fontsize=8.6, color=INK_SECONDARY)
    ax.set_ylabel(f"under-blocking (%)   ·   over-block ≤ {BUDGET}%",
                  fontsize=8.6, color=INK_SECONDARY, labelpad=6)
    ax.grid(True, color=GRIDLINE, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def panel_length(ax, c):
    pts = [p for p in c["length_gate"] if p["under"] is not None]
    full = next(p for p in pts if p["min_tokens"] == 0)
    none = max(pts, key=lambda p: p["min_tokens"])
    mid = [p for p in pts if p is not full and p is not none]

    ax.plot([p["calls_per_prompt"] for p in pts], [p["under"] for p in pts],
            color=C_ALTDIFF, lw=1.6, marker="o", ms=4.5, zorder=3,
            markerfacecolor="white", markeredgewidth=1.4)
    ax.scatter([full["calls_per_prompt"]], [full["under"]], s=64, color=C_SGUARD, zorder=5)
    ax.scatter([none["calls_per_prompt"]], [none["under"]], s=64, color=INK_MUTED, zorder=5)

    # 라벨은 offset points 로 붙인다. 데이터 좌표로 밀면 축 밖으로 나간다.
    ax.annotate(f"전부 적용   {full['calls_per_prompt']:.2f}회 · {full['under']:.1f}%",
                xy=(full["calls_per_prompt"], full["under"]),
                xytext=(-8, 10), textcoords="offset points",
                fontsize=7.6, color=C_SGUARD, ha="right", va="bottom")
    ax.annotate(f"방어 없음   {none['under']:.1f}%",
                xy=(none["calls_per_prompt"], none["under"]),
                xytext=(10, -2), textcoords="offset points",
                fontsize=7.6, color=INK_MUTED, ha="left", va="center")
    for p in mid:
        if p["min_tokens"] in (24, 122, 422):
            below = p["min_tokens"] == 422     # 위에 두면 '방어 없음' 라벨과 겹친다
            ax.annotate(f"{p['min_tokens']}토큰 초과만",
                        xy=(p["calls_per_prompt"], p["under"]),
                        xytext=(0, -11 if below else 9), textcoords="offset points",
                        fontsize=7.0, color=INK_MUTED, ha="center",
                        va="top" if below else "bottom")
    ax.margins(x=0.16, y=0.14)

    _frame(ax, "프롬프트당 안전 필터 호출 수")
    ax.set_title("A   길이 게이트 — 긴 프롬프트에만 적용", fontsize=9.6,
                 color=INK_PRIMARY, loc="left", pad=8)
    ax.text(0.02, 0.035,
            "24토큰 초과로만 좁혀도 이득의 47%, 40토큰이면 84% 를 잃는다.\n"
            "정규화가 고치는 것은 길이가 아니라 어휘라서, 짧은 프롬프트를\n"
            "건너뛰면 희귀 표현 이득이 같이 사라진다.",
            transform=ax.transAxes, fontsize=7.4, color=INK_MUTED, va="bottom")


def panel_early(ax, c):
    pts = [p for p in c["early_block"] if p["under"] is not None]
    off = next(p for p in pts if p["high"] > 1)
    best = min(pts, key=lambda p: (p["under"], p["calls_per_prompt"]))

    ax.plot([p["calls_per_prompt"] for p in pts], [p["under"] for p in pts],
            color=C_DEFENSE, lw=1.6, marker="o", ms=4.5, zorder=3,
            markerfacecolor="white", markeredgewidth=1.4)
    ax.scatter([off["calls_per_prompt"]], [off["under"]], s=64, color=C_SGUARD, zorder=5)
    ax.scatter([best["calls_per_prompt"]], [best["under"]], s=64, color=C_DEFENSE, zorder=5)

    ax.annotate(f"조기 차단 끔   {off['calls_per_prompt']:.2f}회 · {off['under']:.1f}%",
                xy=(off["calls_per_prompt"], off["under"]),
                xytext=(-8, 9), textcoords="offset points",
                fontsize=7.6, color=C_SGUARD, ha="right", va="bottom")
    ax.annotate(f"문턱 {best['high']}   {best['calls_per_prompt']:.2f}회 · {best['under']:.1f}%",
                xy=(best["calls_per_prompt"], best["under"]),
                xytext=(10, -4), textcoords="offset points",
                fontsize=7.6, color=C_DEFENSE, ha="left", va="top")
    ax.margins(x=0.16, y=0.34)   # 설명문이 곡선 꼭짓점을 덮지 않도록 위 여백을 넉넉히

    _frame(ax, "프롬프트당 안전 필터 호출 수")
    ax.set_title("B   조기 차단 — 원본 점수로 판정이 서면 종료", fontsize=9.6,
                 color=INK_PRIMARY, loc="left", pad=8)
    # 곡선이 왼쪽 위에서 오른쪽 아래로 흐르므로 오른쪽 위가 비어 있다.
    ax.text(0.98, 0.97,
            "호출 1.50 → 1.24 구간에서 under-blocking 이\n"
            "64.4~67.1% 사이를 오르내린다. 폭 2.8pp 는 위험\n"
            "프롬프트 216개 중 약 6개로, 방향성 있는 손실이\n"
            "아니라 표본 흔들림이다.",
            transform=ax.transAxes, fontsize=7.4, color=INK_MUTED,
            ha="right", va="top")


def main() -> int:
    if not SCORES_CSV.exists():
        print(f"점수 파일이 없습니다: {SCORES_CSV}")
        return 1
    setup()
    views, meta = load_views(SCORES_CSV)
    c = cascade_curve(views, meta, over_budget=BUDGET)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    panel_length(axes[0], c)
    panel_early(axes[1], c)

    suptitle(fig, "비용 — 호출 수와 성능의 교환",
             f"normalization only · 432 프롬프트 · 집계 max · over-block ≤ {BUDGET}% (RESULTS.md §4)")
    fig.subplots_adjust(top=0.79, bottom=0.15, wspace=0.28)
    save(fig, "figure_cascade")
    return 0


if __name__ == "__main__":
    sys.exit(main())
