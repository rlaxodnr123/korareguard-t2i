#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_tradeoff.py — KoRareGuard-T2I / Student 5

성능–안전성 trade-off curve (명세 필수 산출물).

출력: defense/figures/figure_tradeoff.{png,pdf}

================================================================================
왜 곡선인가
================================================================================
안전 필터의 판정 문턱 0.5 는 자연 상수가 아니다. 이미 있는 점수만으로 τ 를 낮추기만
해도 under-blocking 이 크게 떨어진다. 따라서 조건마다 점 하나를 찍어 비교하면
"방어의 효과"와 "임계값을 낮춘 것"이 구별되지 않는다.

조건별로 τ 를 쓸어 곡선을 그리고, **같은 over-blocking 지점에서 세로로 비교**한다.
baseline 곡선 아래로 내려간 것만이 기여다.

패널 B 는 그 개선이 어디서 왔는지를 보인다. 정규화는 희귀 표현 칸에서만 움직이고
일반 표현 칸은 정확히 그대로다 — 설계가 상정한 실패 모드와 일치한다.

사용법:
    python defense/figures/figure_tradeoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402

from defense.analyze_phase4 import (  # noqa: E402
    SCORES_CSV, completed_conditions, condition_scores, load_views, pareto_front,
    under_at,
)
from defense.decision_aggregator import (  # noqa: E402
    COND_BASELINE, COND_CHUNK_ONLY, COND_COMBINED, COND_NORM_ONLY, RULE_MAX,
)
from defense.figures._style import (  # noqa: E402
    BASELINE, C_ALTDIFF, C_DEFENSE, C_SGUARD, GRIDLINE, INK_MUTED, INK_PRIMARY,
    INK_SECONDARY, save, setup, suptitle,
)

STYLE = {
    COND_BASELINE:   dict(color=INK_MUTED, ls=(0, (1, 1.6)), lw=1.6, label="baseline (방어 없음)", z=2),
    COND_CHUNK_ONLY: dict(color=C_ALTDIFF, ls=(0, (5, 2)), lw=1.6, label="chunk only (대조군)", z=3),
    COND_NORM_ONLY:  dict(color=C_SGUARD, ls="-", lw=2.4, label="normalization only", z=5),
    COND_COMBINED:   dict(color=C_DEFENSE, ls="-", lw=2.0, label="combined", z=4),
}
ORDER = [COND_BASELINE, COND_CHUNK_ONLY, COND_NORM_ONLY, COND_COMBINED]

CELLS = [("common", ("short",), "일반 × 단문"),
         ("common", ("near_limit", "over_limit"), "일반 × 장문"),
         ("rare", ("short",), "희귀 × 단문"),
         ("rare", ("near_limit", "over_limit"), "희귀 × 장문")]


def panel_curves(ax, views, meta, conds):
    label = {p: m["safety_label"] for p, m in meta.items()}
    for cond in ORDER:
        if cond not in conds:
            continue
        s = STYLE[cond]
        f = pareto_front(condition_scores(views, cond, RULE_MAX), label)
        # 계단 함수로 그린다 — τ 사이 구간에서 성적이 변하지 않으므로 직선 보간은 거짓이다.
        xs, ys = [], []
        for ob, ub, _ in f:
            if xs:
                xs.append(ob); ys.append(ys[-1])
            xs.append(ob); ys.append(ub)
        ax.plot(xs, ys, color=s["color"], linestyle=s["ls"], lw=s["lw"],
                label=s["label"], zorder=s["z"], solid_capstyle="round")

    ax.set_xlim(-0.6, 24)
    ax.set_ylim(20, 95)
    ax.set_xlabel("over-blocking — 안전 프롬프트를 차단한 비율 (%)", fontsize=8.6, color=INK_SECONDARY)
    ax.set_ylabel("under-blocking — 위험 프롬프트 통과 비율 (%)", fontsize=8.6,
                  color=INK_SECONDARY, labelpad=6)
    ax.grid(True, color=GRIDLINE, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)

    # 같은 over-blocking 예산에서 세로 비교 — 이 그림이 주장하는 읽는 법
    b = 5.6
    fb = pareto_front(condition_scores(views, COND_BASELINE, RULE_MAX), label)
    fn = pareto_front(condition_scores(views, COND_NORM_ONLY, RULE_MAX), label)
    yb, yn = under_at(fb, b)[0], under_at(fn, b)[0]
    ax.annotate("", xy=(b, yn), xytext=(b, yb),
                arrowprops=dict(arrowstyle="<->", color=INK_PRIMARY, lw=1.1, shrinkA=0, shrinkB=0))
    ax.plot([b, b], [20, min(yb, yn)], color=INK_PRIMARY, lw=0.7, alpha=0.35, zorder=1)
    ax.text(b + 0.7, (yb + yn) / 2, f"{yb - yn:.1f}pp", fontsize=8.4, color=INK_PRIMARY,
            va="center", fontweight="bold")
    # 설명문은 곡선이 지나지 않는 왼쪽 아래 구석에 둔다. 화살표 옆에 두면
    # normalization 곡선과 겹친다.
    ax.text(0.6, 24.5, "세로 비교 — 같은 over-blocking 예산에서\nbaseline 대비 얼마나 내려갔는가",
            fontsize=7.4, color=INK_MUTED, va="bottom", ha="left")

    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.2,
                    handlelength=2.6, labelspacing=0.5)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    ax.set_title("A   조건별 trade-off 곡선   (아래·왼쪽일수록 좋음)",
                 fontsize=9.6, color=INK_PRIMARY, loc="left", pad=8)


def panel_cells(ax, views, meta, conds, budget=5.6):
    label_all = {p: m["safety_label"] for p, m in meta.items()}
    show = [c for c in (COND_BASELINE, COND_NORM_ONLY, COND_CHUNK_ONLY) if c in conds]
    n = len(show)
    w = 0.8 / n
    for j, cond in enumerate(show):
        vals = []
        for rar, lens, _ in CELLS:
            sub = {p for p, m in meta.items()
                   if m["rarity_label"] == rar and m["length_level"] in lens}
            sc = {p: v for p, v in condition_scores(views, cond, RULE_MAX).items() if p in sub}
            r = under_at(pareto_front(sc, {p: label_all[p] for p in sub}), budget)
            vals.append(r[0] if r else 0.0)
        xs = [i - 0.4 + w * (j + 0.5) for i in range(len(CELLS))]
        col = STYLE[cond]["color"]
        ax.bar(xs, vals, width=w * 0.88, color=col,
               alpha=0.30 if cond == COND_CHUNK_ONLY else 0.85,
               edgecolor=col, linewidth=1.0, zorder=3,
               label=STYLE[cond]["label"].split(" (")[0])
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.6, f"{v:.0f}", ha="center", fontsize=7.0, color=col)

    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([c[2] for c in CELLS], fontsize=8.4, color=INK_SECONDARY)
    ax.set_ylim(0, 104)
    ax.set_ylabel("under-blocking (%)", fontsize=8.6, color=INK_SECONDARY)
    ax.grid(True, axis="y", color=GRIDLINE, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=7.8, ncol=1, labelspacing=0.4)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    ax.set_title(f"B   개선이 어디서 오는가   (over-blocking ≤ {budget}% 기준)",
                 fontsize=9.6, color=INK_PRIMARY, loc="left", pad=8)
    ax.text(0.5, -0.185,
            "정규화는 일반 표현 칸을 정확히 0 만큼 움직이고 희귀 표현 칸만 낮춘다 — 어휘 축 전용.\n"
            "chunk 는 희석이 가장 심한 '일반 × 장문' 에서 오히려 악화한다.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.4, color=INK_MUTED)


def main() -> int:
    if not SCORES_CSV.exists():
        print(f"점수 파일이 없습니다: {SCORES_CSV}")
        return 1
    setup()
    views, meta = load_views(SCORES_CSV)
    conds, _ = completed_conditions(views, meta, len(meta))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    panel_curves(axes[0], views, meta, conds)
    panel_cells(axes[1], views, meta, conds)

    suptitle(fig, "성능–안전성 trade-off",
             "432 프롬프트 · 2,835 view · 집계 max · τ 스윕 곡선 (RESULTS.md §1-2)")
    fig.subplots_adjust(top=0.80, bottom=0.20, wspace=0.26)
    save(fig, "figure_tradeoff")
    return 0


if __name__ == "__main__":
    sys.exit(main())
