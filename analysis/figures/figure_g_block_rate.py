#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_g_block_rate.py — KoRareGuard-T2I / Student 2 / Figure G

RQ-T9: 가시성·희귀도·길이가 안전 필터의 판정과 어떻게 연관되는가?

이 그림은 두 가지를 나란히 보여준다.

  왼쪽  희귀도 x 길이별 차단율
        같은 유해 개념인데 표현만 바꾸면 short 에서 50.0% -> 11.1% 로 떨어진다.
  오른쪽 unsafe_score 분포 (로그 축)
        판정만 뒤집히는 것이 아니라 연속 점수 자체가 낮다.
        길이가 길어질수록 점수 분포 전체가 판정 임계값(0.5)에서 멀어진다.

두 패널이 같이 있어야 하는 이유:
  왼쪽만 보면 near_limit / over_limit 에서 희귀도 효과가 사라진 것처럼 보이는데,
  그건 길이 효과가 양쪽을 모두 바닥으로 끌어내렸기 때문이다. 오른쪽 점수 분포가
  그 바닥이 얼마나 깊은지 보여준다.

주의:
  조건 1(절단 없음) 결과만 쓴다. 절단 효과를 배제한 상태에서도 이 현상이
  나타난다는 것이 요점이므로, 절단 조건을 섞으면 논지가 흐려진다.

입력 : evaluation/safety/safety_results.csv  (학생 4)
       benchmarks/prompts/prompts.csv
출력 : analysis/figures/figure_g_block_rate.{png,pdf}

사용법:
    .venv\\Scripts\\python.exe analysis/figures/figure_g_block_rate.py
"""

from __future__ import annotations

import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (  # noqa: E402
    C_SGUARD, C_ALTDIFF, INK_PRIMARY, INK_SECONDARY, INK_MUTED,
    GRIDLINE, BASELINE, SURFACE, LENGTHS,
    setup, tidy_axes, suptitle, save, REPO,
)

import matplotlib.pyplot as plt  # noqa: E402

SAFETY_CSV = REPO / "evaluation" / "safety" / "safety_results.csv"
PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"

# 이 그림의 카테고리 축은 SGuard/AltDiffusion 이 아니라 일반/희귀다.
# 슬롯은 같은 순서(1, 2)를 그대로 쓰고 범례로 구분한다.
#   validate_palette.js --mode light 전 항목 PASS
#   CVD 분리 ΔE 24.7 (protan) / 정상시야 ΔE 33.6 / 대비 둘 다 3:1 이상
C_COMMON = C_SGUARD        # 슬롯 1
C_RARE = C_ALTDIFF         # 슬롯 2

POLICY = "native"          # 조건 1 만 사용
DECISION_THRESHOLD = 0.5   # unsafe_score > 0.5 <-> decision == unsafe
LENGTH_LABEL = {"short": "short", "near_limit": "near_limit", "over_limit": "over_limit"}
RARITY = [("common", "일반 표현", C_COMMON), ("rare", "희귀 표현", C_RARE)]


def load() -> tuple[dict, dict]:
    if not SAFETY_CSV.exists():
        raise FileNotFoundError(
            f"{SAFETY_CSV} 가 없습니다. 학생 4 의 safety_results.csv 가 필요합니다.")
    with open(PROMPTS_CSV, encoding="utf-8-sig", newline="") as f:
        prompts = {r["prompt_id"]: r for r in csv.DictReader(f)}
    with open(SAFETY_CSV, encoding="utf-8-sig", newline="") as f:
        safety = {(r["prompt_id"], r["input_policy"]): r for r in csv.DictReader(f)}
    return prompts, safety


def panel_block_rate(ax, prompts, safety) -> dict:
    """희귀도 x 길이별 차단율. 0 까지 잇는 줄기 + 점 (그림 A 와 같은 마크 규격)."""
    rate = {}
    for lv in LENGTHS:
        for rar, _, _ in RARITY:
            ids = [p for p, x in prompts.items()
                   if x["safety_label"] == "unsafe"
                   and x["length_level"] == lv and x["rarity_label"] == rar]
            hit = sum(1 for p in ids if safety[(p, POLICY)]["decision"] == "unsafe")
            rate[(lv, rar)] = (hit, len(ids))

    ys = range(len(LENGTHS))
    ax.axvline(0, color=BASELINE, lw=1.4, zorder=1)
    for i in ys:
        ax.axhline(i, color=GRIDLINE, lw=0.6, zorder=0)

    DODGE = 0.17
    for (rar, _, color), off in zip(RARITY, (+DODGE, -DODGE)):
        xs = [rate[(lv, rar)][0] / rate[(lv, rar)][1] * 100 for lv in LENGTHS]
        yy = [i + off for i in ys]
        for x, y in zip(xs, yy):
            ax.plot([0, x], [y, y], color=color, lw=1.6, alpha=0.55,
                    solid_capstyle="round", zorder=2)
        ax.scatter(xs, yy, s=46, color=color, zorder=3,
                   edgecolors=SURFACE, linewidths=1.4)          # 2px 서피스 링
        # 직접 라벨 — 값이 6개뿐이라 전부 붙여도 과하지 않다.
        # 0% 는 점이 축선 위에 얹혀 라벨이 축과 겹치므로 여백을 더 준다.
        for x, y, lv in zip(xs, yy, LENGTHS):
            h, n = rate[(lv, rar)]
            ax.text(x + (3.2 if x == 0 else 1.6), y, f"{x:.1f}%  ({h}/{n})",
                    fontsize=8.5, color=INK_SECONDARY, va="center", ha="left")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([LENGTH_LABEL[lv] for lv in LENGTHS],
                       fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(-0.7, len(LENGTHS) - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 66)
    ax.set_xlabel("차단율 (%)  ·  유해 프롬프트 36개 중", fontsize=9, color=INK_SECONDARY)
    ax.set_title("A   희귀도 × 길이별 차단율", fontsize=11, color=INK_PRIMARY,
                 pad=14, loc="left", fontweight="bold")
    ax.text(0, 1.012, "조건 1 (절단 없음) — 표현만 바꾸면 short 에서 4.5배 차이",
            transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY, va="bottom")
    tidy_axes(ax, hide=("top", "right", "left"))
    ax.grid(False)
    return rate


def panel_scores(ax, prompts, safety) -> None:
    """unsafe_score 분포. 값이 0.0002~0.9975 라 로그 축을 쓴다."""
    rng = __import__("random").Random(7)      # 지터만 재현 가능하게
    for i, lv in enumerate(LENGTHS):
        ax.axhline(i, color=GRIDLINE, lw=0.6, zorder=0)

    DODGE = 0.17
    med = {}
    for (rar, _, color), off in zip(RARITY, (+DODGE, -DODGE)):
        for i, lv in enumerate(LENGTHS):
            v = [float(safety[(p, POLICY)]["unsafe_score"])
                 for p, x in prompts.items()
                 if x["safety_label"] == "unsafe"
                 and x["length_level"] == lv and x["rarity_label"] == rar]
            yy = [i + off + rng.uniform(-0.055, 0.055) for _ in v]
            ax.scatter(v, yy, s=26, color=color, zorder=3, alpha=0.75,
                       edgecolors=SURFACE, linewidths=0.8)
            # 중앙값 — 본문이 인용하는 값이므로 그림에도 보여야 한다.
            # 점 구름 위에 얹는 짧은 세로 막대. 색은 시리즈를 그대로 따른다.
            m = __import__("statistics").median(v)
            med[(lv, rar)] = m
            ax.plot([m, m], [i + off - 0.115, i + off + 0.115], color=color,
                    lw=2.6, zorder=4, solid_capstyle="butt",
                    path_effects=[__import__("matplotlib.patheffects",
                                             fromlist=["withStroke"])
                                  .withStroke(linewidth=5.2, foreground=SURFACE)])
    panel_scores.medians = med

    # 중앙값 수치를 붙인다. 로그 축에서 세로선 위치만으로는 값을 못 읽는다.
    # 점 구름과 겹치지 않도록 각 행의 바깥 여백으로 밀어낸다.
    #   일반(위 시리즈) 라벨은 세로선 위, 희귀(아래 시리즈) 라벨은 아래.
    # 배율 주석을 그림 안에 넣었더니 화살표가 행 사이를 가로지르며 다른 라벨과
    # 겹쳤다. 그 문구는 패널 부제로 올렸다.
    for (rar, _, _), off, dy, va in zip(RARITY, (+DODGE, -DODGE),
                                        (+0.21, -0.21), ("top", "bottom")):
        for i, lv in enumerate(LENGTHS):
            m = med[(lv, rar)]
            ax.text(m, i + off + dy, f"{m:.4f}", fontsize=7.8,
                    color=INK_SECONDARY, ha="center", va=va)

    ax.axvline(DECISION_THRESHOLD, color=INK_MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(DECISION_THRESHOLD * 1.15, -0.78, "판정 임계값 0.5", fontsize=8.5,
            color=INK_MUTED, va="center", ha="left")

    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.6)
    # 로그 축 기본 포맷터는 지수의 음수 부호로 U+2212 를 쓴다. 한글 폰트에
    # 그 글자가 없어 사각형으로 렌더링되므로 눈금 라벨을 직접 준다.
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.0001", "0.001", "0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.set_yticks(range(len(LENGTHS)))
    ax.set_yticklabels([LENGTH_LABEL[lv] for lv in LENGTHS],
                       fontsize=9, color=INK_SECONDARY)
    # 라벨을 행 바깥으로 밀었으므로 위아래 여백을 그만큼 넓힌다.
    ax.set_ylim(-0.86, len(LENGTHS) - 0.14)
    ax.invert_yaxis()
    ax.set_xlabel("unsafe_score  (로그 축)", fontsize=9, color=INK_SECONDARY)
    ax.set_title("B   판정 점수 분포", fontsize=11, color=INK_PRIMARY,
                 pad=14, loc="left", fontweight="bold")
    ratio = med[("short", "common")] / med[("short", "rare")]
    ax.text(0, 1.012,
            f"굵은 세로선은 중앙값 — short 에서 {med[('short','common')]:.4f} vs "
            f"{med[('short','rare')]:.4f} ({ratio:.0f}배)",
            transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY, va="bottom")
    tidy_axes(ax, hide=("top", "right", "left"))
    ax.grid(False)


def main() -> int:
    setup()
    prompts, safety = load()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.70, bottom=0.17, wspace=0.24)

    rate = panel_block_rate(axes[0], prompts, safety)
    panel_scores(axes[1], prompts, safety)

    # 범례 — 시리즈가 2개이므로 반드시 둔다. 텍스트는 잉크 색, 점이 정체성을 진다.
    for k, (rar, label, color) in enumerate(RARITY):
        x = 0.085 + k * 0.105
        fig.plot = None
        fig.text(x + 0.016, 0.795, label, fontsize=9, color=INK_SECONDARY,
                 ha="left", va="center")
        fig.add_artist(plt.Line2D([x], [0.795], marker="o", ms=6.5, color=color,
                                  transform=fig.transFigure, linestyle="none"))

    suptitle(fig, "Figure G   희귀 표현과 긴 문맥에서 안전 필터가 무력화된다",
             "SGuard-ContentFilter-2B-v1 · 조건 1(절단 없음) · 유해 프롬프트 216개")

    save(fig, "figure_g_block_rate")

    print("\n  [요약] 차단율")
    for lv in LENGTHS:
        c, r = rate[(lv, "common")], rate[(lv, "rare")]
        cp, rp = c[0]/c[1]*100, r[0]/r[1]*100
        ratio = f"{cp/rp:.1f}x" if rp else "—"
        print(f"    {lv:11} 일반 {c[0]:>2}/{c[1]} = {cp:5.1f}%   "
              f"희귀 {r[0]:>2}/{r[1]} = {rp:5.1f}%   배율 {ratio}")

    med = panel_scores.medians
    print("\n  [요약] unsafe_score 중앙값 (그림의 세로선)")
    for lv in LENGTHS:
        print(f"    {lv:11} 일반 {med[(lv,'common')]:.4f}   희귀 {med[(lv,'rare')]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
