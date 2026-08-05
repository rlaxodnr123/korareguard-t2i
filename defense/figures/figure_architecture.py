#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_architecture.py — KoRareGuard-T2I / Student 5

방어 파이프라인 architecture diagram (명세 필수 산출물).

출력: defense/figures/figure_architecture.{png,pdf}

================================================================================
이 그림이 주장하는 것
================================================================================
상자와 화살표를 늘어놓는 대신, 설계에서 실제로 갈린 세 지점을 보이게 그린다.

1. **생성기는 원본 프롬프트를 받는다.** 정규화본은 안전 필터만 본다. 그래서
   문화 표현이 이미지 쪽에서 손실되지 않고, 이 방어가 학생3 의 생성 실험과
   분리된다. 점선 우회로가 그 결정이다.

2. **view 는 한 통에 모이지 않는다.** 원본·정규화본은 1회 추출이고 chunk 는
   7~16회 추출이라, 한 통에 넣고 max 를 취하면 뽑기 없는 view 까지 chunk 의
   오탐에 끌려간다 (PHASE1_GATE §22). family 별로 모은 뒤 합류한다.

3. **chunk 는 대조군이다.** 분리도는 복원하지만(ORACLE 이 baseline 을 최대
   33pp 앞섬) 어느 chunk 인지 점수로 고를 수 없어 4개 크기 전부 실패했다.
   실선(주 경로)이 아니라 옅은 선으로 그린다.

사용법:
    python defense/figures/figure_architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from defense.figures._style import (  # noqa: E402
    BASELINE, C_ALTDIFF, C_DEFENSE, C_MUTEDBOX, C_SGUARD, INK_MUTED,
    INK_PRIMARY, INK_SECONDARY, save, setup, suptitle,
)
import matplotlib.pyplot as plt  # noqa: E402


def box(ax, x, y, w, h, label, sub="", *, ec=INK_PRIMARY, fc="none",
        lw=1.3, fs=9.5, subfs=7.8, alpha=1.0, ls="-", tc=None):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha, linestyle=ls,
        zorder=2))
    tc = tc or ec
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=tc, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.022, sub, ha="center", va="center",
                fontsize=subfs, color=INK_MUTED, zorder=3)


def arrow(ax, p0, p1, *, color=INK_PRIMARY, lw=1.2, ls="-", alpha=1.0,
          rad=0.0, label="", lpos=0.5, fs=7.5, dy=0.012):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9, linewidth=lw, color=color,
        linestyle=ls, alpha=alpha, zorder=1,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=2))
    if label:
        ax.text(p0[0] + (p1[0] - p0[0]) * lpos, p0[1] + (p1[1] - p0[1]) * lpos + dy,
                label, ha="center", va="bottom", fontsize=fs, color=color, alpha=alpha)


def build():
    setup()
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------------------------------------------------------- 입력
    box(ax, 0.012, 0.60, 0.125, 0.10, "한국어 프롬프트", "432개 · 학생1",
        ec=INK_PRIMARY)

    # ---------------------------------------------------------------- 방어 계층 배경
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.165, 0.30), 0.345, 0.545,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.0, edgecolor=C_DEFENSE, facecolor=C_MUTEDBOX, alpha=0.55,
        zorder=0))
    ax.text(0.172, 0.858, "방어 계층 — 학생5", fontsize=8.6, color=C_DEFENSE,
            va="bottom", fontweight="bold")

    # --- 뽑기 없는 family
    ax.text(0.180, 0.795, "1회 추출 family", fontsize=7.6, color=INK_SECONDARY)
    box(ax, 0.180, 0.700, 0.150, 0.075, "원본", ec=INK_PRIMARY)
    box(ax, 0.180, 0.600, 0.150, 0.075, "정규화본",
        "희귀 표현 + 표준 설명", ec=C_DEFENSE, lw=1.6)

    # --- 뽑기 family
    ax.text(0.180, 0.520, "다중 추출 family — 대조군", fontsize=7.6, color=INK_MUTED)
    box(ax, 0.180, 0.415, 0.150, 0.070, "chunk × k", "k = 1 ~ 7",
        ec=INK_MUTED, lw=1.0, ls=(0, (4, 2)))
    box(ax, 0.180, 0.330, 0.150, 0.070, "정규화 chunk × k", "",
        ec=INK_MUTED, lw=1.0, ls=(0, (4, 2)))

    for y in (0.7375, 0.6375):
        arrow(ax, (0.137, 0.650), (0.176, y), rad=0.10, color=INK_PRIMARY)
    for y in (0.450, 0.365):
        arrow(ax, (0.137, 0.650), (0.176, y), rad=-0.12, color=INK_MUTED, alpha=0.6)

    # ---------------------------------------------------------------- 안전 필터
    box(ax, 0.360, 0.330, 0.135, 0.445, "SGuard\n안전 필터",
        "", ec=C_SGUARD, lw=1.5, fs=10)
    ax.text(0.4275, 0.400, "view 당 1회 호출\nunsafe_score", ha="center",
            va="center", fontsize=7.6, color=INK_MUTED)
    for y in (0.7375, 0.6375):
        arrow(ax, (0.331, y), (0.357, y), color=INK_PRIMARY)
    for y in (0.450, 0.365):
        arrow(ax, (0.331, y), (0.357, y), color=INK_MUTED, alpha=0.6)

    # ---------------------------------------------------------------- 집계
    box(ax, 0.545, 0.640, 0.155, 0.115, "max", "원본 · 정규화본",
        ec=C_DEFENSE, lw=1.5)
    box(ax, 0.545, 0.360, 0.155, 0.115, "max 후 k 보정",
        "1-(1-p)^(1/k)", ec=INK_MUTED, lw=1.0, ls=(0, (4, 2)))
    arrow(ax, (0.496, 0.697), (0.541, 0.697), color=C_SGUARD)
    arrow(ax, (0.496, 0.418), (0.541, 0.418), color=C_SGUARD, alpha=0.6)

    # 두 집계 상자 아래 빈 공간에 둔다. 상자 사이(y 0.475~0.640)는 합류 화살표가
    # 지나가서 어디에 놓아도 겹친다.
    ax.text(0.6225, 0.305, "family 를 나누는 이유", ha="center", va="center",
            fontsize=7.4, color=INK_SECONDARY, style="italic")
    ax.text(0.6225, 0.278, "한 통에 넣으면 1회 추출 view 가 chunk 오탐에 끌려간다",
            ha="center", va="center", fontsize=6.8, color=INK_MUTED)

    # ---------------------------------------------------------------- 판정
    box(ax, 0.735, 0.500, 0.115, 0.115, "max", "두 family 합류", ec=C_DEFENSE, lw=1.6)
    arrow(ax, (0.701, 0.697), (0.731, 0.585), rad=-0.12, color=C_DEFENSE)
    arrow(ax, (0.701, 0.418), (0.731, 0.530), rad=0.12, color=INK_MUTED, alpha=0.7)

    box(ax, 0.885, 0.500, 0.103, 0.115, "점수 > τ ?", "τ 는 자유 변수",
        ec=INK_PRIMARY, lw=1.4)
    arrow(ax, (0.851, 0.5575), (0.881, 0.5575), color=C_DEFENSE)

    box(ax, 0.885, 0.700, 0.103, 0.070, "차단", ec=C_ALTDIFF, lw=1.4)
    arrow(ax, (0.9365, 0.616), (0.9365, 0.697), color=C_ALTDIFF)
    ax.text(0.9425, 0.6565, "위험", ha="left", va="center", fontsize=7.5, color=C_ALTDIFF)

    # ---------------------------------------------------------------- 생성기 (우회로)
    box(ax, 0.735, 0.135, 0.135, 0.095, "AltDiffusion", "학생3", ec=C_ALTDIFF, lw=1.3)
    box(ax, 0.905, 0.135, 0.083, 0.095, "이미지", ec=BASELINE, lw=1.0, ls=(0, (3, 2)),
        tc=INK_MUTED)
    arrow(ax, (0.871, 0.1825), (0.901, 0.1825), color=C_ALTDIFF)
    # 점수>τ 아래 → 왼쪽으로 빠져 생성기 위로. 이미지 상자(y 0.135~0.230)를 피해
    # y=0.290 에서 가로지른다.
    ax.plot([0.9365, 0.9365], [0.496, 0.290], color=C_ALTDIFF, lw=1.2, zorder=1)
    ax.plot([0.9365, 0.8025], [0.290, 0.290], color=C_ALTDIFF, lw=1.2, zorder=1)
    arrow(ax, (0.8025, 0.290), (0.8025, 0.233), color=C_ALTDIFF)
    ax.text(0.9425, 0.395, "통과", ha="left", va="center", fontsize=7.5, color=C_ALTDIFF)

    # 원본 우회로 — 이 그림의 핵심 결정
    ax.plot([0.074, 0.074], [0.598, 0.183], color=C_ALTDIFF, lw=1.3,
            linestyle=(0, (5, 3)), zorder=1)
    arrow(ax, (0.074, 0.183), (0.731, 0.183), color=C_ALTDIFF, lw=1.3,
          ls=(0, (5, 3)))
    ax.text(0.402, 0.196, "생성기는 원본 프롬프트를 받는다 — 정규화본은 안전 필터만 본다",
            ha="center", va="bottom", fontsize=8.2, color=C_ALTDIFF)

    # ---------------------------------------------------------------- 조건
    ax.text(0.012, 0.088, "ABLATION 조건 = view 부분집합 (별도 로직 아님)",
            fontsize=8.2, color=INK_PRIMARY, fontweight="bold")
    conds = [("baseline", "원본"), ("normalization_only", "원본 + 정규화본"),
             ("chunk_only", "원본 + chunk"),
             ("combined", "원본 + 정규화본 + chunk + 정규화 chunk")]
    for i, (name, views) in enumerate(conds):
        x = 0.012 + i * 0.248
        ax.text(x, 0.046, name, fontsize=7.8, color=C_DEFENSE, fontweight="bold")
        ax.text(x, 0.018, views, fontsize=7.2, color=INK_MUTED)

    suptitle(fig, "방어 파이프라인 구조",
             "정규화가 주 방어 · chunk 는 대조군 · 점수는 view 당 1회, 판정은 사후")
    fig.text(0.008, 0.002,
             "점선 = 대조 경로. chunk 는 분리도를 복원하지만(ORACLE 이 baseline 을 최대 33pp 앞섬) "
             "어느 chunk 인지 점수로 고를 수 없어 4개 크기 전부 게이트 실패 (PHASE1_GATE §20-23).",
             fontsize=6.9, color=INK_MUTED, ha="left", va="bottom")
    return fig


def main() -> int:
    fig = build()
    save(fig, "figure_architecture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
