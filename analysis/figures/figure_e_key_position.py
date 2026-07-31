#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_e_key_position.py — Figure E

절단의 메커니즘을 그대로 보여준다.

각 프롬프트를 핵심 표현의 마지막 토큰 위치(key_end_pretrunc)에 점으로 찍고,
그 컴포넌트의 content 예산을 세로선으로 긋는다. 선 오른쪽에 있는 점이 곧
'핵심 표현을 못 본' 프롬프트다. 위치·길이 조건이 이 좌표를 어떻게 밀어내는지가
한눈에 보인다.

색은 key_visibility 의 순서형 램프다 (none 이 가장 옅고 full 이 가장 진하다).
경계에 걸친 partial 이 세로선 주변에만 나타나는 것이 확인 포인트다.

x 축은 로그 스케일이다. over_limit 은 예산의 3~5배라 선형으로 놓으면
정작 중요한 경계 부근이 뭉개진다.

사용법:
    .venv\\Scripts\\python.exe analysis/figures/figure_e_key_position.py
"""

from __future__ import annotations

import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _style as S

# key_visibility 순서형 램프 (blue). 라이트 서피스에서 가장 옅은 단계도
# 2:1 대비를 넘겨야 하므로 step 250 보다 옅게 내려가지 않는다 (palette.md).
VIS_COLOR = {"none": "#86b6ef", "partial": "#3987e5", "full": "#0d366b"}
VIS_SIZE = {"none": 26, "partial": 46, "full": 22}

PANELS = [("sg_cap", "A   SGuard  조건 2 (user content 예산 77)"),
          ("altdiff", "B   AltDiffusion  native (content 예산 75)")]
JITTER = 0.26


def main() -> int:
    S.setup()
    cond = S.split_conditions(S.load_results())
    rng = random.Random(20260801)      # 지터 재현성

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.755, bottom=0.13, wspace=0.07)

    for ax, (key, title) in zip(axes, PANELS):
        rows = cond[key]
        budget = int(rows[0]["content_token_budget"])

        # 위치 조건을 마커 모양으로 — 색(가시성)과 분리해 이중 인코딩
        for pos, marker in (("front", "o"), ("middle", "^"), ("back", "s")):
            for vis in ("full", "partial", "none"):
                xs, ys = [], []
                for r in rows:
                    if r["position_level"] != pos or r["key_visibility"] != vis:
                        continue
                    xs.append(int(r["key_end_pretrunc"]) + 1)   # 로그 스케일용 +1
                    ys.append(S.LENGTHS.index(r["length_level"])
                              + rng.uniform(-JITTER, JITTER))
                if xs:
                    ax.scatter(xs, ys, s=VIS_SIZE[vis], marker=marker,
                               color=VIS_COLOR[vis], alpha=0.85, linewidths=0.6,
                               edgecolors=S.SURFACE, zorder=3)

        ax.axvline(budget, color=S.INK_PRIMARY, lw=1.6, ls="--", zorder=4)
        ax.text(budget * 1.06, -0.62, f"content 예산 {budget}", fontsize=9,
                color=S.INK_PRIMARY, va="center", fontweight="bold")
        ax.text(budget * 1.06, -0.40, "이 오른쪽 = 핵심 표현 소실", fontsize=8.5,
                color=S.INK_SECONDARY, va="center")

        for i in range(len(S.LENGTHS)):
            ax.axhline(i, color=S.GRIDLINE, lw=0.8, zorder=0)

        ax.set_xscale("log")
        # 눈금에 그 패널의 실제 예산을 넣는다. 두 컴포넌트의 예산이 다르므로
        # (77 vs 75) 공통 눈금을 쓰면 점선 위치와 눈금이 어긋나 보인다.
        ax.set_xticks([1, 3, 10, 30, budget, 200, 500])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim(0.8, 600)
        ax.set_ylim(-0.85, len(S.LENGTHS) - 0.4)
        ax.set_yticks(range(len(S.LENGTHS)))
        ax.set_yticklabels(S.LENGTHS)
        ax.set_xlabel("핵심 표현의 마지막 토큰 위치 (절단 전, 로그 스케일)",
                      fontsize=9, color=S.INK_SECONDARY)
        ax.set_title(title, fontsize=10.5, color=S.INK_PRIMARY, pad=10,
                     loc="left", fontweight="bold")
        S.tidy_axes(ax, hide=("top", "right", "left"))
        ax.tick_params(colors=S.INK_SECONDARY, labelsize=9)

    handles = [Line2D([], [], marker="o", ls="", ms=7, color=VIS_COLOR[v],
                      label=f"key_visibility  {v}") for v in ("full", "partial", "none")]
    handles += [Line2D([], [], marker=m, ls="", ms=7, color=S.INK_MUTED,
                       label=f"위치  {p}") for p, m in
                (("front", "o"), ("middle", "^"), ("back", "s"))]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.955),
               frameon=False, fontsize=9, labelcolor=S.INK_SECONDARY, ncol=2,
               handletextpad=0.5, columnspacing=1.4)

    S.suptitle(fig,
               "Figure E   핵심 표현의 토큰 위치와 절단 경계",
               "점 하나가 프롬프트 하나. 세로 점선이 그 컴포넌트의 content 토큰 예산이며, "
               "오른쪽에 놓인 점이 핵심 표현을 잃은 경우다.")
    S.save(fig, "figure_e_key_position")

    print("\n  [요약]")
    for key, title in PANELS:
        rows = cond[key]
        budget = int(rows[0]["content_token_budget"])
        over = sum(1 for r in rows if int(r["key_end_pretrunc"]) >= budget)
        lost = sum(1 for r in rows if r["key_visibility"] != "full")
        print(f"    {title.split('  ')[1]:14} 예산 {budget:>3}  "
              f"key_end >= 예산 {over:>3}개  /  full 아님 {lost:>3}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
