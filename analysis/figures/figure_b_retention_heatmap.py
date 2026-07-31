#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_b_retention_heatmap.py — Figure B

RQ-T5 / RQ-T7: 길이와 위치에 따라 핵심 표현이 얼마나 남는가?

length_level x position_level 9칸에 대해 key_retention_ratio 평균을 색으로,
full / partial / none 개수를 숫자로 함께 보여준다. 평균만 보면 "절반쯤 남았다"가
전부 partial 인지, full 과 none 이 섞인 것인지 구분되지 않는다.

두 패널을 같은 색 스케일(0~1)로 그려서 컴포넌트 간 비교가 가능하게 한다.

사용법:
    .venv\\Scripts\\python.exe analysis/figures/figure_b_retention_heatmap.py
"""

from __future__ import annotations

import io
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np

import _style as S


PANELS = [("sg_cap", "A   SGuard  조건2 (user content 예산 77)"),
          ("altdiff", "B   AltDiffusion  native (content 예산 75)")]


def cell_stats(rows: list[dict]) -> tuple[float, Counter]:
    vals = [float(r["key_retention_ratio"]) for r in rows]
    return (sum(vals) / len(vals) if vals else float("nan"),
            Counter(r["key_visibility"] for r in rows))


def main() -> int:
    S.setup()
    cond = S.split_conditions(S.load_results())

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))
    fig.subplots_adjust(left=0.115, right=0.885, top=0.775, bottom=0.09, wspace=0.16)

    mesh = None
    for ax, (key, title) in zip(axes, PANELS):
        rows = cond[key]
        grid = np.zeros((len(S.LENGTHS), len(S.POSITIONS)))
        notes: dict[tuple[int, int], Counter] = {}
        for i, lv in enumerate(S.LENGTHS):
            for j, pos in enumerate(S.POSITIONS):
                sub = [r for r in rows
                       if r["length_level"] == lv and r["position_level"] == pos]
                grid[i, j], notes[(i, j)] = cell_stats(sub)

        mesh = ax.imshow(grid, cmap=S.CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
        # 셀 사이 2px 표면 간격
        ax.set_xticks(np.arange(-0.5, len(S.POSITIONS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(S.LENGTHS), 1), minor=True)
        ax.grid(which="minor", color=S.SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

        for (i, j), c in notes.items():
            v = grid[i, j]
            # 어두운 칸에는 밝은 글자 — 대비 확보
            ink = "#ffffff" if v > 0.55 else S.INK_PRIMARY
            ax.text(j, i - 0.16, f"{v:.2f}", ha="center", va="center",
                    fontsize=13, color=ink, fontweight="bold")
            parts = "  ".join(f"{k[0]}{c.get(k, 0)}" for k in S.VIS_ORDER)
            ax.text(j, i + 0.21, parts, ha="center", va="center",
                    fontsize=8, color=ink)

        ax.set_xticks(range(len(S.POSITIONS)))
        ax.set_xticklabels(S.POSITIONS)
        ax.set_yticks(range(len(S.LENGTHS)))
        ax.set_yticklabels(S.LENGTHS)
        ax.set_title(title, fontsize=10.5, color=S.INK_PRIMARY, pad=10,
                     loc="left", fontweight="bold")
        S.tidy_axes(ax, hide=("top", "right", "bottom", "left"))
        ax.tick_params(colors=S.INK_SECONDARY, labelsize=9.5)

    cb = fig.colorbar(mesh, ax=axes, fraction=0.028, pad=0.02)
    cb.set_label("key_retention_ratio 평균", fontsize=9, color=S.INK_SECONDARY)
    cb.ax.tick_params(colors=S.INK_MUTED, labelsize=8.5)
    cb.outline.set_visible(False)

    S.suptitle(fig,
               "Figure B   길이·위치별 핵심 표현 보존율",
               "칸 안 큰 숫자는 key_retention_ratio 평균, 작은 글자는 "
               "full / partial / none 개수 (각 칸 48개). 두 패널은 같은 색 스케일이다.")
    S.save(fig, "figure_b_retention_heatmap")

    print("\n  [요약]")
    for key, title in PANELS:
        rows = cond[key]
        c = Counter(r["key_visibility"] for r in rows)
        print(f"    {title.split('  ')[1]:36} {dict(c)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
