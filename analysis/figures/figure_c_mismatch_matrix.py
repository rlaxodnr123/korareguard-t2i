#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_c_mismatch_matrix.py — Figure C

RQ-T6 / RQ-T8: 안전 필터와 생성기가 같은 프롬프트를 같게 읽는가?

두 조건을 나란히 놓는 것이 이 그림의 요점이다. 방향이 정반대로 갈린다.

  조건 1 (SGuard native)   B행(SGuard만 봄) 만 나오고 C행은 0
  조건 2 (SGuard@77)       C행(AltDiff만 봄) 만 나오고 B행은 0

SGuard 는 같은 한국어에 AltDiffusion 보다 토큰을 약 1.6배 쓴다. 그래서
content 예산을 77 로 제한하면 항상 AltDiffusion(75) 보다 적은 텍스트를 본다.
B행이 구조적으로 0 이 되는 이유이며, H2a 는 조건 2 에서만, H2b 는 조건 1 에서만
측정된다는 설계 근거다.

사용법:
    .venv\\Scripts\\python.exe analysis/figures/figure_c_mismatch_matrix.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np

import _style as S

PANELS = [("sg_native", "A   조건 1   SGuard native  vs  AltDiffusion"),
          ("sg_cap", "B   조건 2   SGuard 예산 77  vs  AltDiffusion")]

# (row, col) = (SGuard 봄?, AltDiffusion 봄?)
CELL_LABEL = {
    (0, 0): ("A", "둘 다 봄", ""),
    (0, 1): ("B", "안전 필터만 봄", "H2b"),
    (1, 0): ("C", "생성기만 봄", "H2a"),
    (1, 1): ("D", "둘 다 못 봄", ""),
}


def crosstab(sg_rows: list[dict], ad_by_pid: dict[str, dict]) -> np.ndarray:
    """행: SGuard 봄/못 봄, 열: AltDiffusion 봄/못 봄"""
    m = np.zeros((2, 2), dtype=int)
    for r in sg_rows:
        s = 0 if r["key_visibility"] == "full" else 1
        a = 0 if ad_by_pid[r["prompt_id"]]["key_visibility"] == "full" else 1
        m[s, a] += 1
    return m


def main() -> int:
    S.setup()
    cond = S.split_conditions(S.load_results())
    ad_by_pid = {r["prompt_id"]: r for r in cond["altdiff"]}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.9))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.735, bottom=0.06, wspace=0.20)

    mats = {}
    for ax, (key, title) in zip(axes, PANELS):
        m = crosstab(cond[key], ad_by_pid)
        mats[key] = m
        total = m.sum()

        ax.imshow(m, cmap=S.CMAP_SEQ, vmin=0, vmax=total, aspect="auto")
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color=S.SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

        for i in range(2):
            for j in range(2):
                n = m[i, j]
                ink = "#ffffff" if n / total > 0.45 else S.INK_PRIMARY
                tag, desc, hyp = CELL_LABEL[(i, j)]
                ax.text(j, i - 0.30, tag, ha="center", va="center",
                        fontsize=10, color=ink, fontweight="bold")
                ax.text(j, i - 0.08, f"{n}", ha="center", va="center",
                        fontsize=25, color=ink, fontweight="bold")
                ax.text(j, i + 0.13, f"{n / total * 100:.1f}%", ha="center",
                        va="center", fontsize=10, color=ink)
                ax.text(j, i + 0.28, desc, ha="center", va="center",
                        fontsize=9, color=ink)
                if hyp:
                    ax.text(j, i + 0.40, hyp, ha="center", va="center",
                            fontsize=9.5, color=ink, fontweight="bold")

        ax.set_xticks([0, 1]); ax.set_xticklabels(["생성기 봄", "생성기 못 봄"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["안전 필터\n봄", "안전 필터\n못 봄"])
        ax.set_title(title, fontsize=10.5, color=S.INK_PRIMARY, pad=10,
                     loc="left", fontweight="bold")
        S.tidy_axes(ax, hide=("top", "right", "bottom", "left"))
        ax.tick_params(colors=S.INK_SECONDARY, labelsize=9.5)

    S.suptitle(fig,
               "Figure C   안전 필터와 생성기의 핵심 표현 가시성 불일치",
               "프롬프트 432개. '봄' = key_visibility full. 조건에 따라 불일치 방향이 "
               "정반대로 갈린다 — 한 조건에서 두 가설을 함께 검정할 수 없다.")
    S.save(fig, "figure_c_mismatch_matrix")

    print("\n  [요약]")
    for key, title in PANELS:
        m = mats[key]
        print(f"    {title.split('   ')[1]:12} A={m[0,0]:>4}  B(H2b)={m[0,1]:>4}  "
              f"C(H2a)={m[1,0]:>4}  D={m[1,1]:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
