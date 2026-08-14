# -*- coding: utf-8 -*-
"""Figure F — 가시성과 희귀도가 생성 결과와 어떻게 연관되는가 (논문 X.10 / H6)

두 패널로 나눈 이유
-------------------
  왼쪽  AltDiffusion 핵심표현 가시성별 개념 등장률.
        절단이 유용성을 떨어뜨린다는 것을 보인다. none 은 96건 전부 0 이다.

  오른쪽 희귀도 x 가시성 교차.
        왼쪽만 보면 "희귀 표현이 잘려서 안 그려진다" 로 읽힌다. 가시성을 full 로
        고정한 열에서도 격차가 그대로면 절단으로 설명되지 않는다는 뜻이고,
        그것이 이 그림의 확인 포인트다.

사용법:  python analysis/figures/figure_f_generation_outcome.py
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
import _style as S

import matplotlib.pyplot as plt

REPO = S.REPO
VIS_ORDER = ["full", "partial", "none"]
VIS_KO = {"full": "전부 보임", "partial": "일부만", "none": "안 보임"}


def norm(v):
    return str(v).strip().lower()


def load():
    def rd(p):
        with open(p, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    lab_p = REPO / "evaluation" / "generation" / "image_labels.csv"
    gen_p = REPO / "evaluation" / "generation" / "generation_results.csv"
    for p in (lab_p, gen_p, S.RESULTS_CSV):
        if not p.exists():
            sys.exit(f"입력이 없습니다: {p.relative_to(REPO)}")

    prompts = {r["prompt_id"]: r for r in rd(REPO / "benchmarks" / "prompts" / "prompts.csv")}
    g2p = {r["generation_id"]: r["prompt_id"] for r in rd(gen_p)}
    labels = {g2p[r["generation_id"]]: r for r in rd(lab_p)}
    advis = {r["prompt_id"]: r["key_visibility"]
             for r in S.load_results() if S.ALTDIFF_ID_PART in r["model_id"]}

    rows = []
    for pid, lab in labels.items():
        rows.append({
            "rarity": prompts[pid]["rarity_label"],
            "vis": advis[pid],
            "drawn": norm(lab["concept_present_final"]) == "true",
        })
    return rows


def rate(rows):
    if not rows:
        return None, 0
    return sum(1 for r in rows if r["drawn"]) / len(rows), len(rows)


# 패널 A 전용 색. 이 패널은 일반·희귀를 합친 값이라 카테고리 색(파랑=일반,
# 주황=희귀)을 쓰면 범례와 충돌해 "희귀만 그린 것" 으로 오독된다. 어느 슬롯도
# 아닌 중립 회색을 쓴다.
C_NEUTRAL = "#6f6d67"


def head(ax, title, sub):
    ax.set_title(title, fontsize=11, color=S.INK_PRIMARY, loc="left",
                 pad=30, fontweight="bold")
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=8.5,
            color=S.INK_SECONDARY, va="bottom")


def panel_visibility(ax, rows):
    head(ax, "A   핵심표현 가시성별 개념 등장률",
         "생성 모델이 핵심표현을 어디까지 읽었는가 — 안 보이면 한 건도 그려지지 않았다")

    ys = range(len(VIS_ORDER))
    for i, v in enumerate(VIS_ORDER):
        r, n = rate([x for x in rows if x["vis"] == v])
        ax.barh(i, r * 100, height=0.46, color=C_NEUTRAL, zorder=3,
                edgecolor=S.SURFACE, linewidth=2)
        drawn = sum(1 for x in rows if x["vis"] == v and x["drawn"])
        ax.text(r * 100 + 1.2, i, f"{r*100:.1f}%   ({drawn}/{n})",
                va="center", fontsize=9.5, color=S.INK_PRIMARY)

    ax.set_yticks(list(ys))
    ax.set_yticklabels([VIS_KO[v] for v in VIS_ORDER], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 42)
    ax.set_xlabel("개념 등장률 (%)", fontsize=9.5, color=S.INK_SECONDARY)
    style_axes(ax)


def panel_rarity(ax, rows):
    head(ax, "B   희귀도 × 가시성",
         "‘전부 보임’ 열에서도 격차가 유지된다 — 절단으로 설명되지 않는다")

    groups = ["full", "all"]
    labels_x = ["전부 보임\n(절단 배제)", "전체\n(절단 포함)"]
    width = 0.34
    for j, rar in enumerate(("common", "rare")):
        color = S.C_SGUARD if rar == "common" else S.C_ALTDIFF
        for i, g in enumerate(groups):
            sub = [x for x in rows if x["rarity"] == rar and (g == "all" or x["vis"] == "full")]
            r, n = rate(sub)
            x = i + (j - 0.5) * (width + 0.03)
            ax.bar(x, r * 100, width=width, color=color, zorder=3,
                   edgecolor=S.SURFACE, linewidth=2)
            drawn = sum(1 for s in sub if s["drawn"])
            ax.text(x, r * 100 + 1.4, f"{r*100:.1f}%", ha="center", fontsize=9.5,
                    color=S.INK_PRIMARY, fontweight="bold")
            ax.text(x, r * 100 + 4.6, f"{drawn}/{n}", ha="center", fontsize=8,
                    color=S.INK_MUTED)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels_x, fontsize=9.5)
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(0, 62)
    ax.set_ylabel("개념 등장률 (%)", fontsize=9.5, color=S.INK_SECONDARY, labelpad=18)
    style_axes(ax, horizontal=False)


def style_axes(ax, horizontal=True):
    # 축 스타일은 _style.tidy_axes 가 갖는다. 여기서 다시 정의하면 팔레트를 복사해
    # 뒀다가 조용히 어긋났던 figure_a 와 같은 일이 반복된다. 격자만 덧붙인다.
    S.tidy_axes(ax)
    ax.tick_params(length=0)
    ax.grid(axis="x" if horizontal else "y", color=S.GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def main() -> int:
    S.setup()
    rows = load()

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.635, bottom=0.155, wspace=0.32)

    S.suptitle(fig,
               "Figure F   희귀 표현은 잘리지 않아도 그려지지 않는다",
               "프롬프트 432개 · AltDiffusion-m18 · 개념 등장 = 평가자 2인 합의 라벨 "
               "(κ 0.755)")

    panel_visibility(axes[0], rows)
    panel_rarity(axes[1], rows)

    fig.legend(handles=[
        plt.Line2D([], [], marker="s", ls="", ms=8, color=S.C_SGUARD, label="일반 표현"),
        plt.Line2D([], [], marker="s", ls="", ms=8, color=S.C_ALTDIFF, label="희귀 표현"),
    ], loc="upper right", bbox_to_anchor=(0.985, 0.955), frameon=False,
        fontsize=9.5, labelcolor=S.INK_SECONDARY, ncol=2, columnspacing=1.4,
        handletextpad=0.5)

    S.save(fig, "figure_f_generation_outcome")

    print("\n  [요약]")
    for v in VIS_ORDER:
        r, n = rate([x for x in rows if x["vis"] == v])
        print(f"    가시성 {v:8} {r*100:5.1f}%  (n={n})")
    for rar in ("common", "rare"):
        r_all, n_all = rate([x for x in rows if x["rarity"] == rar])
        r_f, n_f = rate([x for x in rows if x["rarity"] == rar and x["vis"] == "full"])
        print(f"    {rar:8} 전체 {r_all*100:5.1f}% (n={n_all})   "
              f"full 한정 {r_f*100:5.1f}% (n={n_f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
