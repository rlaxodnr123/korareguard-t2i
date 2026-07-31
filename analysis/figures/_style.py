# -*- coding: utf-8 -*-
"""
_style.py — KoRareGuard-T2I / Student 2 / 그림 공용 스타일과 로더

팔레트 값을 그림마다 복사하면 조용히 어긋난다. 여기 한 곳에만 둔다.

색은 dataviz 스킬의 검증된 기본 인스턴스(라이트 서피스 #fcfcfb)를 쓴다.
  카테고리 슬롯 1(blue) / 2(orange) — validate_palette.js 전 항목 PASS
    CVD 분리 ΔE 24.7 (protan) / 정상시야 ΔE 33.6 / 대비 둘 다 3:1 이상
  순차 램프는 같은 문서의 blue 100→700 (연속 크기 인코딩용)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[2]
RESULTS_CSV = REPO / "analysis" / "truncation" / "tokenization_results.csv"
FIG_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 색
C_SGUARD = "#2a78d6"       # 카테고리 슬롯 1 — 안전 필터
C_ALTDIFF = "#eb6834"      # 카테고리 슬롯 2 — 생성기
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# 순차 램프 (blue 100→700). 연속 크기 인코딩 전용 — 카테고리에 쓰지 않는다.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("kg_blue", BLUE_RAMP)

# ---------------------------------------------------------------- 조건 / 축 순서
SGUARD_ID_PART = "SGuard"
ALTDIFF_ID_PART = "AltDiffusion"
POLICY_NATIVE = "native"
POLICY_CAP = "constrained_77"

LENGTHS = ["short", "near_limit", "over_limit"]
POSITIONS = ["front", "middle", "back"]
VIS_ORDER = ["full", "partial", "none"]


def setup() -> None:
    plt.rcParams.update({
        "font.family": "Malgun Gothic",   # 한글 key expression 표시용
        "axes.unicode_minus": False,      # 한글 폰트에 U+2212 가 없다
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
    })


def load_results() -> list[dict[str, Any]]:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{RESULTS_CSV} 가 없습니다. analyze_tokens.py --full 을 먼저 실행하세요.")
    with open(RESULTS_CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_conditions(rows: list[dict]) -> dict[str, list[dict]]:
    """세 분석 조건으로 나눈다. 키: sg_native / sg_cap / altdiff"""
    out: dict[str, list[dict]] = {"sg_native": [], "sg_cap": [], "altdiff": []}
    for r in rows:
        if SGUARD_ID_PART in r["model_id"]:
            out["sg_native" if r["input_policy"] == POLICY_NATIVE else "sg_cap"].append(r)
        else:
            out["altdiff"].append(r)
    return out


def tidy_axes(ax, *, hide=("top", "right")) -> None:
    for s in hide:
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        if s not in hide:
            ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3)


def suptitle(fig, title: str, subtitle: str = "") -> None:
    """제목/부제를 그림 높이와 무관하게 일정 간격으로 배치한다.

    y 를 고정 비율(0.975 / 0.938)로 주면 그림이 낮을 때 두 줄이 겹친다.
    인치 단위로 환산해서 놓는다.
    """
    h = fig.get_figheight()
    y_title = 1.0 - 0.30 / h        # 위에서 0.30 inch
    y_sub = y_title - 0.26 / h      # 제목에서 0.26 inch 아래
    fig.suptitle(title, fontsize=13.5, color=INK_PRIMARY, x=0.008, y=y_title,
                 ha="left", va="top", fontweight="bold")
    if subtitle:
        fig.text(0.008, y_sub, subtitle, fontsize=9, color=INK_SECONDARY,
                 ha="left", va="top")


def save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  저장: {p.relative_to(REPO)}")
