# -*- coding: utf-8 -*-
"""
_style.py — KoRareGuard-T2I / Student 5 / 방어 그림 공용 스타일

색은 학생2 의 `analysis/figures/_style.py` 에서 **가져다 쓴다.** 복사하지 않는다 —
그 파일 첫머리에 "팔레트 값을 그림마다 복사하면 조용히 어긋난다"고 적혀 있고,
같은 논문에 나란히 실릴 그림이라 어긋나면 바로 보인다.

한 가지만 덮어쓴다: 폰트. 그의 setup() 은 `Malgun Gothic` (Windows) 을 고정하는데
macOS·Linux 에는 없어서 한글이 네모로 깨진다. 설치된 것 중 첫 번째를 고른다.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# 학생2 의 검증된 팔레트 (dataviz 기본 인스턴스, CVD·대비 검사 통과)
sys.path.insert(0, str(REPO / "analysis" / "figures"))
from _style import (  # noqa: E402,F401  — 재수출
    BASELINE, BLUE_RAMP, C_ALTDIFF, C_SGUARD, CMAP_SEQ, GRIDLINE,
    INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE,
)

FIG_DIR = Path(__file__).resolve().parent

# 방어 계층 전용 색. 카테고리 슬롯 1·2 는 이미 안전필터·생성기가 쓰고 있으므로,
# 방어는 그 둘과 구분되는 세 번째 색을 쓴다 (같은 램프의 짙은 끝 — 계열은 유지).
C_DEFENSE = "#104281"
C_MUTEDBOX = "#f2f1ec"

# 한글이 있는 폰트 중 실제로 설치된 것을 고른다.
_KO_CANDIDATES = ("Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic",
                  "NanumGothic", "Noto Sans KR", "DejaVu Sans")


def korean_font() -> str:
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in _KO_CANDIDATES:
        if name in installed:
            return name
    return "sans-serif"


def setup() -> None:
    plt.rcParams.update({
        "font.family": korean_font(),
        "axes.unicode_minus": False,     # 한글 폰트에 U+2212 가 없다
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
    })


def suptitle(fig, title: str, subtitle: str = "") -> None:
    h = fig.get_figheight()
    y_title = 1.0 - 0.30 / h
    y_sub = y_title - 0.26 / h
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
