#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_a_fragmentation.py — KoRareGuard-T2I / Student 2 / Figure A

RQ-T1 / RQ-T2 (H1): 희귀 표현이 흔한 표현보다 더 잘게 분절되는가?

이 그림의 요점은 하나다.
  같은 24개 concept pair 를 두 지표로 보면 결론의 부호가 뒤집힌다.

  왼쪽  raw key 토큰 수 차이 (rare - common)      -> 대부분 0 아래
  오른쪽 tokens_per_character 차이 (rare - common) -> 대부분 0 위

왼쪽만 보면 "희귀 표현이 토큰을 덜 쓴다"가 되는데, 그건 희귀도가 아니라
문자열 길이를 재는 것이다. 벤치마크의 common 표현은 서술구(평균 13.4자)이고
rare 표현은 합성명사(평균 6.0자)라서, 길이로 나누지 않으면 길이 효과가
희귀도 효과로 둔갑한다.

이 지표는 key expression 자체에서 계산되므로 filler confound 의 영향을 받지 않는다.
(front/middle/back 세 위치에서 median 이 동일함을 별도로 확인했다)

입력 : analysis/truncation/tokenization_results.csv  (analyze_tokens.py --full 산출물)
출력 : analysis/figures/figure_a_fragmentation.{png,pdf}

사용법:
    .venv\\Scripts\\python.exe analysis/figures/figure_a_fragmentation.py
"""

from __future__ import annotations

import csv
import io
import statistics as st
import sys
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "analysis" / "truncation" / "tokenization_results.csv"
OUT_DIR = Path(__file__).resolve().parent

SGUARD_KEY = "SGuard"
ALTDIFF_KEY = "AltDiffusion"

# --- 검증된 팔레트 (dataviz skill 기본 인스턴스, 라이트 서피스 #fcfcfb) ---
# 카테고리 슬롯 1, 2. validate_palette.js 전 항목 PASS
#   CVD 분리 ΔE 24.7 (protan) / 정상시야 ΔE 33.6 / 대비 둘 다 3:1 이상
C_SGUARD = "#2a78d6"      # slot 1 blue
C_ALTDIFF = "#eb6834"     # slot 2 orange
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

DODGE = 0.19              # 두 시리즈 세로 어긋냄
MARKER_SIZE = 46          # >= 8px


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "Malgun Gothic",     # 한글 key expression 표시용
        "axes.unicode_minus": False,        # 한글 폰트에 U+2212 없음 -> ASCII 하이픈
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
    })


def load_pairs() -> list[dict]:
    """concept 별 common/rare 의 key 분절 지표를 모은다.

    key_token_count_original 과 key_tokens_per_character 는 PASS 1 지표라
    input_policy 와 무관하다. 따라서 tokenizer 당 대표 행 하나만 쓰면 된다.
    """
    if not SRC.exists():
        raise FileNotFoundError(
            f"{SRC} 가 없습니다. 먼저 analyze_tokens.py --full 을 실행하세요.")
    with open(SRC, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # (concept, rarity, tokenizer) -> 지표
    #
    # 주의: 첫 행을 집으면 안 된다. SGuard 는 byte-level BPE 라 key 앞의 공백이
    # 토큰에 합쳐지는지에 따라 key 토큰 수가 달라지고, front 위치는 key 가 문자열
    # 맨 앞이라 앞 공백이 없다. 그래서 48개 (concept, rarity) 중 5개에서
    # front 값이 middle/back 과 1~2 토큰 다르다.
    # 모델이 실제로 받는 입력에서는 front 도 'Prompt: ' 뒤라 앞 공백이 있으므로
    # middle/back 값이 실제와 맞다. 9개 행(3 length x 3 position)의 중앙값을 쓰면
    # 다수인 middle/back 값이 잡힌다.
    acc: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        tk = SGUARD_KEY if SGUARD_KEY in r["model_id"] else ALTDIFF_KEY
        if tk == SGUARD_KEY and r["input_policy"] != "native":
            continue          # 같은 tokenizer 의 두 policy 중 하나만
        e = acc.setdefault((r["concept_id"], r["rarity_label"], tk), {
            "key": r["key_expression"], "chars": int(r["key_character_count"]),
            "safety": r["safety_label"], "tokens": [], "tpc": [],
        })
        e["tokens"].append(int(r["key_token_count_original"]))
        e["tpc"].append(float(r["key_tokens_per_character"]))
    box = {k: {**v, "tokens": st.median(v["tokens"]), "tpc": st.median(v["tpc"])}
           for k, v in acc.items()}

    concepts = sorted({c for c, _, _ in box})
    out = []
    for c in concepts:
        row = {"concept_id": c,
               "safety": box[(c, "rare", SGUARD_KEY)]["safety"],
               "rare_key": box[(c, "rare", SGUARD_KEY)]["key"],
               "common_key": box[(c, "common", SGUARD_KEY)]["key"]}
        for tk in (SGUARD_KEY, ALTDIFF_KEY):
            rr, cc = box[(c, "rare", tk)], box[(c, "common", tk)]
            row[f"d_tok_{tk}"] = rr["tokens"] - cc["tokens"]
            row[f"d_tpc_{tk}"] = rr["tpc"] - cc["tpc"]
        out.append(row)
    return out


def summarize(pairs: list[dict], field: str) -> str:
    v = [p[field] for p in pairs]
    above = sum(1 for x in v if x > 0)
    return f"median {st.median(v):+.2f}   0 초과 {above}/{len(v)}"


def panel(ax, pairs, field_fmt, title, subtitle, xlabel, annotate_fmt) -> None:
    ys = range(len(pairs))
    ax.axvline(0, color=BASELINE, lw=1.4, zorder=1)
    for i in ys:                                   # concept 별 가이드
        ax.axhline(i, color=GRIDLINE, lw=0.6, zorder=0)

    for tk, color, off in ((SGUARD_KEY, C_SGUARD, +DODGE),
                           (ALTDIFF_KEY, C_ALTDIFF, -DODGE)):
        xs = [p[field_fmt.format(tk=tk)] for p in pairs]
        yy = [i + off for i in ys]
        # 0 까지 잇는 얇은 줄기 -> 부호와 크기가 같이 읽힌다
        for x, y in zip(xs, yy):
            ax.plot([0, x], [y, y], color=color, lw=1.6, alpha=0.55,
                    solid_capstyle="round", zorder=2)
        ax.scatter(xs, yy, s=MARKER_SIZE, color=color, zorder=3,
                   edgecolors=SURFACE, linewidths=1.4)   # 겹침 방지 2px 링

    ax.set_title(title, fontsize=11, color=INK_PRIMARY, pad=14, loc="left",
                 fontweight="bold")
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=8.5,
            color=INK_SECONDARY, va="bottom")
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    # 아래쪽에 두 행 분량 여백을 둔다. 요약 문구를 마지막 concept 행 위에 겹치지 않게
    # 놓기 위한 것이다 (axes 좌표로 놓으면 행 수가 바뀔 때 다시 겹친다).
    ax.set_ylim(-0.8, len(pairs) + 1.5)
    ax.invert_yaxis()
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3)
    ax.grid(False)

    # 요약은 텍스트 잉크로. 시리즈 색은 옆의 점이 담당한다 (텍스트에 시리즈 색 금지)
    x0 = ax.get_xlim()[0]
    span = ax.get_xlim()[1] - x0
    for k, (tk, color) in enumerate(((SGUARD_KEY, C_SGUARD), (ALTDIFF_KEY, C_ALTDIFF))):
        y = len(pairs) + 0.25 + 0.62 * k
        ax.plot([x0 + span * 0.012], [y], marker="o", ms=5.5, color=color,
                clip_on=False, zorder=4)
        ax.text(x0 + span * 0.045, y, f"{tk}   {annotate_fmt(tk)}",
                fontsize=8.5, color=INK_SECONDARY, va="center")


def main() -> int:
    setup_style()
    pairs = load_pairs()
    # 오른쪽 패널이 깔끔한 기울기로 읽히도록 정규화 지표로 정렬
    pairs.sort(key=lambda p: p[f"d_tpc_{SGUARD_KEY}"], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 8.2), sharey=True)
    fig.subplots_adjust(left=0.235, right=0.985, top=0.86, bottom=0.11, wspace=0.10)

    panel(axes[0], pairs, "d_tok_{tk}",
          "A   원시 key 토큰 수", "rare - common (음수 = 희귀 표현이 토큰을 적게 씀)",
          "토큰 수 차이", lambda tk: summarize(pairs, f"d_tok_{tk}"))
    panel(axes[1], pairs, "d_tpc_{tk}",
          "B   문자당 토큰 수로 정규화", "rare - common (양수 = 희귀 표현이 더 잘게 분절)",
          "tokens per character 차이", lambda tk: summarize(pairs, f"d_tpc_{tk}"))

    labels = [f"{p['concept_id']}   {p['rare_key']}" for p in pairs]
    axes[0].set_yticks(range(len(pairs)))
    axes[0].set_yticklabels(labels, fontsize=8.5, color=INK_SECONDARY)
    for lbl, p in zip(axes[0].get_yticklabels(), pairs):
        if p["safety"] == "unsafe":
            lbl.set_color(INK_PRIMARY)     # unsafe concept 을 진하게 (색 아닌 굵기/명도로 구분)

    fig.suptitle("Figure A   희귀 한국어 표현의 토큰 분절 — 정규화 여부로 결론이 뒤집힌다",
                 fontsize=13.5, color=INK_PRIMARY, x=0.008, y=0.972,
                 ha="left", fontweight="bold")
    fig.text(0.008, 0.933,
             "concept 24쌍 paired 비교. y축 라벨은 concept_id 와 rare 표현 "
             "(진한 라벨 = unsafe concept). 두 패널의 concept 순서는 동일하다.",
             fontsize=9, color=INK_SECONDARY, ha="left")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=7, color=C_SGUARD,
                               label="SGuard (byte-level BPE)"),
                        Line2D([], [], marker="o", ls="", ms=7, color=C_ALTDIFF,
                               label="AltDiffusion (SentencePiece)")],
               loc="upper right", bbox_to_anchor=(0.985, 0.965), frameon=False,
               fontsize=9.5, labelcolor=INK_SECONDARY, handletextpad=0.5,
               ncol=1, columnspacing=1.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"figure_a_fragmentation.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  저장: {p.relative_to(REPO)}")

    print("\n  [요약]")
    for f_, nm in (("d_tok_{tk}", "원시 토큰 수"), ("d_tpc_{tk}", "tokens/char")):
        for tk in (SGUARD_KEY, ALTDIFF_KEY):
            print(f"    {nm:12} {tk:14} {summarize(pairs, f_.format(tk=tk))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
