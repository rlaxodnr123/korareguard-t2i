#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_budget_consumption.py — KoRareGuard-T2I / Student 2 / RQ-T3

RQ-T3: 희귀 표현으로 바꾸면 prompt 의 token budget 소비가 얼마나 달라지는가?

이 분석은 filler 가 통제된 이후에만 의미가 있다. 같은 concept x length x position
안에서 common 과 rare 의 차이가 key 표현 하나뿐이어야, 총 토큰 수의 차이를
희귀도 효과로 읽을 수 있다. 벤치마크 초기 revision 에서는 filler 가 rarity 와
함께 변해서 이 측정이 불가능했다.

측정:
  1. paired 총 토큰 델타 (rare - common). filler 가 같으므로 key 효과만 남는다.
  2. 그 델타가 key 토큰 델타로 설명되는가 (설명되지 않으면 통제가 덜 된 것)
  3. key 표현이 각 컴포넌트의 content 예산에서 차지하는 비율
  4. 희귀도 전환이 예산 초과 여부를 뒤집는 경우가 있는가

사용법:
    .venv\\Scripts\\python.exe analysis/truncation/analyze_budget_consumption.py
"""

from __future__ import annotations

import csv
import io
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "analysis" / "truncation" / "tokenization_results.csv"
REPORT = REPO / "analysis" / "truncation" / "budget_consumption_report.md"

SG, AD = "SGuard", "AltDiffusion"
LENGTHS = ["short", "near_limit", "over_limit"]
POSITIONS = ["front", "middle", "back"]
SEP = "=" * 88


def describe(v: list[float]) -> str:
    s = sorted(v)
    return (f"n={len(s):>3}  median {st.median(s):+7.2f}  mean {sum(s)/len(s):+7.2f}  "
            f"[{s[0]:+.0f}, {s[-1]:+.0f}]")


def main() -> int:
    with open(RESULTS, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # 조건별로 나눈다. PASS 1 지표는 policy 와 무관하므로 tokenizer 당 하나만 쓴다.
    comp: dict[str, list[dict[str, Any]]] = {SG: [], AD: []}
    for r in rows:
        if SG in r["model_id"]:
            if r["input_policy"] == "native":
                comp[SG].append(r)
        else:
            comp[AD].append(r)

    md: list[str] = ["# RQ-T3 — 희귀 표현의 token budget 소비", ""]
    md.append("filler 가 통제된 벤치마크에서만 유효한 분석이다. 같은 concept x length x")
    md.append("position 안에서 common 과 rare 의 차이가 key 표현뿐이므로, 총 토큰 수의")
    md.append("차이를 희귀도 효과로 읽을 수 있다.")
    md.append("")

    print(SEP)
    print("RQ-T3 — 희귀 표현의 token budget 소비")
    print(SEP)

    # ---------------- 1. paired 총 토큰 델타 ----------------
    print(f"\n{SEP}\n[1] paired 총 토큰 델타 (rare - common)\n{SEP}")
    md.append("## 1. paired 총 토큰 델타 (rare - common)\n")
    md.append("| tokenizer | 조건 | n | median | mean | range |")
    md.append("|---|---|---|---|---|---|")

    deltas: dict[tuple[str, str], list[int]] = defaultdict(list)
    key_deltas: dict[tuple[str, str], list[int]] = defaultdict(list)
    for tk, sub in comp.items():
        idx = {(r["concept_id"], r["length_level"], r["position_level"],
                r["rarity_label"]): r for r in sub}
        for (c, lv, pos, rar), r in idx.items():
            if rar != "rare":
                continue
            cm = idx.get((c, lv, pos, "common"))
            if cm is None:
                continue
            d = int(r["total_tokens_pretrunc"]) - int(cm["total_tokens_pretrunc"])
            kd = int(r["key_token_count_original"]) - int(cm["key_token_count_original"])
            deltas[(tk, "all")].append(d)
            deltas[(tk, lv)].append(d)
            key_deltas[(tk, "all")].append(kd)

    for tk in (SG, AD):
        for scope in ["all"] + LENGTHS:
            v = deltas[(tk, scope)]
            if not v:
                continue
            print(f"  {tk:14} {scope:12} {describe(v)}")
            s = sorted(v)
            md.append(f"| {tk} | {scope} | {len(v)} | {st.median(v):+.2f} | "
                      f"{sum(v)/len(v):+.2f} | {s[0]:+d} … {s[-1]:+d} |")
    md.append("")

    # ---------------- 2. 델타가 key 로 설명되는가 ----------------
    print(f"\n{SEP}\n[2] 총 델타가 key 토큰 델타로 설명되는가 (통제 확인)\n{SEP}")
    md.append("## 2. 총 델타가 key 토큰 델타로 설명되는가\n")
    for tk in (SG, AD):
        tot, key = deltas[(tk, "all")], key_deltas[(tk, "all")]
        resid = [t - k for t, k in zip(tot, key)]
        exact = sum(1 for x in resid if x == 0)
        print(f"  {tk:14} 총 델타 median {st.median(tot):+.1f}  "
              f"key 델타 median {st.median(key):+.1f}  "
              f"잔차 median {st.median(resid):+.1f}  잔차 0 인 쌍 {exact}/{len(resid)}")
        md.append(f"- **{tk}** 총 델타 median {st.median(tot):+.1f}, "
                  f"key 델타 median {st.median(key):+.1f}, "
                  f"잔차 median {st.median(resid):+.1f} "
                  f"(잔차 0: {exact}/{len(resid)})")
    print("\n  잔차는 key 경계에서 생기는 토큰 병합 때문이며, filler 가 통제되지 않았다면")
    print("  훨씬 크게 벌어진다. 이 값이 0 근처라는 것이 통제가 유효함을 뒷받침한다.")
    md.append("\n잔차는 key 경계의 토큰 병합에서 나온다. filler 가 통제되지 않았다면")
    md.append("훨씬 크게 벌어지므로, 0 근처라는 것이 통제 유효성의 근거가 된다.\n")

    # ---------------- 3. key 가 예산에서 차지하는 비율 ----------------
    print(f"\n{SEP}\n[3] key 표현이 content 예산에서 차지하는 비율\n{SEP}")
    md.append("## 3. key 표현이 content 예산에서 차지하는 비율\n")
    md.append("| tokenizer | 예산 | rarity | key 토큰 median | 예산 대비 |")
    md.append("|---|---|---|---|---|")
    budgets = {SG: 77, AD: 75}          # 조건 2 / native
    for tk in (SG, AD):
        b = budgets[tk]
        for rar in ("common", "rare"):
            v = sorted(int(r["key_token_count_original"]) for r in comp[tk]
                       if r["rarity_label"] == rar)
            m = st.median(v)
            print(f"  {tk:14} 예산 {b:>3}  {rar:7} key 토큰 median {m:>5.1f}  "
                  f"예산의 {m/b*100:5.1f}%   [{v[0]}, {v[-1]}]")
            md.append(f"| {tk} | {b} | {rar} | {m:.1f} | {m/b*100:.1f}% |")
    md.append("")

    # ---------------- 4. 희귀도가 예산 초과 여부를 뒤집는가 ----------------
    print(f"\n{SEP}\n[4] 희귀도 전환이 '예산 초과' 여부를 뒤집는 경우\n{SEP}")
    md.append("## 4. 희귀도 전환이 예산 초과 여부를 뒤집는 경우\n")
    md.append("| tokenizer | 예산 | 뒤집힌 쌍 | 방향 |")
    md.append("|---|---|---|---|")
    for tk in (SG, AD):
        b = budgets[tk]
        idx = {(r["concept_id"], r["length_level"], r["position_level"],
                r["rarity_label"]): r for r in comp[tk]}
        flip = Counter()
        cells = Counter()
        for (c, lv, pos, rar), r in idx.items():
            if rar != "rare":
                continue
            cm = idx.get((c, lv, pos, "common"))
            if cm is None:
                continue
            # key 의 마지막 토큰이 예산 안에 들어오는가
            ro = int(r["key_end_pretrunc"]) < b
            co = int(cm["key_end_pretrunc"]) < b
            if ro != co:
                d = "rare 만 보임" if ro else "common 만 보임"
                flip[d] += 1
                cells[(lv, pos, d)] += 1
        total = sum(flip.values())
        print(f"  {tk:14} 예산 {b:>3}  뒤집힌 쌍 {total}/216   {dict(flip)}")
        if cells:
            print(f"                 셀별 {dict(cells)}")
        md.append(f"| {tk} | {b} | {total}/216 | {dict(flip) or '-'} |")
    md.append("")

    print(f"\n{SEP}")
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"  보고서 저장: {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
