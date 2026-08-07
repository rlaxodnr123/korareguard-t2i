#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
length_calibration.py — KoRareGuard-T2I / Student 2 / PHASE 3

432 개 프롬프트 전체를 3 개 분석 조건으로 사전 토큰화하여

  1) length_level (short / near_limit / over_limit) 이 실제 토큰 경계를
     제대로 나타내는지 판정하고
  2) 설계가 실제로 신호(H1 / H2a / H2b 표본)를 만들어내는지 확인한다.

이미지 생성도 모델 추론도 하지 않는다. tokenizer 만 쓰므로 GPU 가 필요 없다.

이 스크립트는 tokenization_results.csv 를 만들지 않는다. 그것은 PHASE 5 다.
여기서는 "PHASE 5 로 가도 되는가" 만 판정한다.

주의:
  - length_level 은 설계 단계에서 문자 길이로 정한 label 이며
    tokenizer 로 캘리브레이션된 값이 아니다. 그것이 실제로 어떤 토큰 분포를
    만드는지가 이 스크립트의 측정 대상이다.
  - EXPERIMENTAL_TOKEN_CAP 은 본 연구가 정의한 값이며
    SGuard 의 native maximum context length 가 아니다.

사용법:
    .venv\\Scripts\\python.exe analysis/tokenizer/length_calibration.py
"""

from __future__ import annotations

import csv
import io
import logging
import statistics as st
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
log = logging.getLogger("length_calibration")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from key_span import (  # noqa: E402
    InputPolicy, analyze_key_span,
    VISIBILITY_FULL, VISIBILITY_PARTIAL, VISIBILITY_NONE, STATUS_OK,
    ROLE_TEXT_SAFETY, ROLE_GENERATOR,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.common import config  # noqa: E402  — 모델 id / revision 의 팀 공용 SSOT

PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"
REPORT_MD = Path(__file__).resolve().parent / "length_calibration_report.md"

SGUARD_MODEL_ID = config.SGUARD_MODEL_ID
SGUARD_REVISION = config.SGUARD_REVISION
ALTDIFF_MODEL_ID = config.ALTDIFF_MODEL_ID
ALTDIFF_REVISION = config.ALTDIFF_REVISION

EXPERIMENTAL_TOKEN_CAP = 77          # 연구가 정의한 cap. native limit 아님

LENGTHS = ["short", "near_limit", "over_limit"]
POSITIONS = ["front", "middle", "back"]
VIS_ORDER = [VISIBILITY_FULL, VISIBILITY_PARTIAL, VISIBILITY_NONE]

SEP = "=" * 96
SUB = "-" * 96

COND_SG_NATIVE = "SGuard native"
COND_SG_CAP = f"SGuard constrained_{EXPERIMENTAL_TOKEN_CAP}"
COND_AD = "AltDiffusion native"


def describe(vals: list[int]) -> dict[str, Any]:
    v = sorted(vals)
    n = len(v)
    q = st.quantiles(v, n=4) if n >= 4 else [v[0], v[n // 2], v[-1]]
    return {"n": n, "min": v[0], "p25": q[0], "median": st.median(v),
            "mean": sum(v) / n, "p75": q[2], "max": v[-1]}


def fmt_desc(d: dict[str, Any]) -> str:
    return (f"{d['n']:>4} {d['min']:>6} {d['p25']:>7.0f} {d['median']:>8.0f} "
            f"{d['mean']:>8.1f} {d['p75']:>7.0f} {d['max']:>6}")


def sguard_formatted_tokens(tok: Any, prompt: str) -> int:
    """SGuard 가 실제 추론 시 받는 전체 입력 토큰 수 (chat template 포함)."""
    msgs = [{"role": "user", "prompt": prompt}]
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if len(out) > 0 and isinstance(out[0], (list, tuple)):
        out = out[0]
    return len(out)


def main() -> int:
    from transformers import AutoTokenizer

    if not PROMPTS_CSV.exists():
        log.error("prompts.csv 없음: %s", PROMPTS_CSV)
        return 1
    with open(PROMPTS_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"prompts.csv rows: {len(rows)}")

    sg_tok = AutoTokenizer.from_pretrained(SGUARD_MODEL_ID, revision=SGUARD_REVISION)
    ad_tok = AutoTokenizer.from_pretrained(ALTDIFF_MODEL_ID, revision=ALTDIFF_REVISION,
                                           subfolder="tokenizer")
    ad_declared = int(ad_tok.model_max_length)
    # diffusers 는 tokenizer(padding='max_length', max_length=77, truncation=True) 를 쓴다.
    # HF 는 special token 자리를 먼저 확보한 뒤 content 를 자르므로 실제 content 예산은
    # 77 이 아니라 77 - 2 = 75 다. 세 조건을 모두 content 토큰 공간에서 정의해야
    # 위치·retention 지표를 서로 비교할 수 있으므로 add_special_tokens=False 로 맞춘다.
    # (analyze_tokens.py 와 같은 정의여야 한다. 예전에 여기만 77/True 로 남아 있어
    #  AltDiffusion full 이 1 많고 H2a 가 1 많게 나왔다.)
    ad_n_special = len(ad_tok("", add_special_tokens=True)["input_ids"])
    ad_content_budget = ad_declared - ad_n_special

    policies = [
        (COND_SG_NATIVE, sg_tok,
         InputPolicy(name="native", model_id=SGUARD_MODEL_ID, model_role=ROLE_TEXT_SAFETY,
                     add_special_tokens=False, cap=None, cap_kind="none")),
        (COND_SG_CAP, sg_tok,
         InputPolicy(name=f"constrained_{EXPERIMENTAL_TOKEN_CAP}", model_id=SGUARD_MODEL_ID,
                     model_role=ROLE_TEXT_SAFETY, add_special_tokens=False,
                     cap=EXPERIMENTAL_TOKEN_CAP, cap_kind="user_content")),
        (COND_AD, ad_tok,
         InputPolicy(name="native", model_id=ALTDIFF_MODEL_ID, model_role=ROLE_GENERATOR,
                     add_special_tokens=False, cap=ad_content_budget, cap_kind="native",
                     declared_max_length=ad_declared,
                     special_tokens_reserved=ad_n_special)),
    ]

    t0 = time.time()
    results: dict[str, dict[str, Any]] = {c: {} for c, _, _ in policies}
    n_error = 0
    for r in rows:
        for cond, tok, pol in policies:
            res = analyze_key_span(tok, r["raw_prompt"], r["key_expression"], pol,
                                   prompt_id=r["prompt_id"])
            if res.analysis_status != STATUS_OK:
                n_error += 1
                log.error("%s / %s : %s", r["prompt_id"], cond, res.error_message)
            results[cond][r["prompt_id"]] = res

    # SGuard 실제 전체 입력 (chat template 포함)
    formatted = {r["prompt_id"]: sguard_formatted_tokens(sg_tok, r["raw_prompt"]) for r in rows}
    elapsed = time.time() - t0

    meta = {r["prompt_id"]: r for r in rows}
    md: list[str] = ["# PHASE 3 — Length Calibration & Signal Preview", ""]
    # 실행 시간은 콘솔에만 찍는다. 리포트에 넣으면 실행할 때마다 값이 달라져서,
    # "재생성 후 diff 가 비어야 stale 이 아니다" 라는 검사가 매번 오탐을 낸다.
    md.append(f"- 프롬프트 {len(rows)}개 × 3조건 = {len(rows)*3}행, error {n_error}건")
    md.append(f"- SGuard experimental token cap: **{EXPERIMENTAL_TOKEN_CAP}** "
              f"(user content budget, native limit 아님)")
    md.append(f"- AltDiffusion declared max length: **{ad_declared}** (runtime), "
              f"special token {ad_n_special}개를 빼면 content 예산 **{ad_content_budget}**")
    md.append("")

    print(f"\n{SEP}\nPHASE 3 — Length Calibration & Signal Preview\n{SEP}")
    print(f"  {len(rows)} prompts x 3 conditions = {len(rows)*3} rows / "
          f"error {n_error} / {elapsed:.1f}s")

    # ---------- [1] 토큰 분포 ----------
    print(f"\n{SEP}\n[1] length_level 별 절단 전 토큰 분포\n{SEP}")
    md.append("## 1. length_level 별 절단 전 토큰 분포\n")
    for cond, _, pol in policies:
        cap = pol.cap
        print(f"\n  --- {cond}  (cap={cap}) ---")
        print(f"  {'level':12} {'n':>4} {'min':>6} {'p25':>7} {'median':>8} "
              f"{'mean':>8} {'p75':>7} {'max':>6}")
        md.append(f"### {cond} (cap={cap})\n")
        md.append("| level | n | min | p25 | median | mean | p75 | max |")
        md.append("|---|---|---|---|---|---|---|---|")
        for lv in LENGTHS:
            vals = [results[cond][p].total_tokens_pretrunc
                    for p in results[cond] if meta[p]["length_level"] == lv]
            d = describe(vals)
            print(f"  {lv:12} {fmt_desc(d)}")
            md.append(f"| {lv} | {d['n']} | {d['min']} | {d['p25']:.0f} | {d['median']:.0f} | "
                      f"{d['mean']:.1f} | {d['p75']:.0f} | {d['max']} |")
        md.append("")

    # ---------- [2] cap 초과율 ----------
    print(f"\n{SEP}\n[2] cap 초과율 (프롬프트 전체가 잘리는 비율)\n{SEP}")
    md.append("## 2. cap 초과율\n")
    md.append("| level | SGuard native | SGuard@77 | AltDiffusion@77 |")
    md.append("|---|---|---|---|")
    print(f"  {'level':12} " + " ".join(f"{c:>24}" for c, _, _ in policies))
    for lv in LENGTHS:
        cells = []
        for cond, _, _ in policies:
            sub = [results[cond][p] for p in results[cond] if meta[p]["length_level"] == lv]
            k = sum(1 for x in sub if x.prompt_truncated)
            cells.append(f"{k:>3}/{len(sub):<3} ({k/len(sub)*100:5.1f}%)")
        print(f"  {lv:12} " + " ".join(f"{c:>24}" for c in cells))
        md.append(f"| {lv} | " + " | ".join(cells) + " |")
    md.append("")

    # ---------- [3] SGuard 실제 전체 입력 ----------
    print(f"\n{SEP}\n[3] SGuard 실제 전체 입력 (chat template 포함) — 조건1 절단 여부\n{SEP}")
    md.append("## 3. SGuard 실제 전체 입력 (chat template 포함)\n")
    md.append("| level | min | median | max | native context 131,072 초과 |")
    md.append("|---|---|---|---|---|")
    print(f"  {'level':12} {'min':>7} {'median':>8} {'max':>7}   native 131072 초과")
    for lv in LENGTHS:
        vals = [formatted[p] for p in formatted if meta[p]["length_level"] == lv]
        over = sum(1 for v in vals if v > 131072)
        print(f"  {lv:12} {min(vals):>7} {st.median(vals):>8.0f} {max(vals):>7}   "
              f"{over}/{len(vals)}")
        md.append(f"| {lv} | {min(vals)} | {st.median(vals):.0f} | {max(vals)} | "
                  f"{over}/{len(vals)} |")
    ovh = [formatted[p] - results[COND_SG_NATIVE][p].total_tokens_pretrunc for p in formatted]
    print(f"\n  template overhead: min {min(ovh)}  median {st.median(ovh):.0f}  max {max(ovh)}")
    md.append(f"\n- template overhead: min {min(ovh)} / median {st.median(ovh):.0f} / max {max(ovh)}")
    md.append("")

    # ---------- [4] key visibility ----------
    print(f"\n{SEP}\n[4] key_visibility 분포 — length x position\n{SEP}")
    md.append("## 4. key_visibility 분포 (length × position)\n")
    for cond, _, _ in policies:
        print(f"\n  --- {cond} ---")
        print(f"  {'level':12} {'position':9} {'full':>6} {'partial':>8} {'none':>6}")
        md.append(f"### {cond}\n")
        md.append("| level | position | full | partial | none |")
        md.append("|---|---|---|---|---|")
        for lv in LENGTHS:
            for pos in POSITIONS:
                sub = [results[cond][p] for p in results[cond]
                       if meta[p]["length_level"] == lv and meta[p]["position_level"] == pos]
                c = Counter(x.key_visibility for x in sub)
                print(f"  {lv:12} {pos:9} {c.get(VISIBILITY_FULL,0):>6} "
                      f"{c.get(VISIBILITY_PARTIAL,0):>8} {c.get(VISIBILITY_NONE,0):>6}")
                md.append(f"| {lv} | {pos} | {c.get(VISIBILITY_FULL,0)} | "
                          f"{c.get(VISIBILITY_PARTIAL,0)} | {c.get(VISIBILITY_NONE,0)} |")
        md.append("")

    # ---------- [5] mismatch preview ----------
    print(f"\n{SEP}\n[5] directional mismatch 미리보기 — H2a / H2b 표본이 나오는가\n{SEP}")
    md.append("## 5. directional mismatch (H2a / H2b 표본)\n")
    md.append("| 비교 | A 둘다 봄 | B SGuard✓ AltDiff✗ (H2b) | C SGuard✗ AltDiff✓ (H2a) | D 둘다 못봄 |")
    md.append("|---|---|---|---|---|")
    for sg_cond in (COND_SG_NATIVE, COND_SG_CAP):
        cnt = Counter()
        by_cell = defaultdict(Counter)
        for p in results[sg_cond]:
            s = results[sg_cond][p].key_visibility
            a = results[COND_AD][p].key_visibility
            s_ok = s == VISIBILITY_FULL
            a_ok = a == VISIBILITY_FULL
            case = "A" if (s_ok and a_ok) else "B" if (s_ok and not a_ok) \
                else "C" if (not s_ok and a_ok) else "D"
            cnt[case] += 1
            by_cell[(meta[p]["length_level"], meta[p]["position_level"])][case] += 1
        n = sum(cnt.values())
        print(f"\n  --- {sg_cond}  vs  {COND_AD} ---")
        for case, label in (("A", "둘 다 봄"), ("B", "SGuard O AltDiff X  <- H2b"),
                            ("C", "SGuard X AltDiff O  <- H2a"), ("D", "둘 다 못 봄")):
            print(f"    {case}  {label:32} {cnt[case]:>4} ({cnt[case]/n*100:5.1f}%)")
        md.append(f"| {sg_cond} vs AltDiff | {cnt['A']} | **{cnt['B']}** | "
                  f"**{cnt['C']}** | {cnt['D']} |")
        print(f"    셀별 H2a(C) 분포: "
              f"{ {k: v['C'] for k, v in sorted(by_cell.items()) if v['C']} }")
        print(f"    셀별 H2b(B) 분포: "
              f"{ {k: v['B'] for k, v in sorted(by_cell.items()) if v['B']} }")
    md.append("")

    # ---------- [6] 판정 ----------
    print(f"\n{SEP}\n[6] 판정\n{SEP}")
    md.append("## 6. 판정\n")
    verdicts: list[str] = []

    for cond, _, pol in policies:
        if pol.cap is None:
            continue
        sub_near = [results[cond][p] for p in results[cond]
                    if meta[p]["length_level"] == "near_limit"]
        rate = sum(1 for x in sub_near if x.prompt_truncated) / len(sub_near)
        verdicts.append(f"{cond}: near_limit 프롬프트 절단율 {rate*100:.1f}%"
                        + ("  (경계 조건으로 작동)" if 0.1 < rate < 0.9
                           else "  (경계가 아니라 전부/전무)"))

    for cond, _, _ in policies:
        vis = Counter(results[cond][p].key_visibility for p in results[cond])
        verdicts.append(f"{cond}: visibility {dict(vis)}")

    n_partial = sum(1 for c, _, _ in policies
                    for p in results[c]
                    if results[c][p].key_visibility == VISIBILITY_PARTIAL)
    verdicts.append(f"partial(경계에 정확히 걸린 key) 총 {n_partial}행")

    n_split = sum(1 for c, _, _ in policies
                  for p in results[c] if results[c][p].key_split_mid_character)
    verdicts.append(f"절단이 key 내부 글자 중간을 지난 행: {n_split}")

    n_trunc_full = sum(1 for c, _, _ in policies for p in results[c]
                       if results[c][p].prompt_truncated
                       and results[c][p].key_visibility == VISIBILITY_FULL)
    verdicts.append(f"prompt 잘렸지만 key full: {n_trunc_full}행 "
                    f"(prompt_truncated != key truncated 구분이 살아있음)")
    verdicts.append(f"error 행: {n_error}")

    for v in verdicts:
        print(f"  - {v}")
        md.append(f"- {v}")

    REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n  보고서 저장: {REPORT_MD}")
    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())
