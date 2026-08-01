#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
span_smoke_test.py — KoRareGuard-T2I / Student 2 / PHASE 2

key_span.analyze_key_span() 이 실제로 맞는지 사람이 눈으로 확인하기 위한
token-by-token 검증표를 만든다.

432 행 전수 분석 전에 반드시 이 단계를 통과해야 한다.
여기서 자동 계산과 사람 판단이 어긋나면 이후 모든 지표가 무의미하다.

샘플은 무작위가 아니라 의도적으로 고른다:
  - byte-level BPE 에서 한글이 여러 토큰으로 쪼개지는 경우
  - common / rare 대응쌍
  - 절단 경계에 걸치는 경우
  - 프롬프트는 잘렸지만 key 는 살아있는 경우 (prompt_truncated != key truncated)

사용법:
    .venv\\Scripts\\python.exe analysis/tokenizer/span_smoke_test.py
"""

from __future__ import annotations

import csv
import io
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
log = logging.getLogger("span_smoke_test")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from key_span import InputPolicy, analyze_key_span, KeySpanResult  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"
REPORT_MD = Path(__file__).resolve().parent / "span_smoke_test_report.md"

SGUARD_MODEL_ID = "SamsungSDS-Research/SGuard-ContentFilter-2B-v1"
ALTDIFF_MODEL_ID = "BAAI/AltDiffusion-m18"

# 본 연구가 정의한 experimental token cap (SGuard native limit 이 아님)
EXPERIMENTAL_TOKEN_CAP = 77

# 의도적으로 고른 검증 샘플. 각각 무엇을 확인하려는지 명시한다.
SAMPLES = [
    ("SAFE_CULT_02_RARE_SHORT_FRONT",
     "byte-level BPE 글자 분할 (강강술래). 절단 없음 — 기본 매핑 검증"),
    ("SAFE_CULT_02_COMMON_SHORT_FRONT",
     "위와 같은 concept 의 common 대응쌍 — key 토큰 수 비교"),
    ("SAFE_CULT_04_RARE_SHORT_BACK",
     "최단 key (택견, 2 글자) — 경계 처리"),
    ("UNSAFE_VIOL_13_RARE_NEAR_LIMIT_MIDDLE",
     "SGuard@77 경계 구간 (43/48 생존) — partial 이 나오는지"),
    ("UNSAFE_VIOL_13_RARE_NEAR_LIMIT_BACK",
     "H2a 후보: SGuard 는 못 보고 AltDiffusion 은 보는 조건"),
    ("UNSAFE_VIOL_13_COMMON_NEAR_LIMIT_BACK",
     "위의 common 대응쌍"),
    ("SAFE_CULT_01_RARE_OVER_LIMIT_FRONT",
     "★ prompt_truncated=True 인데 key_visibility=full 이어야 하는 경우"),
    ("SAFE_CULT_01_RARE_OVER_LIMIT_MIDDLE",
     "양쪽 모두 key 절단 — none 이 나오는지"),
]

SEP = "=" * 100
SUB = "-" * 100


def load_prompts() -> dict[str, dict[str, str]]:
    if not PROMPTS_CSV.exists():
        raise FileNotFoundError(f"prompts.csv 를 찾을 수 없습니다: {PROMPTS_CSV}")
    with open(PROMPTS_CSV, encoding="utf-8-sig", newline="") as f:
        return {r["prompt_id"]: r for r in csv.DictReader(f)}


def build_policies(sg_tok: Any, ad_tok: Any) -> list[tuple[str, Any, InputPolicy, str]]:
    """
    3 개 분석 조건. AltDiffusion 의 77 은 하드코딩하지 않고 runtime 값을 쓴다.
    SGuard 의 cap 은 content 토큰에 적용하므로 add_special_tokens=False 다
    (chat template 의 prefix/suffix 1,480 토큰은 별도 보존 대상).
    """
    ad_native = int(ad_tok.model_max_length)
    return [
        ("SGuard native", sg_tok,
         InputPolicy(name="native", model_id=SGUARD_MODEL_ID, model_role="safety",
                     add_special_tokens=False, cap=None, cap_kind="none",
                     note="SGuard native context 131,072 — 본 데이터셋에서는 절단 없음"),
         "model native context (131072)"),
        (f"SGuard constrained_{EXPERIMENTAL_TOKEN_CAP}", sg_tok,
         InputPolicy(name=f"constrained_{EXPERIMENTAL_TOKEN_CAP}", model_id=SGUARD_MODEL_ID,
                     model_role="safety", add_special_tokens=False,
                     cap=EXPERIMENTAL_TOKEN_CAP, cap_kind="user_content",
                     note="연구가 정의한 experimental cap. SGuard native limit 아님"),
         "experimental (user content budget)"),
        ("AltDiffusion native", ad_tok,
         InputPolicy(name="native", model_id=ALTDIFF_MODEL_ID, model_role="generator",
                     add_special_tokens=True, cap=ad_native, cap_kind="native",
                     note="tokenizer.model_max_length 실측값"),
         "tokenizer.model_max_length (runtime)"),
    ]


def print_token_table(res: KeySpanResult, window: int = 6) -> None:
    """key 주변과 절단 경계만 보여준다. 전체를 다 찍으면 사람이 못 읽는다."""
    rows = res.token_rows
    if not rows:
        return
    focus = set()
    for i in range(res.key_start_pretrunc - window, res.key_end_pretrunc + window + 1):
        if 0 <= i < len(rows):
            focus.add(i)
    if res.prompt_truncated:
        for i in range(res.total_tokens_used - window, res.total_tokens_used + window):
            if 0 <= i < len(rows):
                focus.add(i)
    for i in range(0, min(3, len(rows))):
        focus.add(i)

    print(f"    {'idx':>5} {'id':>7} {'token':>20} {'offset':>12} {'text':>8} "
          f"{'KEY':>4} {'keep':>5}")
    prev = -99
    for i in sorted(focus):
        if i - prev > 1:
            print(f"    {'...':>5}")
        r = rows[i]
        mark = "◆" if r["overlaps_key"] else ""
        keep = "O" if r["retained"] else "X"
        cut = "  <-- CUT" if (res.prompt_truncated and i == res.total_tokens_used) else ""
        print(f"    {r['idx']:>5} {r['token_id']:>7} {r['token']!r:>20} "
              f"{str(r['offset']):>12} {r['text']!r:>8} {mark:>4} {keep:>5}{cut}")
        prev = i


def main() -> int:
    from transformers import AutoTokenizer

    prompts = load_prompts()
    log.info("prompts.csv rows: %d", len(prompts))

    sg_tok = AutoTokenizer.from_pretrained(SGUARD_MODEL_ID)
    ad_tok = AutoTokenizer.from_pretrained(ALTDIFF_MODEL_ID, subfolder="tokenizer")
    policies = build_policies(sg_tok, ad_tok)

    md: list[str] = ["# PHASE 2 — Key Span 수작업 검증 보고서", ""]
    md.append(f"- experimental token cap (SGuard): **{EXPERIMENTAL_TOKEN_CAP}** "
              f"(user content budget, native limit 아님)")
    md.append(f"- AltDiffusion native max length: **{int(ad_tok.model_max_length)}** (runtime)")
    md.append("")

    all_results: list[KeySpanResult] = []
    n_error = 0

    for pid, why in SAMPLES:
        row = prompts.get(pid)
        if row is None:
            log.error("prompt_id 없음: %s — 건너뜀", pid)
            continue

        raw, key = row["raw_prompt"], row["key_expression"]
        print(f"\n{SEP}\n{pid}\n  검증 목적: {why}\n{SEP}")
        print(f"  rarity={row['rarity_label']}  length={row['length_level']}  "
              f"position={row['position_level']}  safety={row['safety_label']}")
        print(f"  key   : {key!r}  ({len(key)} chars)")
        preview = raw if len(raw) <= 160 else raw[:80] + " … " + raw[-70:]
        print(f"  prompt: {preview!r}  ({len(raw)} chars)")

        md.append(f"## {pid}")
        md.append(f"*{why}*")
        md.append("")
        md.append(f"- key: `{key}` / rarity: {row['rarity_label']} / "
                  f"length: {row['length_level']} / position: {row['position_level']}")
        md.append("")
        md.append("| condition | pretrunc | key tok | key span | center | used | "
                  "prompt_trunc | retained | ratio(tok) | ratio(char) | visibility |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")

        for label, tok, policy, src in policies:
            res = analyze_key_span(tok, raw, key, policy, prompt_id=pid, max_length_source=src)
            all_results.append(res)
            if res.analysis_status != "ok":
                n_error += 1
                print(f"\n  [{label}]  ERROR: {res.error_message}")
                md.append(f"| {label} | ERROR: {res.error_message} |||||||||| ")
                continue

            print(f"\n  [{label}]  cap={res.max_length_effective}  ({res.max_length_source})")
            print(f"    pretrunc={res.total_tokens_pretrunc}  key_tokens={res.key_token_count_original}"
                  f"  span=[{res.key_start_pretrunc}:{res.key_end_pretrunc}]"
                  f"  center={res.key_center_ratio:.3f}")
            print(f"    used={res.total_tokens_used}  prompt_truncated={res.prompt_truncated}"
                  f"  retained={res.key_tokens_retained}/{res.key_token_count_original}"
                  f"  ratio_tok={res.key_retention_ratio:.3f}"
                  f"  ratio_char={res.key_retention_ratio_char:.3f}"
                  f"  -> {res.key_visibility.upper()}")
            if res.tokens_sharing_char_offset:
                print(f"    한 글자를 나눠 가진 token {res.tokens_sharing_char_offset} 개"
                      f"{'  / 절단이 글자 중간을 지남!' if res.key_split_mid_character else ''}")
            print_token_table(res)
            if res.prompt_truncated:
                rt = res.removed_text
                print(f"    removed_text ({len(rt)} chars): "
                      f"{(rt[:70] + ' …') if len(rt) > 70 else rt!r}")

            md.append(f"| {label} | {res.total_tokens_pretrunc} | {res.key_token_count_original} | "
                      f"[{res.key_start_pretrunc}:{res.key_end_pretrunc}] | {res.key_center_ratio:.3f} | "
                      f"{res.total_tokens_used} | {res.prompt_truncated} | "
                      f"{res.key_tokens_retained} | {res.key_retention_ratio:.3f} | "
                      f"{res.key_retention_ratio_char:.3f} | **{res.key_visibility}** |")
        md.append("")

    # ---------- 요약 ----------
    print(f"\n{SEP}\n검증 요약\n{SEP}")
    print(f"  분석 행 수      : {len(all_results)}")
    print(f"  error 행        : {n_error}")

    ok = [r for r in all_results if r.analysis_status == "ok"]
    mid_split = [r for r in ok if r.key_split_mid_character]
    trunc_but_full = [r for r in ok if r.prompt_truncated and r.key_visibility == "full"]
    print(f"  prompt 잘렸지만 key full : {len(trunc_but_full)}  <- 이 구분이 살아있어야 함")
    print(f"  절단이 글자 중간 통과     : {len(mid_split)}")

    print("\n  visibility 분포:")
    for label, _, policy, _ in policies:
        sub = [r for r in ok if r.model_id == policy.model_id and r.input_policy == policy.name]
        dist: dict[str, int] = {}
        for r in sub:
            dist[r.key_visibility] = dist.get(r.key_visibility, 0) + 1
        print(f"    {label:28} {dist}")

    md.append("## 요약")
    md.append(f"- 분석 행: {len(all_results)} / error: {n_error}")
    md.append(f"- prompt 잘렸지만 key full: {len(trunc_but_full)}")
    md.append(f"- 절단이 글자 중간 통과: {len(mid_split)}")

    REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n  보고서 저장: {REPORT_MD}")
    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())
