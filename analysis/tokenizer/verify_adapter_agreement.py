#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_adapter_agreement.py — KoRareGuard-T2I / Student 2

내 독립 구현(key_span.py)과 팀 공용 어댑터(SGuardAdapter.tokenize)가
432 x 2 조건 전부에서 같은 값을 내는지 대조한다.

왜 필요한가:
  두 구현은 content 구간을 잡는 방식이 다르다.

    내 쪽    raw_prompt 를 단독으로 토큰화
    어댑터   chat template 에 넣어 한 번 토큰화하고 offset 으로 content 를 잘라냄

  template 의 'Prompt: ' 뒤 공백이 content 첫 글자와 한 토큰으로 합쳐지므로,
  원리적으로 두 방식은 key 가 문자열 맨 앞(front)일 때 어긋날 수 있다.
  모델이 실제로 받는 것은 어댑터 쪽이므로, 어긋나면 어댑터가 기준이다.

  샘플 3건으로는 일치했지만 3건은 결론이 못 된다. 전수로 확인한다.

  일치하면  내 논문 수치를 그대로 쓸 수 있고 X.8 의 pending 항목이 해소된다.
  다르면    어긋난 컬럼과 행 수를 정확히 세어, 어댑터 값으로 옮긴다.

주의:
  이 스크립트는 모델을 로드하지만 추론은 하지 않는다. 어댑터의 실제 토큰화
  경로를 그대로 타기 위해서다. 내가 backend 를 새로 짜면 검증 대상이
  어댑터가 아니라 내 backend 가 되어버린다.

사용법:
    .venv\\Scripts\\python.exe analysis/tokenizer/verify_adapter_agreement.py
    .venv\\Scripts\\python.exe analysis/tokenizer/verify_adapter_agreement.py --limit 20
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.common import schema  # noqa: E402

PROMPTS = REPO / "benchmarks" / "prompts" / "prompts.csv"
MINE = REPO / "analysis" / "truncation" / "tokenization_results.csv"
REPORT = REPO / "analysis" / "tokenizer" / "adapter_agreement_report.md"

SEP = "=" * 88

# 두 구현이 모두 내놓는 필드만 비교한다.
INT_FIELDS = [
    "total_tokens_pretrunc", "key_token_count_original",
    "key_start_pretrunc", "key_end_pretrunc",
    "total_tokens_used", "key_tokens_retained",
    "key_chars_retained", "key_chars_covered", "key_chars_uncovered",
]
FLOAT_FIELDS = [
    "key_retention_ratio", "key_retention_ratio_char",
    "key_start_ratio", "key_center_ratio", "key_end_ratio",
    "key_tokens_per_character",
]
STR_FIELDS = ["key_visibility"]
BOOL_FIELDS = ["prompt_truncated", "key_split_mid_character"]

POLICIES = [schema.POLICY_NATIVE, schema.POLICY_CONSTRAINED_77]


def as_bool(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개 프롬프트만 (빠른 확인용)")
    args = ap.parse_args()

    if not MINE.exists():
        print(f"{MINE.name} 이 없습니다. analyze_tokens.py --full 을 먼저 실행하세요.")
        return 1

    prompts = list(csv.DictReader(open(PROMPTS, encoding="utf-8-sig")))
    if args.limit:
        prompts = prompts[:args.limit]
    mine = {(r["prompt_id"], r["input_policy"]): r
            for r in csv.DictReader(open(MINE, encoding="utf-8-sig"))
            if r["model_role"] == schema.ROLE_TEXT_SAFETY}

    print(f"{SEP}\n어댑터 대조 — 내 구현 vs SGuardAdapter.tokenize\n{SEP}")
    print(f"  프롬프트 {len(prompts)} x 조건 {len(POLICIES)} = {len(prompts)*len(POLICIES)} 비교")
    print("  모델 로드 중 (추론은 하지 않음, CPU)...")

    from src.adapters.text_safety.sguard import load_real_sguard_adapter
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = load_real_sguard_adapter(device=device)
    print(f"  로드 완료 (device={device})\n")

    # (field, policy) -> 불일치 수 / 예시
    bad: dict[tuple[str, str], int] = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_position: dict[tuple[str, str], Counter] = defaultdict(Counter)
    n_cmp = 0
    missing = 0

    for p in prompts:
        for pol in POLICIES:
            m = mine.get((p["prompt_id"], pol))
            if m is None:
                missing += 1
                continue
            tr = adapter.tokenize(p["raw_prompt"], p["key_expression"], pol)
            n_cmp += 1

            for f in INT_FIELDS:
                if int(getattr(tr, f)) != int(m[f]):
                    bad[(f, pol)] += 1
                    by_position[(f, pol)][p["position_level"]] += 1
                    if len(examples[(f, pol)]) < 3:
                        examples[(f, pol)].append(
                            f"{p['prompt_id']}: 어댑터 {getattr(tr, f)} / 내 것 {m[f]}")
            for f in FLOAT_FIELDS:
                if abs(float(getattr(tr, f)) - float(m[f])) > 1e-6:
                    bad[(f, pol)] += 1
                    by_position[(f, pol)][p["position_level"]] += 1
                    if len(examples[(f, pol)]) < 3:
                        examples[(f, pol)].append(
                            f"{p['prompt_id']}: 어댑터 {getattr(tr, f):.6f} / 내 것 {m[f]}")
            for f in STR_FIELDS:
                if str(getattr(tr, f)) != str(m[f]):
                    bad[(f, pol)] += 1
                    by_position[(f, pol)][p["position_level"]] += 1
                    if len(examples[(f, pol)]) < 3:
                        examples[(f, pol)].append(
                            f"{p['prompt_id']}: 어댑터 {getattr(tr, f)} / 내 것 {m[f]}")
            for f in BOOL_FIELDS:
                if bool(getattr(tr, f)) != as_bool(m[f]):
                    bad[(f, pol)] += 1
                    by_position[(f, pol)][p["position_level"]] += 1
                    if len(examples[(f, pol)]) < 3:
                        examples[(f, pol)].append(
                            f"{p['prompt_id']}: 어댑터 {getattr(tr, f)} / 내 것 {m[f]}")

    all_fields = INT_FIELDS + FLOAT_FIELDS + STR_FIELDS + BOOL_FIELDS
    md = ["# 어댑터 대조 — 내 구현 vs SGuardAdapter.tokenize", "",
          f"비교 {n_cmp}건 ({len(prompts)} 프롬프트 x {len(POLICIES)} 조건), "
          f"필드 {len(all_fields)}개", ""]

    print(f"{SEP}\n결과\n{SEP}")
    if missing:
        print(f"  경고: 내 결과에 없는 조건 {missing}건")
        md.append(f"> 경고: 내 결과에 없는 조건 {missing}건\n")

    total_bad = sum(bad.values())
    if not total_bad:
        print(f"  두 구현이 완전히 일치한다. 불일치 0 / {n_cmp * len(all_fields)} 비교")
        md += ["## 결과: 완전 일치", "",
               f"{n_cmp * len(all_fields)}개 값 비교에서 불일치 0건.", "",
               "내 독립 구현과 팀 어댑터가 같은 값을 낸다. 'Prompt: ' 뒤 공백 병합 때문에",
               "front 위치에서 어긋날 수 있다고 본 우려는 실제로는 발생하지 않았다.",
               "논문 X.8 의 measurement basis 보류 항목을 해소할 수 있다.", ""]
    else:
        print(f"  불일치 {total_bad}건")
        md += ["## 결과: 불일치 있음", "",
               "| 필드 | 조건 | 불일치 | 위치 분포 | 예시 |", "|---|---|---|---|---|"]
        for f in all_fields:
            for pol in POLICIES:
                if not bad[(f, pol)]:
                    continue
                pos = dict(by_position[(f, pol)])
                print(f"    {f:28} {pol:16} {bad[(f, pol)]:>4}건  위치 {pos}")
                for e in examples[(f, pol)]:
                    print(f"        {e}")
                md.append(f"| `{f}` | `{pol}` | {bad[(f, pol)]} | {pos} | "
                          f"{examples[(f, pol)][0] if examples[(f, pol)] else ''} |")
        md += ["", "모델이 실제로 받는 입력은 어댑터 쪽이므로, 어긋난 필드는 어댑터 값이 옳다.", ""]

    # 결론에 영향이 있는 필드인지 구분해 알려준다
    decisive = ["key_retention_ratio", "key_visibility", "total_tokens_used"]
    dec_bad = sum(bad[(f, pol)] for f in decisive for pol in POLICIES)
    print(f"\n  결론에 직접 쓰는 필드({', '.join(decisive)}) 불일치: {dec_bad}건")
    md += [f"결론에 직접 쓰는 필드(`{'`, `'.join(decisive)}`) 불일치: **{dec_bad}건**", ""]

    md += check_altdiff(prompts)
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"  보고서: {REPORT.relative_to(REPO)}")
    print(SEP)
    return 0


def check_altdiff(prompts: list[dict]) -> list[str]:
    """AltDiffusion 쪽은 SGuard 와 다른 문제를 본다.

    SGuard 는 byte-level BPE 라 key 의 모든 문자가 어떤 토큰엔가 덮인다.
    AltDiffusion 은 SentencePiece 라 '▁' 마커가 선행 공백을 표현하되 offset 에는
    넣지 않으므로, key 안의 공백은 어떤 토큰도 덮지 않는다. 이 문자를 분모에
    넣으면 절단이 전혀 없어도 ratio_char < 1 이 된다.
    """
    print(f"\n{SEP}\nAltDiffusion — 글자 단위 retention 분모 확인\n{SEP}")
    from transformers import XLMRobertaTokenizer
    from src.adapters.generators.altdiffusion import AltDiffusionAdapter
    from src.common import config

    hf = XLMRobertaTokenizer.from_pretrained(
        config.ALTDIFF_MODEL_ID, subfolder="tokenizer")

    class _Tok:
        tokenizer_class = type(hf).__name__
        revision = ""

        def encode_content(self, text):
            enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
            return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

        def decode(self, ids):
            return hf.decode(ids)

    ad = AltDiffusionAdapter(_Tok())
    spaced = [p for p in prompts if " " in p["key_expression"]]
    bad_rows = []
    for p in spaced:
        tr = ad.tokenize(p["raw_prompt"], p["key_expression"])
        # 절단이 전혀 없는데(토큰 기준 전부 보존) 글자 기준이 1 미만이면 분모가 틀린 것
        if tr.key_visibility == schema.VISIBILITY_FULL and tr.key_retention_ratio_char < 1.0:
            bad_rows.append((p["prompt_id"], p["key_expression"],
                             tr.key_retention_ratio_char, tr.key_chars_covered,
                             len(p["key_expression"])))
    print(f"  공백 포함 key {len(spaced)}행 중, 절단 없는데 ratio_char<1 인 행: {len(bad_rows)}")
    for pid, key, rc, cov, tot in bad_rows[:3]:
        print(f"    {pid[:40]:40} key={key[:16]:16} ratio_char {rc:.3f} (covered {cov}/{tot})")

    md = ["", "## AltDiffusion — 글자 단위 retention 분모", ""]
    if not bad_rows:
        md += ["절단이 없는 행에서 `key_retention_ratio_char` 가 1 미만인 경우 없음.", ""]
    else:
        md += [f"공백을 포함한 key {len(spaced)}행 중 **{len(bad_rows)}행**에서, 절단이 전혀",
               "없는데도 `key_retention_ratio_char < 1` 이 나온다.", "",
               "SentencePiece 는 선행 공백을 `▁` 마커로 표현하되 offset 에 넣지 않으므로",
               "key 안의 공백은 어떤 토큰도 덮지 않는다. 이 문자를 분모에 넣으면 안 된다.",
               "", "| prompt_id | key | ratio_char | covered / 전체 |", "|---|---|---|---|"]
        for pid, key, rc, cov, tot in bad_rows[:5]:
            md.append(f"| `{pid}` | {key} | {rc:.3f} | {cov} / {tot} |")
        md += ["", "`analyze_content_tokens` 는 `key_chars_covered` 에 `key_chars_retained` 와",
               "같은 값을 넣고, 비율의 분모로 전체 글자 수를 쓴다. 덮이지 않은 문자를",
               "분모에서 빼야 한다.", ""]
    return md

    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"  보고서: {REPORT.relative_to(REPO)}")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
