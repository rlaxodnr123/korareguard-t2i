#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_results.py — KoRareGuard-T2I / Student 2

analyze_tokens.py 가 기록한 CSV 를 다시 읽어서 검증한다.

메모리 상태가 아니라 실제로 파일에 쓰인 내용을 검사하는 것이 핵심이다.
직렬화 과정에서 값이 깨지거나 컬럼이 밀리는 사고는 in-memory 검증으로는 잡히지 않는다.

검사 항목:
  A. 데이터셋 무결성   중복 키 / 결측 / 라벨 값 / 요인 균형
  B. 토큰화 정합성     span 범위 / retention 범위 / visibility 일관성 / cap 초과
  C. 조건 설계         조건별 행 수 / policy 별 cap / 조건 1 절단 여부
  D. 교차 검증         같은 프롬프트의 조건 간 불변량 (key 표현, 문자 수 등)

오류 행을 삭제하지 않는다. 발견한 문제를 보고서에 남긴다.

사용법:
    .venv\\Scripts\\python.exe analysis/truncation/validate_results.py
    .venv\\Scripts\\python.exe analysis/truncation/validate_results.py --full
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.common import schema  # noqa: E402  — 팀 공용 값 어휘와 대조하기 위해

OUT_DIR = REPO / "analysis" / "truncation"

VALID_SAFETY = {"safe", "unsafe"}
VALID_RARITY = {"common", "rare"}
VALID_LENGTH = {"short", "near_limit", "over_limit"}
VALID_POSITION = {"front", "middle", "back"}
VALID_VISIBILITY = {"full", "partial", "none"}
VALID_STATUS = {"ok", "error"}

SEP = "=" * 88


class Report:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"   — {detail}" if detail else ""))

    @property
    def n_fail(self) -> int:
        return sum(1 for _, ok, _ in self.checks if not ok)


def f(v: str) -> float:
    return float(v) if v not in ("", None) else float("nan")


def i(v: str) -> int:
    return int(float(v)) if v not in ("", None) else -1


def b(v: str) -> bool:
    return str(v).strip().lower() == "true"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--variant", default="",
                    help="벤치마크 변형 접미사. analyze_tokens.py --prompts 로 만든 "
                         "tokenization_results<variant>.csv 를 검증한다. 예: _77")
    args = ap.parse_args()
    suffix = args.variant + ("" if args.full else "_pilot")

    csv_path = OUT_DIR / f"tokenization_results{suffix}.csv"
    meta_path = OUT_DIR / f"run_metadata{suffix}.json"
    report_path = OUT_DIR / f"validation_report{suffix}.md"

    if not csv_path.exists():
        print(f"결과 파일이 없습니다: {csv_path}")
        return 1

    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    print(SEP)
    print(f"검증 대상: {csv_path.relative_to(REPO)}   ({len(rows)} 행)")
    print(SEP)

    R = Report()
    md: list[str] = [f"# Tokenization 결과 검증 보고서 ({'full' if args.full else 'pilot'})", ""]
    md.append(f"- 대상: `{csv_path.relative_to(REPO)}` ({len(rows)} 행)")
    md.append(f"- git commit: `{meta.get('git_commit', 'unknown')}`")
    md.append(f"- 생성 시각: {meta.get('generated_at_utc', 'unknown')}")
    md.append("")

    # ---------- A. 데이터셋 무결성 ----------
    print("\n[A] 데이터셋 무결성")
    keys = [(r["prompt_id"], r["model_id"], r["input_policy"]) for r in rows]
    dup = [k for k, c in Counter(keys).items() if c > 1]
    R.check("primary key (prompt_id x model_id x input_policy) 중복 없음",
            not dup, f"중복 {len(dup)}건" if dup else "")

    empties = {c: sum(1 for r in rows if not r[c].strip())
               for c in ("prompt_id", "concept_id", "model_id", "input_policy",
                         "key_expression", "key_visibility", "analysis_status")}
    bad_empty = {k: v for k, v in empties.items() if v}
    R.check("필수 컬럼 결측 없음", not bad_empty, str(bad_empty) if bad_empty else "")

    for col, valid in (("safety_label", VALID_SAFETY), ("rarity_label", VALID_RARITY),
                       ("length_level", VALID_LENGTH), ("position_level", VALID_POSITION),
                       ("key_visibility", VALID_VISIBILITY),
                       ("analysis_status", VALID_STATUS)):
        seen = {r[col] for r in rows}
        R.check(f"{col} 값 어휘 유효", seen <= valid,
                f"허용 외 값 {seen - valid}" if seen - valid else str(sorted(seen)))

    n_err = sum(1 for r in rows if r["analysis_status"] != "ok")
    R.check("analysis_status = error 행 없음", n_err == 0, f"{n_err}건")

    # 팀 공용 SSOT 와 값 어휘가 일치하는지. 여기가 어긋나면 통합 join 이
    # 에러 없이 0건이 된다 (model_role 을 'safety' 로 쓰다 겪은 문제).
    seen_role = {r["model_role"] for r in rows}
    valid_role = {schema.ROLE_TEXT_SAFETY, schema.ROLE_GENERATOR}
    R.check("model_role 이 schema 값 어휘와 일치", seen_role <= valid_role,
            f"{sorted(seen_role)} vs schema {sorted(valid_role)}")
    seen_pol = {r["input_policy"] for r in rows}
    R.check("input_policy 가 schema.ALL_POLICIES 안에 있음",
            seen_pol <= set(schema.ALL_POLICIES),
            f"{sorted(seen_pol)} vs schema {list(schema.ALL_POLICIES)}")
    R.check("key_visibility 가 schema.ALL_VISIBILITY 안에 있음",
            {r["key_visibility"] for r in rows} <= set(schema.ALL_VISIBILITY))

    # 요인 균형 — 프롬프트 단위로 본다 (조건마다 중복되므로)
    prompts = {r["prompt_id"]: r for r in rows}
    cell = Counter((r["concept_id"], r["rarity_label"], r["length_level"], r["position_level"])
                   for r in prompts.values())
    R.check("concept x rarity x length x position 각 셀 정확히 1개",
            all(v == 1 for v in cell.values()), f"{len(cell)} 셀")
    for col in ("rarity_label", "length_level", "position_level"):
        c = Counter(r[col] for r in prompts.values())
        R.check(f"{col} 균형", len(set(c.values())) == 1, str(dict(c)))

    # ---------- B. 토큰화 정합성 ----------
    print("\n[B] 토큰화 정합성")
    problems: list[str] = []
    for r in rows:
        pid = f"{r['prompt_id']}/{r['input_policy']}"
        pre, used = i(r["total_tokens_pretrunc"]), i(r["total_tokens_used"])
        ks, ke = i(r["key_start_pretrunc"]), i(r["key_end_pretrunc"])
        rt, rc = f(r["key_retention_ratio"]), f(r["key_retention_ratio_char"])
        # cap 은 content 토큰 예산이다. declared_max_length(77/131072)가 아니라
        # content_token_budget(75 / 77 / 무제한)으로 검사해야 실제 위반을 잡는다.
        cap = i(r["content_token_budget"])

        if not (0 <= ks <= ke < pre):
            problems.append(f"{pid}: span [{ks}:{ke}] 이 시퀀스 범위 밖 (pre={pre})")
        if not (0.0 <= rt <= 1.0):
            problems.append(f"{pid}: key_retention_ratio={rt}")
        if not (0.0 <= rc <= 1.0):
            problems.append(f"{pid}: key_retention_ratio_char={rc}")
        expect = "full" if rt >= 1.0 else "partial" if rt > 0 else "none"
        if r["key_visibility"] != expect:
            problems.append(f"{pid}: visibility={r['key_visibility']} 인데 ratio={rt}")
        if used > pre:
            problems.append(f"{pid}: used {used} > pretrunc {pre}")
        if cap > 0 and used > cap:
            problems.append(f"{pid}: used {used} > cap {cap}")
        if b(r["prompt_truncated"]) != (used < pre):
            problems.append(f"{pid}: prompt_truncated 플래그와 used/pretrunc 불일치")
        if i(r["key_tokens_retained"]) > i(r["key_token_count_original"]):
            problems.append(f"{pid}: retained > original")
        if rt >= 1.0 and not b(r["key_split_mid_character"]) and rc < 1.0:
            problems.append(f"{pid}: 토큰 전부 남았는데 ratio_char={rc} < 1")
        for c in ("key_start_ratio", "key_center_ratio", "key_end_ratio"):
            if not (0.0 <= f(r[c]) <= 1.0):
                problems.append(f"{pid}: {c}={r[c]}")

    R.check("토큰화 정합성 위반 없음", not problems, f"{len(problems)}건")
    for p in problems[:15]:
        print(f"        - {p}")

    # ---------- C. 조건 설계 ----------
    print("\n[C] 조건 설계")
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[(r["model_id"], r["input_policy"])].append(r)
    R.check("조건 수 = 3", len(by_cond) == 3, str([f"{m.split('/')[-1]}|{p}" for m, p in by_cond]))
    counts = {k: len(v) for k, v in by_cond.items()}
    R.check("조건별 행 수 동일", len(set(counts.values())) == 1, str(list(counts.values())))

    for (mid, pol), sub in by_cond.items():
        tag = f"{mid.split('/')[-1]}|{pol}"
        caps = {r["experimental_token_cap"] for r in sub}
        if pol.startswith("constrained"):
            R.check(f"{tag}: experimental_token_cap 기록됨", caps == {"77"}, str(caps))
            n_none = sum(1 for r in sub if r["key_visibility"] == "none")
            R.check(f"{tag}: 절단이 실제로 발생", n_none > 0, f"visibility=none {n_none}행")
        else:
            R.check(f"{tag}: experimental_token_cap 비어있음 (native)",
                    caps <= {"", "None"}, str(caps))
        if "AltDiffusion" in mid:
            # diffusers 는 special token 자리를 먼저 확보한 뒤 content 를 자른다.
            # content_token_budget = declared_max_length - special_tokens_reserved 여야 한다.
            bad = [r for r in sub
                   if i(r["content_token_budget"])
                   != i(r["declared_max_length"]) - i(r["special_tokens_reserved"])]
            s = sub[0]
            R.check(f"{tag}: content budget = declared - special tokens",
                    not bad,
                    f"declared={s['declared_max_length']} "
                    f"special={s['special_tokens_reserved']} "
                    f"budget={s['content_token_budget']}")

    sg_native = by_cond.get((next(m for m, _ in by_cond if "SGuard" in m), "native"), [])
    if sg_native:
        n_tr = sum(1 for r in sg_native if b(r["prompt_truncated"]))
        R.check("조건 1 (SGuard native) 은 절단 0건", n_tr == 0, f"{n_tr}건 절단")
        fmt_max = max(i(r["formatted_pretrunc_tokens"]) for r in sg_native)
        R.check("조건 1 실제 입력이 native context 미만",
                fmt_max < meta.get("sguard_native_context", 131072),
                f"최대 {fmt_max} < {meta.get('sguard_native_context', 131072)}")

    # ---------- D. 조건 간 불변량 ----------
    print("\n[D] 조건 간 불변량 (같은 프롬프트는 조건이 달라도 같아야 하는 값)")
    inv_bad = []
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["prompt_id"]].append(r)
    for pid, g in grouped.items():
        for col in ("key_expression", "character_count", "key_character_count",
                    "safety_label", "rarity_label", "length_level", "position_level"):
            if len({r[col] for r in g}) > 1:
                inv_bad.append(f"{pid}: {col} 가 조건마다 다름")
    R.check("조건 간 불변량 유지", not inv_bad, f"{len(inv_bad)}건")

    # 같은 tokenizer 의 두 policy 는 PASS 1 이 동일해야 한다
    sg_rows = defaultdict(dict)
    for r in rows:
        if "SGuard" in r["model_id"]:
            sg_rows[r["prompt_id"]][r["input_policy"]] = r
    pass1_bad = [pid for pid, d in sg_rows.items()
                 if len(d) == 2 and len({v["total_tokens_pretrunc"] for v in d.values()}) > 1]
    R.check("SGuard 두 policy 의 pre-truncation 토큰 수 동일 (같은 tokenizer)",
            not pass1_bad, f"{len(pass1_bad)}건")

    # ---------- 요약 통계 ----------
    print("\n[요약]")
    md.append("## 검증 결과\n")
    md.append("| 항목 | 결과 | 비고 |")
    md.append("|---|---|---|")
    for name, ok, detail in R.checks:
        md.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
    md.append("")

    md.append("## visibility 분포\n")
    md.append("| 조건 | full | partial | none |")
    md.append("|---|---|---|---|")
    for (mid, pol), sub in by_cond.items():
        c = Counter(r["key_visibility"] for r in sub)
        line = f"{mid.split('/')[-1]} | {pol}"
        print(f"  {line:52} full={c['full']:>4} partial={c['partial']:>3} none={c['none']:>4}")
        md.append(f"| {mid.split('/')[-1]} ({pol}) | {c['full']} | {c['partial']} | {c['none']} |")

    n_tr_full = sum(1 for r in rows if b(r["prompt_truncated"]) and r["key_visibility"] == "full")
    n_split = sum(1 for r in rows if b(r["key_split_mid_character"]))
    print(f"\n  prompt 잘렸지만 key full : {n_tr_full}행")
    print(f"  절단이 key 글자 중간 통과 : {n_split}행")
    md.append(f"\n- prompt 잘렸지만 key full: {n_tr_full}행")
    md.append(f"- 절단이 key 글자 중간 통과: {n_split}행")

    print(f"\n{SEP}")
    print(f"  검사 {len(R.checks)}건 중 FAIL {R.n_fail}건")
    print(SEP)
    md.append(f"\n**검사 {len(R.checks)}건 중 FAIL {R.n_fail}건**")
    if problems:
        md.append("\n### 정합성 위반 상세\n")
        md += [f"- {p}" for p in problems[:50]]

    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  보고서: {report_path.relative_to(REPO)}")
    return 1 if R.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
