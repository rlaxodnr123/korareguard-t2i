#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_benchmark_variants.py — KoRareGuard-T2I / Student 2

벤치마크 프롬프트 파일 세 종(prompts.csv / prompts_77.csv / prompts_127.csv)에
**동일한 분석 파이프라인**을 돌린 결과를 나란히 놓는다.

목적은 논문 수치를 만드는 것이 아니다. "왜 정식 분석에 prompts.csv 를 썼는가"를
기록으로 남기는 것이다. 파일 선택은 결과를 보고 유리한 쪽을 고르는 행위가 될 수
있으므로, 선택 근거를 결과값이 아니라 **측정 가능한 설계 결함**에 둔다.

  결함 판정 기준 (결과와 무관하게 사전에 정해진 것):
    C1  filler 통제   같은 concept x length x position 안에서 common 과 rare 의
                      차이가 key 표현 하나뿐인가. 아니면 총 토큰 델타를
                      희귀도 효과로 읽을 수 없다 (RQ-T3 가 성립하지 않는다).
    C2  key 쌍 동일   세 파일이 같은 concept 에 같은 common/rare 표현을 쓰는가.
                      다르면 RQ-T2 의 조작 자체가 달라진다.

비교 지표는 참고용으로만 싣는다. C1 을 통과하지 못한 파일의 RQ-T3 수치는
해석할 수 없으므로 표에 실어도 논문에 인용하지 않는다.

현재 상태(2026-08-02):
    이 비교의 결론에 따라 prompts_77.csv / prompts_127.csv 와 그 builder 는
    PR #12 에서 리포에서 제거됐다. 따라서 이 스크립트는 지금 그대로는 돌지 않는다.
    결과는 benchmark_variant_comparison.md 에 남아 있고, 그것이 파일 선택의
    근거 기록이다. 스크립트는 재현 절차를 남기기 위해 보존한다.

전제:
    analyze_tokens.py --full --prompts benchmarks/prompts/<파일> 이
    세 파일 모두에 대해 실행되어 있어야 한다.

사용법:
    .venv\\Scripts\\python.exe analysis/truncation/compare_benchmark_variants.py
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
PROMPT_DIR = REPO / "benchmarks" / "prompts"
OUT_DIR = REPO / "analysis" / "truncation"
REPORT = OUT_DIR / "benchmark_variant_comparison.md"

# (표시 이름, prompts 파일, 결과 CSV 접미사)
VARIANTS = [
    ("prompts.csv", "prompts.csv", ""),
    ("prompts_77.csv", "prompts_77.csv", "_77"),
    ("prompts_127.csv", "prompts_127.csv", "_127"),
]
SEP = "=" * 88


def load(name: str, suffix: str) -> tuple[list[dict], list[dict]]:
    src = PROMPT_DIR / name
    if not src.exists():
        raise SystemExit(
            f"{name} 이 리포에 없습니다.\n"
            "prompts_77.csv / prompts_127.csv 와 그 builder 는 PR #12 에서 삭제됐습니다.\n"
            "이 비교의 결론(C1 filler 통제를 통과한 파일은 prompts.csv 뿐)은\n"
            "benchmark_variant_comparison.md 에 이미 기록돼 있으므로 그것을 보면 됩니다.\n"
            "다시 돌리려면 해당 커밋에서 파일을 꺼내야 합니다:\n"
            f"    git show 1aeeb95^:benchmarks/prompts/{name} > benchmarks/prompts/{name}"
        )
    with open(src, encoding="utf-8-sig", newline="") as f:
        prompts = list(csv.DictReader(f))
    p = OUT_DIR / f"tokenization_results{suffix}.csv"
    if not p.exists():
        raise SystemExit(
            f"{p.name} 이 없습니다. 먼저 실행하세요:\n"
            f"  analyze_tokens.py --full --prompts benchmarks/prompts/{name}")
    with open(p, encoding="utf-8-sig", newline="") as f:
        return prompts, list(csv.DictReader(f))


def split_conditions(res: list[dict]) -> dict[str, dict[str, dict]]:
    """(tokenizer, input_policy) -> {prompt_id: row}"""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in res:
        tk = "sg" if "SGuard" in r["model_id"] else "ad"
        out[f"{tk}:{r['input_policy']}"][r["prompt_id"]] = r
    return out


# ---------------------------------------------------------------------------
# C1 — filler 통제
# ---------------------------------------------------------------------------

def filler_control(prompts: list[dict]) -> tuple[int, int, list[str]]:
    """common 과 rare 의 차이가 key 표현뿐인 쌍의 수.

    raw_prompt 에서 key 표현을 지운 나머지(= filler + 문형)를 비교한다.
    이것이 같아야만 총 토큰 델타를 희귀도 효과로 읽을 수 있다.
    """
    idx: dict[tuple[str, str, str, str], dict] = {}
    for p in prompts:
        idx[(p["concept_id"], p["length_level"],
             p["position_level"], p["rarity_label"])] = p

    ok = total = 0
    examples: list[str] = []
    for (c, lv, pos, rar), r in idx.items():
        if rar != "rare":
            continue
        cm = idx.get((c, lv, pos, "common"))
        if cm is None:
            continue
        total += 1
        a = r["raw_prompt"].replace(r["key_expression"], "\x00", 1)
        b = cm["raw_prompt"].replace(cm["key_expression"], "\x00", 1)
        if a == b:
            ok += 1
        elif len(examples) < 2:
            examples.append(f"{c}/{lv}/{pos}")
    return ok, total, examples


def key_pairs(prompts: list[dict]) -> dict[str, tuple[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for p in prompts:
        out[p["concept_id"]].setdefault(p["rarity_label"], p["key_expression"])
    return {c: (v.get("common", ""), v.get("rare", "")) for c, v in out.items()}


# ---------------------------------------------------------------------------
# 참고 지표
# ---------------------------------------------------------------------------

def visibility(table: dict[str, dict]) -> tuple[int, int, int]:
    c = Counter(r["key_visibility"] for r in table.values())
    return c["full"], c["partial"], c["none"]


def crosstab(sg: dict[str, dict], ad: dict[str, dict]) -> tuple[int, int, int, int]:
    """A: 둘 다 full / B: SGuard 만 / C: AltDiffusion 만(=H2a 창) / D: 둘 다 아님"""
    c = Counter()
    for pid, r in sg.items():
        s = r["key_visibility"] == "full"
        a = ad[pid]["key_visibility"] == "full"
        c["A" if s and a else "B" if s else "C" if a else "D"] += 1
    return c["A"], c["B"], c["C"], c["D"]


def rq_t3(table: dict[str, dict]) -> tuple[float, int, int]:
    """paired 총 토큰 델타 median, 잔차 0 인 쌍 수, 전체 쌍 수."""
    idx = {(r["concept_id"], r["length_level"], r["position_level"],
            r["rarity_label"]): r for r in table.values()}
    tot, resid = [], []
    for (c, lv, pos, rar), r in idx.items():
        if rar != "rare":
            continue
        cm = idx.get((c, lv, pos, "common"))
        if cm is None:
            continue
        d = int(r["total_tokens_pretrunc"]) - int(cm["total_tokens_pretrunc"])
        k = int(r["key_token_count_original"]) - int(cm["key_token_count_original"])
        tot.append(d)
        resid.append(d - k)
    return st.median(tot), sum(1 for x in resid if x == 0), len(resid)


def main() -> int:
    data: dict[str, dict[str, Any]] = {}
    for label, fname, suffix in VARIANTS:
        prompts, res = load(fname, suffix)
        cond = split_conditions(res)
        ok, tot, ex = filler_control(prompts)
        data[label] = {
            "prompts": prompts, "cond": cond,
            "filler_ok": ok, "filler_total": tot, "filler_ex": ex,
            "keys": key_pairs(prompts),
        }

    md: list[str] = [
        "# 벤치마크 변형 비교 — 정식 분석에 어떤 프롬프트 파일을 쓸 것인가", "",
        "세 파일에 **동일한 파이프라인**(`analyze_tokens.py --full`)을 돌린 결과다.",
        "", "> **이 문서의 수치는 논문에 인용하지 않는다.** 파일 선택 근거를 남기기 위한",
        "> 내부 기록이다. 결과값이 좋은 쪽을 고르는 것을 막기 위해, 선택 기준은",
        "> 아래 C1·C2 두 가지 **설계 조건**으로 사전에 고정했다.", "",
    ]
    print(SEP)
    print("벤치마크 변형 비교")
    print(SEP)

    # ---------------- C1 ----------------
    print(f"\n{SEP}\n[C1] filler 통제 — common/rare 차이가 key 표현뿐인가\n{SEP}")
    md += ["## C1 — filler 통제 (선택 기준)", "",
           "같은 concept x length x position 안에서 `raw_prompt` 에서 key 표현을 지운",
           "나머지가 common 과 rare 사이에 동일해야 한다. 동일하지 않으면 총 토큰",
           "델타에 filler 차이가 섞여 RQ-T3 를 희귀도 효과로 해석할 수 없다.", "",
           "| 파일 | 통제된 쌍 | 판정 | 예시 |", "|---|---|---|---|"]
    for label, _, _ in VARIANTS:
        d = data[label]
        ok, tot, ex = d["filler_ok"], d["filler_total"], d["filler_ex"]
        verdict = "PASS" if ok == tot else "FAIL"
        print(f"  {label:16} {ok:>3}/{tot:<3}  {verdict}"
              + (f"   예: {', '.join(ex)}" if ex else ""))
        md.append(f"| `{label}` | {ok}/{tot} | **{verdict}** | "
                  f"{', '.join(ex) if ex else '—'} |")
    md.append("")

    # ---------------- C2 ----------------
    print(f"\n{SEP}\n[C2] key 표현 쌍이 파일 간 동일한가\n{SEP}")
    md += ["## C2 — key 표현 쌍 동일성 (선택 기준)", "",
           "`prompts.csv` 를 기준으로 concept 별 (common, rare) 쌍을 비교한다.", "",
           "| 파일 | 기준과 동일한 concept | 다른 concept |", "|---|---|---|"]
    base_keys = data["prompts.csv"]["keys"]
    for label, _, _ in VARIANTS:
        ks = data[label]["keys"]
        same = [c for c in base_keys if ks.get(c) == base_keys[c]]
        diff = sorted(set(base_keys) - set(same))
        print(f"  {label:16} {len(same):>2}/{len(base_keys)} 동일"
              + (f"   다름: {', '.join(diff)}" if diff else ""))
        md.append(f"| `{label}` | {len(same)}/{len(base_keys)} | "
                  f"{', '.join(f'`{c}`' for c in diff) if diff else '—'} |")
    md.append("")

    # ---------------- 참고 1: 길이 ----------------
    print(f"\n{SEP}\n[참고 1] pre-truncation 토큰 수 (median)\n{SEP}")
    md += ["## 참고 1 — pre-truncation 토큰 수 (median)", "",
           "| 파일 | SGuard | AltDiffusion | 비율 |", "|---|---|---|---|"]
    for label, _, _ in VARIANTS:
        cond = data[label]["cond"]
        s = st.median(int(r["total_tokens_pretrunc"]) for r in cond["sg:native"].values())
        a = st.median(int(r["total_tokens_pretrunc"]) for r in cond["ad:native"].values())
        print(f"  {label:16} SGuard {s:>6.1f}   AltDiff {a:>6.1f}   비율 {s/a:.2f}")
        md.append(f"| `{label}` | {s:.1f} | {a:.1f} | {s/a:.2f} |")
    md.append("")

    # ---------------- 참고 2: 가시성 ----------------
    print(f"\n{SEP}\n[참고 2] key_visibility (full / partial / none, 432 중)\n{SEP}")
    md += ["## 참고 2 — key_visibility (full / partial / none, 432 중)", "",
           "| 파일 | SGuard native | SGuard constrained_77 | AltDiffusion (75) |",
           "|---|---|---|---|"]
    for label, _, _ in VARIANTS:
        cond = data[label]["cond"]
        cells = [visibility(cond[k]) for k in
                 ("sg:native", "sg:constrained_77", "ad:native")]
        print(f"  {label:16} " + "   ".join(f"{f}/{p}/{n}" for f, p, n in cells))
        md.append(f"| `{label}` | " + " | ".join(
            f"{f} / {p} / {n}" for f, p, n in cells) + " |")
    md.append("")

    # ---------------- 참고 3: 교차표 ----------------
    print(f"\n{SEP}\n[참고 3] SGuard@77 x AltDiffusion 교차표\n{SEP}")
    print("  A 둘 다 full   B SGuard 만   C AltDiff 만 (H2a 창)   D 둘 다 아님")
    md += ["## 참고 3 — SGuard constrained_77 × AltDiffusion 교차표", "",
           "**C** 가 H2a(안전 필터는 위험 표현을 못 보지만 생성 모델은 보는 구간)의",
           "구조적 창이다. 실제 필터 판정이 아니라 입력 가시성만으로 정의된다.", "",
           "| 파일 | A 둘 다 full | B SGuard만 | **C AltDiff만 (H2a)** | D 둘 다 아님 |",
           "|---|---|---|---|---|"]
    for label, _, _ in VARIANTS:
        cond = data[label]["cond"]
        A, B, C, D = crosstab(cond["sg:constrained_77"], cond["ad:native"])
        print(f"  {label:16} A {A:>3}   B {B:>3}   C {C:>3}   D {D:>3}")
        md.append(f"| `{label}` | {A} | {B} | **{C}** | {D} |")
    md.append("")

    # ---------------- 참고 4: RQ-T3 ----------------
    print(f"\n{SEP}\n[참고 4] RQ-T3 paired 델타 — C1 을 통과한 파일에서만 해석 가능\n{SEP}")
    md += ["## 참고 4 — RQ-T3 paired 총 토큰 델타", "",
           "잔차 = (총 토큰 델타) − (key 토큰 델타). filler 가 통제되면 0 이어야 한다.",
           "C1 FAIL 인 파일에서는 이 수치를 희귀도 효과로 해석할 수 없다.", "",
           "| 파일 | C1 | SGuard median | 잔차 0 | AltDiff median | 잔차 0 |",
           "|---|---|---|---|---|---|"]
    for label, _, _ in VARIANTS:
        cond, d = data[label]["cond"], data[label]
        c1 = "PASS" if d["filler_ok"] == d["filler_total"] else "FAIL"
        sm, sz, sn = rq_t3(cond["sg:native"])
        am, az, an = rq_t3(cond["ad:native"])
        print(f"  {label:16} C1 {c1}   SGuard {sm:+6.1f} (잔차0 {sz}/{sn})   "
              f"AltDiff {am:+6.1f} (잔차0 {az}/{an})")
        md.append(f"| `{label}` | {c1} | {sm:+.1f} | {sz}/{sn} | {am:+.1f} | {az}/{an} |")
    md.append("")

    # ---------------- 결론 ----------------
    winners = [l for l, _, _ in VARIANTS
               if data[l]["filler_ok"] == data[l]["filler_total"]]
    md += ["## 결론", "",
           f"C1 을 통과한 파일: {', '.join(f'`{w}`' for w in winners) if winners else '없음'}",
           "",
           "정식 분석(`tokenization_results.csv`, figure A–E, 논문 X장)은 "
           f"`{winners[0] if winners else '?'}` 로만 수행했다.",
           "나머지 파일의 수치는 위 표에만 남기고 인용하지 않는다.", ""]
    print(f"\n{SEP}")
    print(f"  C1 통과: {winners or '없음'}")
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"  보고서 저장: {REPORT.relative_to(REPO)}")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
