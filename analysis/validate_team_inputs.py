#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_team_inputs.py — KoRareGuard-T2I / Student 2

팀원의 결과 CSV(학생 3·4)가 root_cause.py 에 들어갈 수 있는 형태인지 검사한다.

왜 필요한가:
  스키마를 말로 합의하면 반드시 어긋난다. 어긋난 걸 조인 시점에 발견하면
  이미 GPU 시간을 다 쓴 뒤라 되돌릴 수 없다. 계약을 실행 가능한 코드로 써서
  미리 건네면, 팀원이 결과를 만들면서 스스로 확인할 수 있다.

무엇을 보는가:
  1. 컬럼      schema.py 의 정의와 정확히 일치하는가 (누락/오타/추가)
  2. 기본키    중복이 없는가
  3. 참조 무결 image_labels 의 generation_id 가 generation_results 에 있는가
               safety_results 의 prompt_id 가 prompts.csv 에 있는가
  4. 값 어휘   safe/unsafe, true/false 를 한 컬럼 안에서 섞어 쓰지 않는가
  5. 규칙      schema.py 가 문서화한 "5개 중 하나라도 unsafe 면 decision=unsafe"
  6. 커버리지  432 프롬프트가 조건마다 다 있는가

사용법:
    # 팀원에게 건넬 빈 템플릿 만들기
    .venv\\Scripts\\python.exe analysis/validate_team_inputs.py --template outputs/

    # 도착한 결과 검사하기 (있는 파일만 검사한다)
    .venv\\Scripts\\python.exe analysis/validate_team_inputs.py \\
        --safety outputs/safety_results.csv \\
        --generation outputs/generation_results.csv \\
        --labels outputs/image_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.common import schema  # noqa: E402

PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"
SEP = "=" * 88

TRUE_WORDS = {"true", "1", "yes", "y", "t"}
FALSE_WORDS = {"false", "0", "no", "n", "f"}


class Report:
    def __init__(self) -> None:
        self.n_fail = 0
        self.n_warn = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if not ok:
            self.n_fail += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   — {detail}" if detail else ""))
        return ok

    def warn(self, name: str, clean: bool, detail: str = "") -> None:
        if not clean:
            self.n_warn += 1
        print(f"  [{'PASS' if clean else 'WARN'}] {name}" + (f"   — {detail}" if detail else ""))


def read_csv(p: Path) -> list[dict[str, str]]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def preview(items: Any, n: int = 4) -> str:
    s = sorted(items)
    return ", ".join(map(str, s[:n])) + (f" 외 {len(s) - n}개" if len(s) > n else "")


# ---------------------------------------------------------------------------
# 공통 검사
# ---------------------------------------------------------------------------

def check_columns(rep: Report, rows: list[dict], expected: list[str], label: str) -> bool:
    got = list(rows[0].keys()) if rows else []
    missing = [c for c in expected if c not in got]
    extra = [c for c in got if c not in expected]
    ok = rep.check(f"{label} 컬럼이 schema 정의와 일치", not missing,
                   f"누락: {preview(missing)}" if missing else f"{len(got)}개")
    # 추가 컬럼은 조인을 깨지 않으므로 경고로만 둔다.
    rep.warn(f"{label} 정의에 없는 컬럼 없음", not extra,
             f"추가됨: {preview(extra)}" if extra else "")
    return ok


def check_pk(rep: Report, rows: list[dict], keys: list[str], label: str) -> None:
    seen = Counter(tuple(r.get(k, "") for k in keys) for r in rows)
    dup = [k for k, c in seen.items() if c > 1]
    rep.check(f"{label} 기본키 {tuple(keys)} 중복 없음", not dup,
              f"중복 {len(dup)}건: {preview([' / '.join(d) for d in dup], 2)}" if dup
              else f"{len(rows)}행")


def check_vocab(rep: Report, rows: list[dict], col: str,
                allowed: set[str], label: str) -> None:
    if not rows or col not in rows[0]:
        return
    vals = Counter(str(r[col]).strip().lower() for r in rows)
    bad = {v: c for v, c in vals.items() if v not in allowed}
    rep.check(f"{label}.{col} 값이 {sorted(allowed)} 안에 있음", not bad,
              f"허용 밖: {preview(bad)}" if bad else f"{dict(vals)}")


def check_boolean(rep: Report, rows: list[dict], col: str, label: str) -> None:
    """true/false 계열 표기를 한 컬럼 안에서 섞어 쓰지 않는지 본다.

    'true' 와 '1' 이 섞이면 파싱은 되지만, 나중에 누가 == 'true' 로 비교하는 순간
    조용히 절반이 빠진다. 표기를 하나로 고정하게 한다.
    """
    if not rows or col not in rows[0]:
        return
    vals = Counter(str(r[col]).strip().lower() for r in rows if str(r[col]).strip())
    bad = {v: c for v, c in vals.items() if v not in TRUE_WORDS | FALSE_WORDS}
    rep.check(f"{label}.{col} 이 참/거짓 값", not bad,
              f"해석 불가: {preview(bad)}" if bad else "")
    forms = set(vals) & (TRUE_WORDS | FALSE_WORDS)
    mixed = len({v for v in forms if v in TRUE_WORDS}) > 1 or \
            len({v for v in forms if v in FALSE_WORDS}) > 1
    rep.warn(f"{label}.{col} 표기가 한 가지로 통일됨", not mixed,
             f"섞여 있음: {preview(forms)}" if mixed else f"{preview(forms)}")


# ---------------------------------------------------------------------------
# 파일별 검사
# ---------------------------------------------------------------------------

def validate_safety(rep: Report, rows: list[dict], prompt_ids: set[str]) -> None:
    print(f"\n{SEP}\nsafety_results.csv (학생 4)\n{SEP}")
    if not check_columns(rep, rows, schema.SAFETY_COLUMNS, "safety_results"):
        return
    check_pk(rep, rows, [schema.SafetyCols.PROMPT_ID, schema.SafetyCols.INPUT_POLICY],
             "safety_results")

    unknown = {r[schema.SafetyCols.PROMPT_ID] for r in rows} - prompt_ids
    rep.check("prompt_id 가 모두 prompts.csv 에 있음", not unknown,
              f"모르는 id {len(unknown)}개: {preview(unknown)}" if unknown else "")

    check_vocab(rep, rows, schema.SafetyCols.INPUT_POLICY, set(schema.ALL_POLICIES),
                "safety_results")
    check_vocab(rep, rows, schema.SafetyCols.DECISION, {schema.SAFE, schema.UNSAFE},
                "safety_results")
    for c in schema.SG_CATEGORY_COLUMNS:
        check_vocab(rep, rows, c, {schema.SAFE, schema.UNSAFE}, "safety_results")

    # schema.py 가 문서화한 규칙: 5개 중 하나라도 unsafe 면 decision 은 unsafe
    viol = [r[schema.SafetyCols.PROMPT_ID] for r in rows
            if (any(str(r.get(c, "")).strip().lower() == schema.UNSAFE
                    for c in schema.SG_CATEGORY_COLUMNS)
                != (str(r[schema.SafetyCols.DECISION]).strip().lower() == schema.UNSAFE))]
    rep.check("decision 이 5개 카테고리 OR 과 일치 (schema.py 정의)", not viol,
              f"불일치 {len(viol)}건: {preview(viol)}" if viol else "")

    per = Counter(r[schema.SafetyCols.INPUT_POLICY] for r in rows)
    rep.warn("조건마다 432 프롬프트가 모두 있음",
             all(v == len(prompt_ids) for v in per.values()),
             f"{dict(per)} (기대 {len(prompt_ids)})")

    err = [r for r in rows if str(r.get(schema.SafetyCols.ERROR_TYPE, "")).strip()]
    rep.warn("실행 오류 행 없음", not err, f"{len(err)}행에 error_type 있음" if err else "")


def validate_generation(rep: Report, rows: list[dict], prompt_ids: set[str]) -> set[str]:
    print(f"\n{SEP}\ngeneration_results.csv (학생 3)\n{SEP}")
    if not check_columns(rep, rows, schema.GENERATION_COLUMNS, "generation_results"):
        return set()
    check_pk(rep, rows, [schema.GenCols.GENERATION_ID], "generation_results")

    unknown = {r[schema.GenCols.PROMPT_ID] for r in rows} - prompt_ids
    rep.check("prompt_id 가 모두 prompts.csv 에 있음", not unknown,
              f"모르는 id {len(unknown)}개: {preview(unknown)}" if unknown else "")

    ok_rows = [r for r in rows if not str(r.get(schema.GenCols.ERROR_TYPE, "")).strip()]
    noimg = [r[schema.GenCols.GENERATION_ID] for r in ok_rows
             if not str(r.get(schema.GenCols.IMAGE_PATH, "")).strip()]
    rep.check("오류가 없는 행에는 image_path 가 있음", not noimg,
              f"{len(noimg)}건 비어 있음: {preview(noimg)}" if noimg else "")

    seeds = Counter(r[schema.GenCols.PROMPT_ID] for r in rows)
    n = set(seeds.values())
    rep.warn("프롬프트당 생성 횟수가 일정함", len(n) <= 1,
             f"횟수가 여러 가지: {preview(n)}" if len(n) > 1 else f"프롬프트당 {preview(n)}회")
    return {r[schema.GenCols.GENERATION_ID] for r in rows}


def validate_labels(rep: Report, rows: list[dict], gen_ids: Optional[set[str]]) -> None:
    print(f"\n{SEP}\nimage_labels.csv (학생 3)\n{SEP}")
    if not check_columns(rep, rows, schema.IMAGE_LABEL_COLUMNS, "image_labels"):
        return
    check_pk(rep, rows, [schema.ImgCols.GENERATION_ID], "image_labels")

    if gen_ids is None:
        rep.warn("generation_id 참조 무결성", False,
                 "generation_results.csv 가 없어 확인 못 함")
    else:
        unknown = {r[schema.ImgCols.GENERATION_ID] for r in rows} - gen_ids
        rep.check("generation_id 가 모두 generation_results 에 있음", not unknown,
                  f"모르는 id {len(unknown)}개: {preview(unknown)}" if unknown else "")
        missing = gen_ids - {r[schema.ImgCols.GENERATION_ID] for r in rows}
        rep.warn("모든 생성물에 라벨이 있음", not missing,
                 f"라벨 없는 생성물 {len(missing)}건: {preview(missing)}" if missing else "")

    check_boolean(rep, rows, schema.ImgCols.CONCEPT_PRESENT, "image_labels")
    check_vocab(rep, rows, schema.ImgCols.IMAGE_SAFETY_LABEL,
                {schema.SAFE, schema.UNSAFE}, "image_labels")

    # 미해결 이슈: 평가 대상은 둘인데 평가자 칸은 한 벌뿐이다.
    # 무엇을 평가한 값인지 확정되기 전에는 annotator_* / final_label 을 쓰지 않는다.
    filled = sum(1 for r in rows
                 if str(r.get(schema.ImgCols.ANNOTATOR_1, "")).strip()
                 or str(r.get(schema.ImgCols.FINAL_LABEL, "")).strip())
    if filled:
        rep.warn("annotator_* / final_label 의 평가 대상이 확정됨", False,
                 f"{filled}행이 채워져 있으나, 이 값이 concept_present 를 평가한 "
                 "것인지 image_safety_label 을 평가한 것인지 schema.py 에 정의가 없다")


def write_templates(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, cols in (("safety_results.csv", schema.SAFETY_COLUMNS),
                       ("generation_results.csv", schema.GENERATION_COLUMNS),
                       ("image_labels.csv", schema.IMAGE_LABEL_COLUMNS)):
        p = outdir / name
        if p.exists():
            print(f"  건너뜀 (이미 있음): {p}")
            continue
        with open(p, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(cols)
        print(f"  생성: {p}  ({len(cols)}개 컬럼)")
    print("\n  이 헤더 그대로 채우면 root_cause.py 에 바로 들어갑니다.")
    print("  검사: analysis/validate_team_inputs.py --safety ... --generation ... --labels ...")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--safety")
    ap.add_argument("--generation")
    ap.add_argument("--labels")
    ap.add_argument("--template", metavar="DIR",
                    help="빈 템플릿 CSV 를 만들어 팀원에게 건넨다")
    args = ap.parse_args()

    if args.template:
        d = Path(args.template)
        write_templates(d if d.is_absolute() else REPO / d)
        return 0

    if not any((args.safety, args.generation, args.labels)):
        ap.error("검사할 파일을 하나 이상 지정하세요 (--safety / --generation / --labels). "
                 "빈 템플릿이 필요하면 --template DIR")

    prompt_ids = {r["prompt_id"] for r in read_csv(PROMPTS_CSV)}
    rep = Report()
    print(f"{SEP}\n팀 입력 검사   (기준 prompts.csv: {len(prompt_ids)} 프롬프트)\n{SEP}")

    gen_ids: Optional[set[str]] = None
    if args.generation:
        gen_ids = validate_generation(rep, read_csv(Path(args.generation)), prompt_ids)
    if args.safety:
        validate_safety(rep, read_csv(Path(args.safety)), prompt_ids)
    if args.labels:
        validate_labels(rep, read_csv(Path(args.labels)), gen_ids)

    print(f"\n{SEP}\n  FAIL {rep.n_fail}건 / WARN {rep.n_warn}건")
    print("  FAIL 은 조인이 깨지므로 반드시 고쳐야 하고, WARN 은 확인만 하면 됩니다.")
    print(SEP)
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
