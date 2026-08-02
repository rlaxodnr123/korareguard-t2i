#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_safety_checker.py — KoRareGuard-T2I / Student 4

SGuardAdapter.predict() 를 432 프롬프트 x 2 조건(native, constrained_77)에 돌려
safety_results.csv (864행) 를 만든다.

절대 규칙 (src/README.md):
  - 토큰화/템플릿 조립을 재구현하지 않는다. src.adapters.text_safety.sguard 의
    SGuardAdapter(load_real_sguard_adapter) 만 쓴다. 직접 재구현했다가 공식 어댑터와
    112건 불일치가 난 전례가 있다 (analysis/tokenizer/adapter_agreement_report.md).
  - 컬럼명 · 정책 값 · 카테고리 값은 schema.py / config.py 상수만 쓴다.
  - 추론 실패 행에 safe/unsafe 를 임의로 채우지 않는다. decision 을 비워 두면
    root_cause.py 의 classify_outcome() 이 undecided 로 걸러낸다 — 실패가 많을수록
    "필터가 놓쳤다"는 결론이 강해지는 편향을 막기 위함.

선행 검증: analysis/tokenizer/verify_sguard_behavior.py (PILOT GATE 0a) 가 이미
실모델로 라벨 전용 토큰 생성 / response="" 정상 동작 / safe·unsafe 구분을 확인했다
(analysis/tokenizer/sguard_behavior_gate.json, 전부 PASS).

주의 — 게이트가 통과했다고 predict() 경로까지 검증된 것은 아니다. 게이트는 판정을
토큰 id 로 직접 읽었고(categories_from_tokens), predict() 는 디코드된 문자열을 읽는다.
같은 게이트 기록의 gen_text 는 8행 전부 "" 였다. 그래서 아래 preflight() 로 대량 실행
전에 문자열 경로가 실제로 파싱되는지 1행만 돌려 확인한다. 이 확인 없이 --full 을
돌리면 추론 1회당 약 2분 x 864행을 태우고 decision 이 전부 빈 오류 행만 남는다.

사용법:
    python evaluation/safety/run_safety_checker.py                  # pilot (12행)
    python evaluation/safety/run_safety_checker.py --full            # 전체 864행
    python evaluation/safety/run_safety_checker.py --full --resume   # 중단 후 이어서
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.common import schema  # noqa: E402
from src.common.io import check_primary_key, read_csv, write_csv  # noqa: E402
from src.adapters.text_safety.sguard import (  # noqa: E402
    load_real_sguard_adapter, parse_sguard_output,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("run_safety_checker")

PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"
OUT_DIR = REPO / "evaluation" / "safety"
# pilot 과 full 은 파일을 나눈다. 같은 파일을 쓰면 pilot 12행 때문에 --full 이
# --overwrite/--resume 없이는 막히고, run_metadata.json 도 덮어써져 pilot 기록이 사라진다.
OUT_CSV = OUT_DIR / "safety_results.csv"
OUT_META = OUT_DIR / "run_metadata.json"
PILOT_OUT_CSV = OUT_DIR / "safety_results_pilot.csv"
PILOT_OUT_META = OUT_DIR / "run_metadata_pilot.json"

# 연속 실패가 이만큼 쌓이면 중단한다. 모델/파싱이 깨진 상태로 864행을 끝까지
# 도는 것(추론 1회 약 2분)을 막는 안전장치.
MAX_CONSECUTIVE_ERRORS = 5

INPUT_POLICIES = [schema.POLICY_NATIVE, schema.POLICY_CONSTRAINED_77]

# PILOT GATE 0a(verify_sguard_behavior.py)와 같은 표본. safe/unsafe, common/rare 를
# 모두 포함해 두 조건 모두를 최소 비용으로 사전 점검한다.
PILOT_PROMPT_IDS = [
    "SAFE_CULT_01_RARE_SHORT_FRONT",
    "SAFE_CULT_01_COMMON_SHORT_FRONT",
    "UNSAFE_VIOL_13_COMMON_SHORT_FRONT",
    "UNSAFE_VIOL_13_RARE_SHORT_FRONT",
    "UNSAFE_CRIM_24_COMMON_SHORT_FRONT",
    "UNSAFE_NSFW_18_COMMON_SHORT_FRONT",
]

CATEGORY_COLUMN = {
    "crime": schema.SafetyCols.SG_CRIME,
    "manipulation": schema.SafetyCols.SG_MANIPULATION,
    "privacy": schema.SafetyCols.SG_PRIVACY,
    "sexual": schema.SafetyCols.SG_SEXUAL,
    "violence": schema.SafetyCols.SG_VIOLENCE,
}

CHECKPOINT_EVERY = 25
PK_COLUMNS = [schema.SafetyCols.PROMPT_ID, schema.SafetyCols.INPUT_POLICY]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def empty_row(prompt_id: str, input_policy: str, model_id: str,
             error_type: str, runtime_ms: float) -> dict[str, Any]:
    row = {c: "" for c in schema.SAFETY_COLUMNS}
    row.update({
        schema.SafetyCols.PROMPT_ID: prompt_id,
        schema.SafetyCols.INPUT_POLICY: input_policy,
        schema.SafetyCols.MODEL_ID: model_id,
        schema.SafetyCols.ERROR_TYPE: error_type,
        schema.SafetyCols.RUNTIME_MS: runtime_ms,
    })
    return row


def result_row(prompt_id: str, result) -> dict[str, Any]:
    row = {c: "" for c in schema.SAFETY_COLUMNS}
    row.update({
        schema.SafetyCols.PROMPT_ID: prompt_id,
        schema.SafetyCols.INPUT_POLICY: result.input_policy,
        schema.SafetyCols.MODEL_ID: result.model_id,
        schema.SafetyCols.DECISION: result.decision,
        schema.SafetyCols.UNSAFE_SCORE:
            "" if result.unsafe_score is None else round(result.unsafe_score, 6),
        schema.SafetyCols.RAW_OUTPUT: result.raw_output,
        schema.SafetyCols.RUNTIME_MS: result.runtime_ms,
        schema.SafetyCols.ERROR_TYPE: result.error_type,
    })
    for cat, col in CATEGORY_COLUMN.items():
        row[col] = result.categories.get(cat, "")
    return row


def preflight(adapter, sample: dict[str, Any]) -> str | None:
    """대량 실행 전에 1행만 돌려 '판정을 실제로 읽을 수 있는지' 확인한다.

    PILOT GATE 0a 는 판정을 토큰 id 로 읽어서 통과했지만(categories_from_tokens),
    predict() 는 디코드된 문자열을 읽는다. 두 경로는 별개라 게이트 통과가 여기까지
    보장하지 않는다. 비용은 추론 1회(약 2분)이고, 막아주는 손실은 864행 전체다.

    반환값: 문제가 없으면 None, 있으면 중단 사유 문자열.
    """
    prep = adapter.prepare_input(sample["raw_prompt"], sample["key_expression"],
                                 schema.POLICY_NATIVE)
    raw = adapter.model.generate(prep.formatted_input_ids)

    if not raw.strip():
        return (
            "SGuard 가 낸 판정을 문자열로 읽지 못했습니다 (빈 출력).\n"
            "    원인: 판정 라벨 5종(id 49159~49168)은 special token 이라,\n"
            "          skip_special_tokens=True 로 디코드하면 전부 사라집니다.\n"
            "          같은 현상이 sguard_behavior_gate.json 에 이미 기록돼 있습니다\n"
            "          (gen_text 8행 전부 \"\"). 게이트는 토큰 id 로 읽어서 통과했습니다.\n"
            "    고칠 곳: src/adapters/text_safety/sguard.py 의 _RealModel.generate()\n"
            "          -> hf_tok.decode(..., skip_special_tokens=False)\n"
            "    이 파일은 CODEOWNERS 상 학생5(@hyeon5702) 소유입니다. 어댑터를 직접\n"
            "    재구현하지 말고(112건 불일치 전례) 소유자에게 수정을 요청하세요."
        )

    try:
        parse_sguard_output(raw)
    except Exception as exc:  # noqa: BLE001 — 사유를 그대로 보여주는 게 목적
        return (f"SGuard 출력을 파싱하지 못했습니다: {type(exc).__name__}: {exc}\n"
                f"    raw_output={raw!r}")

    if adapter.model.label_logits(prep.formatted_input_ids) is None:
        log.warning("label_logits 가 None 입니다 — unsafe_score 가 전부 빈 값이 됩니다.")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="432 전체 (기본은 pilot 6개 프롬프트)")
    ap.add_argument("--overwrite", action="store_true", help="기존 결과 파일을 새로 시작한다")
    ap.add_argument("--resume", action="store_true",
                    help="기존 결과 파일에서 '성공한' 조합만 건너뛴다 (오류 행은 다시 시도)")
    ap.add_argument("--out", default=None,
                    help="출력 CSV 경로 (기본 evaluation/safety/safety_results.csv)")
    ap.add_argument("--no-preflight", action="store_true",
                    help="사전 점검(1행 시험 실행)을 생략한다. 직전에 통과한 게 확실할 때만.")
    args = ap.parse_args()

    if not PROMPTS_CSV.exists():
        log.error("프롬프트 파일이 없습니다: %s", PROMPTS_CSV)
        return 1
    prompts = read_csv(str(PROMPTS_CSV))

    mode = "full" if args.full else "pilot"
    if args.out:
        out_csv = Path(args.out).resolve()
        out_meta = out_csv.with_name(out_csv.stem + "_metadata.json")
    else:
        out_csv = OUT_CSV if mode == "full" else PILOT_OUT_CSV
        out_meta = OUT_META if mode == "full" else PILOT_OUT_META

    if mode == "pilot":
        prompts = [p for p in prompts if p["prompt_id"] in PILOT_PROMPT_IDS]
        missing = set(PILOT_PROMPT_IDS) - {p["prompt_id"] for p in prompts}
        if missing:
            log.error("pilot 프롬프트가 prompts.csv 에 없습니다: %s", sorted(missing))
            return 1
    total = len(prompts) * len(INPUT_POLICIES)
    log.info("mode=%s  프롬프트 %d개  x  조건 %d개  =  %d행", mode, len(prompts), len(INPUT_POLICIES), total)
    if not prompts:
        log.error("대상 프롬프트가 없습니다")
        return 1

    done: dict[tuple[str, str], dict] = {}
    n_retry = 0
    if out_csv.exists():
        if args.resume:
            # 오류 행은 done 에 넣지 않는다 → 다시 시도된다.
            # 넣어 버리면 decision 이 빈 행이 영구히 남고, 오류 때문에 재실행한
            # --resume 이 정작 그 오류 행만 골라 건너뛰는 모순이 생긴다.
            for r in read_csv(str(out_csv), schema.SAFETY_COLUMNS):
                if str(r.get(schema.SafetyCols.ERROR_TYPE, "")).strip():
                    n_retry += 1
                    continue
                done[(r[schema.SafetyCols.PROMPT_ID], r[schema.SafetyCols.INPUT_POLICY])] = r
            log.info("--resume: 성공 %d행은 건너뛰고, 오류 %d행은 다시 시도합니다",
                     len(done), n_retry)
        elif not args.overwrite:
            log.error("이미 존재합니다: %s  (--overwrite 로 새로 시작하거나 --resume 으로 이어서)", out_csv)
            return 1

    log.info("SGuard 모델 로딩 중 (최초 실행이면 가중치 약 5GB 다운로드)...")
    adapter = load_real_sguard_adapter()
    log.info("모델 로드 완료: %s @ %s", adapter.model_id, adapter.revision)

    n_pending = total - len(done)
    if n_pending <= 0:
        log.info("남은 조합이 없습니다 — 사전 점검을 건너뜁니다")
    elif args.no_preflight:
        log.warning("--no-preflight: 사전 점검을 건너뜁니다")
    else:
        log.info("사전 점검: 1행만 돌려 판정을 읽을 수 있는지 확인합니다 (약 2분)...")
        reason = preflight(adapter, prompts[0])
        if reason:
            log.error("사전 점검 실패 — 실행을 중단합니다.\n    %s", reason)
            log.error("이대로 진행하면 %d행 전부 decision 이 빈 오류 행이 됩니다.", total)
            return 2
        log.info("사전 점검 통과")

    out_rows: list[dict[str, Any]] = list(done.values())
    n_error = 0
    n_new = 0
    n_consecutive_errors = 0
    aborted = False
    t_start = time.time()

    def checkpoint() -> None:
        # 재실행/이어받기 순서와 무관하게 항상 같은 정렬로 쓴다 (병합·diff 안정).
        out_rows.sort(key=lambda r: (r[schema.SafetyCols.PROMPT_ID],
                                     r[schema.SafetyCols.INPUT_POLICY]))
        check_primary_key(out_rows, PK_COLUMNS, str(out_csv))
        write_csv(str(out_csv), out_rows, schema.SAFETY_COLUMNS)

    for r in prompts:
        pid, raw, key = r["prompt_id"], r["raw_prompt"], r["key_expression"]
        for policy in INPUT_POLICIES:
            if (pid, policy) in done:
                continue
            t0 = time.perf_counter()
            try:
                result = adapter.predict(raw, key, policy)
                result.prompt_id = pid
                result.runtime_ms = round((time.perf_counter() - t0) * 1000, 3)
                row = result_row(pid, result)
            except Exception as exc:  # noqa: BLE001 — 실패도 행으로 남겨야 root_cause 가 undecided 로 처리
                runtime_ms = round((time.perf_counter() - t0) * 1000, 3)
                row = empty_row(pid, policy, adapter.model_id, type(exc).__name__, runtime_ms)
                n_error += 1
                n_consecutive_errors += 1
                log.error("%s / %s : %s: %s", pid, policy, type(exc).__name__, exc)
            else:
                n_consecutive_errors = 0
            out_rows.append(row)
            n_new += 1

            if n_new % CHECKPOINT_EVERY == 0:
                checkpoint()
                elapsed = time.time() - t_start
                rate = n_new / elapsed
                remaining = max(total - len(done) - n_new, 0) / rate if rate > 0 else float("nan")
                log.info("진행 %d/%d  (%.1fs 경과, 남은 시간 약 %.0fs, 오류 %d건)",
                          len(done) + n_new, total, elapsed, remaining, n_error)

            if n_consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.error("연속 %d행 실패 — 중단합니다. 여기까지 결과는 저장됩니다.",
                          n_consecutive_errors)
                log.error("원인을 고친 뒤 --resume 으로 다시 돌리면 오류 행부터 재시도합니다.")
                aborted = True
                break
        if aborted:
            break

    checkpoint()
    elapsed = time.time() - t_start

    meta = {
        "mode": mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_id": adapter.model_id,
        "model_revision": adapter.revision,
        "n_prompts": len(prompts),
        "n_policies": len(INPUT_POLICIES),
        "n_rows": len(out_rows),
        "n_rows_expected": total,
        "n_new_rows": n_new,
        "n_error_rows": n_error,
        "n_retried_error_rows": n_retry,
        "complete": (not aborted) and len(out_rows) == total and n_error == 0,
        "aborted_on_consecutive_errors": aborted,
        "preflight_skipped": args.no_preflight,
        "elapsed_seconds": round(elapsed, 2),
        "resumed_from_existing": len(done),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "inputs": {"prompts_csv": str(PROMPTS_CSV.relative_to(REPO)).replace("\\", "/")},
    }
    for m in ("torch", "transformers"):
        try:
            meta["environment"][m] = __import__(m).__version__
        except Exception:
            meta["environment"][m] = None
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 78}")
    print(f"  mode            : {mode}")
    print(f"  행               : {len(out_rows)} / {total}  (신규 {n_new}, 이어받음 {len(done)})")
    print(f"  오류 행          : {n_error}")
    print(f"  소요             : {elapsed:.1f}s")
    print(f"  결과 CSV         : {out_csv}")
    print(f"  실행 메타데이터   : {out_meta}")
    if not meta["complete"]:
        print("  ! 아직 완료 아님 — 원인을 고친 뒤 --resume 으로 다시 실행하세요")
        print("    (--resume 은 성공 행만 건너뛰고 오류 행은 재시도합니다)")
    print(f"{'=' * 78}")
    return 0 if meta["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
