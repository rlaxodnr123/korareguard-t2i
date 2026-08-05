#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_phase3_scores.py — KoRareGuard-T2I / Student 5

Phase 3. 432 프롬프트의 모든 view 에 SGuard 를 한 번씩 돌려 점수를 저장한다.
**판정은 하지 않는다.** 조건(4개) · 집계 규칙(5개) · 임계값 τ 는 전부 사후
자유 변수이므로, Phase 4 에서 이 CSV 만 읽어 CPU 로 몇 초에 다시 계산한다.

    이 스크립트     view 마다 점수 1회        GPU, 1회 배치
    Phase 4        조건 × 규칙 × τ 스윕       CPU, 무한 반복

τ 를 파이프라인 안에 두면 τ 마다 재실행이라 스윕이 불가능해진다. 그래서 나눈다.

================================================================================
왜 원본 view 도 다시 재는가 (학생4 결과가 이미 있는데)
================================================================================
학생4 의 `safety_results.csv` 에 432개 원본 점수가 이미 있다. 그래도 다시 잰다.

  - `prompts.csv` 가 UNSAFE_CRIM_24 수정으로 바뀌었고, 그의 결과는 바뀐 9행을
    부분 재실행하게 된다. 한 파일 안에 두 입력 버전의 행이 섞이는 동안 내 baseline
    이 그것에 의존하면 비교가 오염된다.
  - 내 모든 점수가 같은 실행·같은 입력 해시에서 나오면 조건 간 비교가 깨끗해진다.
  - 그의 864행과 대조하면 무료 교차검증이 된다 (같은 값이 나와야 한다).

비용은 432회 ≈ 32분으로 전체 배치의 15% 다. 그 값어치가 있다.

================================================================================
비용
================================================================================
view 수는 프롬프트마다 다르다 (일반·짧음 2개 ~ 희귀·장문 16개). 총 2,835회.
`predict()` 대신 `label_logits()` 만 부르므로 추론은 1회다 (UNSAFE_SCORE.md §6).

Colab 무료 세션은 약 4시간이고 이 배치는 약 3.5시간이라 빠듯하다. 체크포인트를
25건마다 남기고 `--resume` 을 지원하지만, **세션이 죽으면 VM 파일도 사라지므로**
드라이브에 주기적으로 복사하는 쪽이 안전하다 (`--mirror` 참조).

사용법:
    python defense/run_phase3_scores.py --smoke                 # 4 프롬프트, 속도 확인
    python defense/run_phase3_scores.py                         # 전체 432
    python defense/run_phase3_scores.py --resume                # 이어서
    python defense/run_phase3_scores.py --mirror /content/drive/MyDrive/kr_phase3
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.common import schema  # noqa: E402
from src.common.io import (  # noqa: E402
    check_primary_key, input_provenance, read_csv, write_csv,
)
from src.adapters.text_safety.sguard import load_real_sguard_adapter  # noqa: E402
from defense.defense_pipeline import (  # noqa: E402
    PIPELINE_BUDGET, PIPELINE_STRIDE, DefensePipeline,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("phase3")

PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"
OUT_DIR = REPO / "defense" / "phase3"
OUT_CSV = OUT_DIR / "view_scores.csv"
OUT_JSON = OUT_DIR / "run_metadata.json"
SMOKE_CSV = OUT_DIR / "view_scores_smoke.csv"
SMOKE_JSON = OUT_DIR / "run_metadata_smoke.json"

CHECKPOINT_EVERY = 25
MAX_CONSECUTIVE_ERRORS = 5

# smoke 는 view 구성이 가장 다른 네 조합을 고른다 (일반/희귀 × 짧음/장문).
SMOKE_PROMPT_IDS = [
    "SAFE_CULT_01_COMMON_SHORT_FRONT",
    "SAFE_CULT_01_RARE_SHORT_FRONT",
    "UNSAFE_VIOL_13_COMMON_OVER_LIMIT_BACK",
    "UNSAFE_VIOL_13_RARE_OVER_LIMIT_BACK",
]

COLUMNS = [
    "prompt_id", "concept_id", "safety_label", "rarity_label", "length_level",
    "position_level", "key_expression",
    "view_kind", "view_name", "view_tokens", "input_tokens",
    "normalization_applied", "added_chars", "n_content_tokens",
    "unsafe_score", "runtime_ms", "error_type",
]
PK_COLUMNS = ["prompt_id", "view_name"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def score_view(adapter, input_ids: list[int]) -> float | None:
    """view 하나의 unsafe_score. 판정 문자열은 뽑지 않는다 (추론 1회)."""
    probs = adapter.model.label_logits(input_ids)
    return max(probs.values()) if probs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="4 프롬프트만 (속도 확인)")
    ap.add_argument("--resume", action="store_true", help="성공한 view 는 건너뛴다")
    ap.add_argument("--overwrite", action="store_true", help="새로 시작")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (기본 자동)")
    ap.add_argument("--mirror", default=None,
                    help="체크포인트마다 이 디렉터리로 결과를 복사한다 (예: 구글 드라이브)")
    args = ap.parse_args()

    if not PROMPTS_CSV.exists():
        log.error("프롬프트 파일이 없습니다: %s", PROMPTS_CSV)
        return 1
    prompts = read_csv(str(PROMPTS_CSV))
    if args.smoke:
        prompts = [r for r in prompts if r["prompt_id"] in SMOKE_PROMPT_IDS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = SMOKE_CSV if args.smoke else OUT_CSV
    out_json = SMOKE_JSON if args.smoke else OUT_JSON

    done: dict[tuple[str, str], dict] = {}
    n_retry = 0
    if out_csv.exists():
        if args.resume:
            for r in read_csv(str(out_csv), COLUMNS):
                if str(r.get("error_type", "")).strip():
                    n_retry += 1
                    continue
                done[(r["prompt_id"], r["view_name"])] = r
            log.info("--resume: 성공 %d view 건너뜀, 오류 %d view 재시도", len(done), n_retry)
        elif args.overwrite or args.smoke:
            done = {}
        else:
            log.error("이미 존재합니다: %s  (--overwrite 또는 --resume)", out_csv)
            return 1

    log.info("SGuard 로딩 중 (최초 실행이면 가중치 약 5GB 다운로드)...")
    adapter = load_real_sguard_adapter(device=args.device)
    log.info("로드 완료: %s @ %s", adapter.model_id, adapter.revision)
    pipe = DefensePipeline(adapter)

    out_rows: list[dict[str, Any]] = list(done.values())
    n_new = n_error = n_consecutive = 0
    aborted = False
    t_start = time.time()

    def checkpoint() -> None:
        out_rows.sort(key=lambda r: (r["prompt_id"], r["view_name"]))
        check_primary_key(out_rows, PK_COLUMNS, str(out_csv))
        write_csv(str(out_csv), out_rows, COLUMNS)
        if args.mirror:
            try:
                d = Path(args.mirror)
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(out_csv, d / out_csv.name)
            except Exception as exc:  # noqa: BLE001 — 미러 실패로 배치를 죽이지 않는다
                log.warning("미러 복사 실패 (%s): %s", args.mirror, exc)

    log.info("프롬프트 %d개 — view 구성 중...", len(prompts))
    plans = []
    for r in prompts:
        try:
            plans.append((r, pipe.build_views(r["raw_prompt"], r["key_expression"],
                                              r["prompt_id"])))
        except Exception as exc:  # noqa: BLE001
            log.error("%s : view 구성 실패 %s: %s", r["prompt_id"], type(exc).__name__, exc)
            n_error += 1
    total = sum(p.n_calls for _, p in plans)
    log.info("총 view %d개 (남은 것 %d) · budget=%d stride=%d",
             total, total - len(done), PIPELINE_BUDGET, PIPELINE_STRIDE)

    for r, plan in plans:
        pid = r["prompt_id"]
        for v in plan.views:
            if (pid, v.name) in done:
                continue
            row = {c: "" for c in COLUMNS}
            row.update({
                "prompt_id": pid, "concept_id": r["concept_id"],
                "safety_label": r["safety_label"], "rarity_label": r["rarity_label"],
                "length_level": r["length_level"], "position_level": r["position_level"],
                "key_expression": r["key_expression"],
                "view_kind": v.kind, "view_name": v.name, "view_tokens": v.n_tokens,
                "input_tokens": len(plan.input_ids[v.name]),
                "normalization_applied": plan.normalization_applied,
                "added_chars": plan.added_chars,
                "n_content_tokens": plan.n_content_tokens,
            })
            t0 = time.perf_counter()
            try:
                sc = score_view(adapter, plan.input_ids[v.name])
                row["runtime_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                if sc is None:
                    # 0 으로 채우면 "안전하다" 쪽으로 결과가 편향된다.
                    row["error_type"] = "NoLabelLogits"
                    n_error += 1
                    n_consecutive += 1
                else:
                    row["unsafe_score"] = round(sc, 6)
                    n_consecutive = 0
            except Exception as exc:  # noqa: BLE001
                row["runtime_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                row["error_type"] = type(exc).__name__
                n_error += 1
                n_consecutive += 1
                log.error("%s / %s : %s: %s", pid, v.name, type(exc).__name__, exc)

            out_rows.append(row)
            n_new += 1
            if n_new % CHECKPOINT_EVERY == 0:
                checkpoint()
                el = time.time() - t_start
                left = max(total - len(done) - n_new, 0)
                log.info("진행 %d/%d view (%.0fs, %.2fs/view, 남은 시간 약 %.0f분, 오류 %d)",
                         len(done) + n_new, total, el, el / n_new,
                         left * (el / n_new) / 60, n_error)
            if n_consecutive >= MAX_CONSECUTIVE_ERRORS:
                log.error("연속 %d회 실패 — 중단합니다. 여기까지는 저장됩니다.", n_consecutive)
                log.error("원인을 고친 뒤 --resume 으로 재시도하세요.")
                aborted = True
                break
        if aborted:
            break

    checkpoint()
    elapsed = time.time() - t_start

    meta = {
        "phase": 3,
        "purpose": "view 별 unsafe_score 수집. 판정·집계·τ 는 Phase 4 에서 사후 계산.",
        "smoke": args.smoke,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "inputs": input_provenance([str(PROMPTS_CSV), str(REPO / "defense" / "glossary.json")]),
        "model_id": adapter.model_id,
        "model_revision": adapter.revision,
        "chunking": {"budget": PIPELINE_BUDGET, "stride": PIPELINE_STRIDE,
                     "overlap": PIPELINE_BUDGET - PIPELINE_STRIDE},
        "n_prompts": len(prompts),
        "n_views_expected": total,
        "n_views_written": len(out_rows),
        "n_new": n_new,
        "n_error": n_error,
        "complete": (not aborted) and len(out_rows) == total and n_error == 0,
        "aborted_on_consecutive_errors": aborted,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_view": round(elapsed / n_new, 3) if n_new else None,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    for m in ("torch", "transformers"):
        try:
            meta["environment"][m] = __import__(m).__version__
        except Exception:
            meta["environment"][m] = None
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.mirror:
        try:
            shutil.copy(out_json, Path(args.mirror) / out_json.name)
        except Exception:
            pass

    print(f"\n{'=' * 78}")
    print(f"  view          : {len(out_rows)} / {total}  (신규 {n_new}, 이어받음 {len(done)})")
    print(f"  오류          : {n_error}")
    print(f"  소요          : {elapsed:.1f}s" + (f"  ({elapsed/n_new:.2f}s/view)" if n_new else ""))
    print(f"  결과 CSV      : {out_csv}")
    print(f"  메타데이터     : {out_json}")
    if args.smoke and n_new:
        print(f"\n  전체 2,835 view 예상: 약 {2835 * (elapsed/n_new) / 3600:.1f}시간")
    if not meta["complete"] and not args.smoke:
        print("  ! 아직 완료 아님 — --resume 으로 이어서 실행하세요")
    print(f"{'=' * 78}")
    return 0 if meta["complete"] or args.smoke else 1


if __name__ == "__main__":
    sys.exit(main())
