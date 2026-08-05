#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_generation.py — KoRareGuard-T2I / Student 3 (정명섭)

AltDiffusionAdapter.generate() 를 432 프롬프트에 돌려
generation_results.csv 및 image_labels.csv 템플릿을 생성한다.

절대 규칙 (AGENTS.md & src/README.md):
  1. 디커플링 (Decoupled Execution):
     - SGuard 2B LLM 모델 추론 연산은 완전히 배제한다.
     - SGuard 차단 여부와 무관하게 432개 프롬프트를 전량 생성한다 (실험 A).
  2. 어댑터 사용:
     - src.adapters.generators.altdiffusion 의 load_real_altdiffusion_adapter 만 쓴다.
  3. 스키마 상수 사용:
     - 컬럼명 및 값 상수는 src.common.schema 및 src.common.ids 만 쓴다.

사용법:
    python evaluation/generation/run_generation.py                  # pilot (6개 프롬프트)
    python evaluation/generation/run_generation.py --full           # 전체 432개 프롬프트
    python evaluation/generation/run_generation.py --full --resume  # 중단 후 이어서
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

from src.common import config, schema  # noqa: E402
from src.common.ids import make_generation_id  # noqa: E402
from src.common.io import check_primary_key, read_csv, write_csv  # noqa: E402
from src.adapters.generators.altdiffusion import load_real_altdiffusion_adapter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("run_generation")

PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"
OUT_DIR = REPO / "outputs"
IMAGES_DIR = OUT_DIR / "images"

GEN_OUT_CSV = OUT_DIR / "generation_results.csv"
LABELS_OUT_CSV = OUT_DIR / "image_labels.csv"
META_OUT_JSON = OUT_DIR / "generation_metadata.json"

PILOT_GEN_OUT_CSV = OUT_DIR / "generation_results_pilot.csv"
PILOT_LABELS_OUT_CSV = OUT_DIR / "image_labels_pilot.csv"
PILOT_META_OUT_JSON = OUT_DIR / "generation_metadata_pilot.json"

MAX_CONSECUTIVE_ERRORS = 5
CHECKPOINT_EVERY = 10

PILOT_PROMPT_IDS = [
    "SAFE_CULT_01_RARE_SHORT_FRONT",
    "SAFE_CULT_01_COMMON_SHORT_FRONT",
    "UNSAFE_VIOL_13_COMMON_SHORT_FRONT",
    "UNSAFE_VIOL_13_RARE_SHORT_FRONT",
    "UNSAFE_CRIM_24_COMMON_SHORT_FRONT",
    "UNSAFE_NSFW_18_COMMON_SHORT_FRONT",
]


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def make_empty_gen_row(gen_id: str, prompt_id: str, model_id: str, seed: int,
                       error_type: str, runtime_ms: float) -> dict[str, Any]:
    row = {c: "" for c in schema.GENERATION_COLUMNS}
    row.update({
        schema.GenCols.GENERATION_ID: gen_id,
        schema.GenCols.PROMPT_ID: prompt_id,
        schema.GenCols.GENERATOR_ID: model_id,
        schema.GenCols.SEED: seed,
        schema.GenCols.IMAGE_PATH: "",
        schema.GenCols.RUNTIME_MS: runtime_ms,
        schema.GenCols.ERROR_TYPE: error_type,
    })
    return row


def make_label_row(gen_id: str, prompt_id: str) -> dict[str, Any]:
    row = {c: "" for c in schema.IMAGE_LABEL_COLUMNS}
    row.update({
        schema.ImgCols.GENERATION_ID: gen_id,
        schema.ImgCols.CONCEPT_PRESENT: "",
        schema.ImgCols.IMAGE_SAFETY_LABEL: "",
        schema.ImgCols.CONCEPT_PRESENT_A1: "",
        schema.ImgCols.CONCEPT_PRESENT_A2: "",
        schema.ImgCols.CONCEPT_PRESENT_FINAL: "",
        schema.ImgCols.IMAGE_SAFETY_A1: "",
        schema.ImgCols.IMAGE_SAFETY_A2: "",
        schema.ImgCols.IMAGE_SAFETY_FINAL: "",
        schema.ImgCols.DISAGREEMENT_RESOLVED_BY: "",
    })
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="432 전체 프롬프트 생성 (기본: pilot 6개)")
    ap.add_argument("--overwrite", action="store_true", help="기존 결과 파일 및 이미지를 새로 덮어쓴다")
    ap.add_argument("--resume", action="store_true", help="기존 성공한 생성 결과는 건너뛴다")
    ap.add_argument("--seed", type=int, default=42, help="생성 시드 (기본: 42)")
    ap.add_argument("--device", default=None, help="실행 디바이스 (cuda / cpu / mps)")
    args = ap.parse_args()

    if not PROMPTS_CSV.exists():
        log.error("프롬프트 파일이 없습니다: %s", PROMPTS_CSV)
        return 1
    prompts = read_csv(str(PROMPTS_CSV))

    mode = "full" if args.full else "pilot"
    gen_csv = GEN_OUT_CSV if mode == "full" else PILOT_GEN_OUT_CSV
    labels_csv = LABELS_OUT_CSV if mode == "full" else PILOT_LABELS_OUT_CSV
    meta_json = META_OUT_JSON if mode == "full" else PILOT_META_OUT_JSON

    if mode == "pilot":
        prompts = [p for p in prompts if p["prompt_id"] in PILOT_PROMPT_IDS]
        missing = set(PILOT_PROMPT_IDS) - {p["prompt_id"] for p in prompts}
        if missing:
            log.error("pilot 프롬프트가 prompts.csv 에 없습니다: %s", sorted(missing))
            return 1

    total = len(prompts)
    log.info("mode=%s  프롬프트 %d개  seed=%d", mode, total, args.seed)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done_gen: dict[str, dict] = {}
    existing_labels: dict[str, dict] = {}

    if gen_csv.exists():
        if args.resume:
            for r in read_csv(str(gen_csv), schema.GENERATION_COLUMNS):
                if not str(r.get(schema.GenCols.ERROR_TYPE, "")).strip():
                    done_gen[r[schema.GenCols.GENERATION_ID]] = r
            log.info("--resume: 기존 성공 행 %d건 건너뜁니다.", len(done_gen))
        elif not args.overwrite:
            log.error("이미지 결과 파일이 존재합니다: %s (--overwrite 또는 --resume 사용)", gen_csv)
            return 1

    if labels_csv.exists() and args.resume:
        for r in read_csv(str(labels_csv), schema.IMAGE_LABEL_COLUMNS):
            existing_labels[r[schema.ImgCols.GENERATION_ID]] = r

    log.info("AltDiffusion 모델 로딩 중 (BAAI/AltDiffusion-m18)...")
    adapter = load_real_altdiffusion_adapter(
        device=args.device,
        input_ids_dump_path=str(OUT_DIR / "input_ids_sample.jsonl"),
        input_ids_dump_n=10,
    )
    log.info("모델 로드 완료: %s @ %s", adapter.model_id, adapter.revision)

    n_error = 0
    n_new = 0
    n_consecutive_errors = 0
    aborted = False
    t_start = time.time()

    def checkpoint() -> None:
        g_rows = list(done_gen.values())
        g_rows.sort(key=lambda r: r[schema.GenCols.GENERATION_ID])
        check_primary_key(g_rows, [schema.GenCols.GENERATION_ID], str(gen_csv))
        write_csv(str(gen_csv), g_rows, schema.GENERATION_COLUMNS)

        l_rows = list(existing_labels.values())
        l_rows.sort(key=lambda r: r[schema.ImgCols.GENERATION_ID])
        check_primary_key(l_rows, [schema.ImgCols.GENERATION_ID], str(labels_csv))
        write_csv(str(labels_csv), l_rows, schema.IMAGE_LABEL_COLUMNS)

    for p in prompts:
        pid, raw = p["prompt_id"], p["raw_prompt"]
        gid = make_generation_id(pid, adapter.model_id, args.seed)
        out_img_rel = f"images/{gid}.png"
        out_img_abs = OUT_DIR / out_img_rel

        if gid in done_gen and out_img_abs.exists():
            continue

        t0 = time.perf_counter()
        gen_out = adapter.generate(raw, args.seed, str(out_img_abs))
        runtime_ms = round((time.perf_counter() - t0) * 1000, 3)

        if gen_out.error_type:
            n_error += 1
            n_consecutive_errors += 1
            log.error("%s (gen_id=%s) 생성 실패: %s", pid, gid, gen_out.error_type)
            grow = make_empty_gen_row(gid, pid, adapter.model_id, args.seed,
                                      gen_out.error_type, runtime_ms)
        else:
            n_consecutive_errors = 0
            grow = {
                schema.GenCols.GENERATION_ID: gid,
                schema.GenCols.PROMPT_ID: pid,
                schema.GenCols.GENERATOR_ID: adapter.model_id,
                schema.GenCols.SEED: args.seed,
                schema.GenCols.IMAGE_PATH: out_img_rel,
                schema.GenCols.RUNTIME_MS: runtime_ms,
                schema.GenCols.ERROR_TYPE: "",
            }
            if gid not in existing_labels:
                lrow = make_label_row(gid, pid)
                existing_labels[gid] = lrow

        done_gen[gid] = grow
        n_new += 1

        # 매 이미지 생성 직후 MPS/CUDA 캐시 및 Python 메모리 강제 해제
        try:
            import gc
            import torch
            if hasattr(torch, "mps") and torch.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except ImportError:
            pass

        if n_new % CHECKPOINT_EVERY == 0:
            checkpoint()
            elapsed = time.time() - t_start
            rate = n_new / elapsed
            remaining = max(total - len(done_gen), 0) / rate if rate > 0 else float("nan")
            log.info("진행 %d/%d (%.1fs 경과, 남은 시간 약 %.0fs, 오류 %d건)",
                     len(done_gen), total, elapsed, remaining, n_error)

        if n_consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            log.error("연속 %d개 생성 실패 — 중단합니다.", n_consecutive_errors)
            aborted = True
            break

    checkpoint()
    elapsed = time.time() - t_start

    meta = {
        "mode": mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_id": adapter.model_id,
        "model_revision": adapter.revision,
        "seed": args.seed,
        "n_prompts": total,
        "n_generated": len(done_gen),
        "n_new": n_new,
        "n_errors": n_error,
        "complete": (not aborted) and len(done_gen) == total and n_error == 0,
        "elapsed_seconds": round(elapsed, 2),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 78}")
    print(f"  AltDiffusion 이미지 생성 결과 요약")
    print(f"  mode            : {mode}")
    print(f"  생성 성공/전체    : {len(done_gen) - n_error} / {total} (신규 {n_new})")
    print(f"  오류 건수        : {n_error}")
    print(f"  소요 시간        : {elapsed:.1f}s")
    print(f"  결과 CSV        : {gen_csv}")
    print(f"  라벨 CSV 템플릿  : {labels_csv}")
    print(f"  이미지 저장 위치 : {IMAGES_DIR}")
    print(f"{'=' * 78}")

    return 0 if meta["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
