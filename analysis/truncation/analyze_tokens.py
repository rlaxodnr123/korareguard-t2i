#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_tokens.py — KoRareGuard-T2I / Student 2

동일한 raw_prompt 를 세 가지 분석 조건에 독립적으로 넣어
각 컴포넌트가 핵심 표현을 실제로 어디까지 읽는지 기록한다.

  조건 1  SGuard native            절단 없음 (native context 131,072)
  조건 2  SGuard constrained_77    user content 예산 77 (연구가 정의한 experimental cap)
  ---     AltDiffusion native      tokenizer.model_max_length (runtime 값)

primary key 는 prompt_id x model_id x input_policy 다.
같은 tokenizer 라도 input policy 가 다르면 다른 조건이므로
prompt_id x model_id 만으로는 조건 1 과 조건 2 를 구분할 수 없다.

주의:
  - 77 은 본 연구가 정의한 experimental token cap 이며
    SGuard 의 native maximum context length 가 아니다.
  - SGuard 의 cap 은 chat template 을 제외한 user content 토큰에 적용한다.
    template overhead 가 약 1,480 토큰이라 전체 입력에 77 을 걸 수 없다.
  - raw_prompt 는 파괴적으로 자르지 않는다. 절단은 분석 단계에서만 적용한다.

사용법:
    .venv\\Scripts\\python.exe analysis/truncation/analyze_tokens.py            # pilot
    .venv\\Scripts\\python.exe analysis/truncation/analyze_tokens.py --full     # 전체 432
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("analyze_tokens")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "tokenizer"))
from key_span import InputPolicy, analyze_key_span, KeySpanResult, STATUS_OK  # noqa: E402
from src.common import schema  # noqa: E402  — 컬럼명/값 어휘의 팀 공용 SSOT
from src.common import config  # noqa: E402  — 모델 id / revision 의 팀 공용 SSOT

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"
OUT_DIR = REPO / "analysis" / "truncation"
TOKEN_LOG_DIR = REPO / "outputs" / "token_logs"

# 모델 id 와 revision 은 config 에서 가져온다. 예전에는 여기 문자열을 복사해 뒀고
# from_pretrained 에 revision 을 넘기지 않아, 캐시가 비워지고 업스트림이 바뀌면
# 조용히 다른 스냅샷으로 실행될 수 있었다. resolve_revision() 이 실제 로드된 SHA 를
# 되읽어 기록하므로 거짓 기록은 없었지만, 기록만으로는 재현이 보장되지 않는다.
SGUARD_MODEL_ID = config.SGUARD_MODEL_ID
SGUARD_REVISION = config.SGUARD_REVISION
ALTDIFF_MODEL_ID = config.ALTDIFF_MODEL_ID
ALTDIFF_REVISION = config.ALTDIFF_REVISION
ALTDIFF_TOKENIZER_SUBFOLDER = "tokenizer"

# 본 연구가 정의한 experimental token cap. SGuard native limit 이 아니다.
EXPERIMENTAL_TOKEN_CAP = 77

# SGuard 의 native context length (config.max_position_embeddings, 실측)
SGUARD_NATIVE_CONTEXT = 131072

# pilot 대상. 두 가지 기준으로 고른다.
#   1) semantic pair 검토가 끝나지 않은 concept(VIOL_14, VIOL_15, CRIM_22)은 제외
#   2) full / partial / none 세 경로를 모두 밟아야 한다.
#      partial 이 한 번도 안 나오면 그 코드 경로가 검증되지 않은 채로 남는다.
PILOT_CONCEPTS = [
    "SAFE_CULT_01",     # 상모돌리기 / 전통 모자를 돌리는 농악 공연
    "SAFE_CULT_03",     # 쥐불놀이 / 논밭 두렁에 불을 놓는 전통 민속놀이
                        #   -> SGuard 와 AltDiffusion 양쪽에서 partial 을 발생시킨다
    "SAFE_VOCAB_09",    # 밤하늘 미리내 / 밤하늘을 가로지르는 은하수
    "UNSAFE_VIOL_13",   # 예리한 비수 / 위협적인 칼
    "UNSAFE_NSFW_17",   # 옛 속곳 차림 / 속옷 차림의 인물
]

# CSV 컬럼 순서. 값 어휘(full/partial/none 등)는 key_span.py 의 상수를 따른다.
FIELDNAMES = [
    # 키
    "prompt_id", "concept_id", "model_id", "model_role", "input_policy",
    # 정책
    "experimental_token_cap", "content_token_budget", "declared_max_length",
    "special_tokens_reserved", "max_length_effective", "max_length_source",
    # 데이터셋 조건
    "safety_label", "rarity_label", "hazard_category", "sguard_category",
    "length_level", "position_level", "key_expression",
    # 원본
    "character_count", "key_character_count", "key_char_start", "key_char_end",
    # PASS 1 — pre-truncation
    "total_tokens_pretrunc", "key_token_count_original",
    "key_start_pretrunc", "key_end_pretrunc",
    "key_start_ratio", "key_center_ratio", "key_end_ratio",
    "key_tokens_per_character",
    # PASS 2 — actual used input
    "total_tokens_used", "prompt_truncated",
    "key_start_token", "key_end_token",
    "key_tokens_retained", "key_retention_ratio", "key_visibility",
    "key_chars_retained", "key_chars_covered", "key_chars_uncovered",
    "key_retention_ratio_char",
    # byte-level BPE 진단
    "tokens_sharing_char_offset", "key_split_mid_character",
    # 유효 입력
    "decoded_used_input", "removed_text",
    # SGuard 전용
    "template_overhead_tokens", "formatted_pretrunc_tokens",
    # tokenizer 메타데이터
    "tokenizer_class", "tokenizer_revision", "padding_side", "truncation_side",
    "special_token_count",
    # 신뢰성
    "analysis_status", "error_message", "token_ids_path", "runtime_ms",
]


def resolve_revision(repo_id: str, subfolder: Optional[str] = None) -> str:
    """실제로 로드된 snapshot 의 commit SHA. 재현성에 필요하다 (§45)."""
    from huggingface_hub import hf_hub_download
    try:
        kw = {"subfolder": subfolder} if subfolder else {}
        p = Path(hf_hub_download(repo_id, "tokenizer_config.json", **kw))
        parts = p.parts
        if "snapshots" in parts:
            return parts[parts.index("snapshots") + 1]
    except Exception as exc:
        log.warning("revision 확인 실패 (%s): %s", repo_id, exc)
    return "unknown"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def sguard_formatted_tokens(tok: Any, prompt: str) -> int:
    """SGuard 가 실제 추론 시 받는 전체 입력 토큰 수 (chat template 포함).

    template 은 'content' 가 아니라 'prompt' 키를 쓴다. 이 형식이 어긋나면
    학생 4 가 실제로 넣는 입력과 우리가 기록한 입력이 달라진다.
    """
    out = tok.apply_chat_template(
        [{"role": "user", "prompt": prompt}], tokenize=True, add_generation_prompt=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if len(out) > 0 and isinstance(out[0], (list, tuple)):
        out = out[0]
    return len(out)


def build_conditions(sg_tok: Any, ad_tok: Any) -> list[dict[str, Any]]:
    """세 조건 모두 **content 토큰 공간**에서 정의한다.

    content 공간으로 통일해야 두 컴포넌트의 위치·retention 지표를 서로 비교할 수 있다.
    special token 과 chat template 은 각 컴포넌트의 고정 오버헤드로 따로 기록한다.
    """
    ad_declared = int(ad_tok.model_max_length)
    # diffusers 는 tokenizer(padding="max_length", max_length=77, truncation=True) 를 쓴다.
    # HF 는 special token 자리를 먼저 확보한 뒤 content 를 자르므로
    # 실제 content 예산은 77 이 아니라 77 - 2 = 75 다. (<s> content... </s>)
    ad_n_special = len(ad_tok("", add_special_tokens=True)["input_ids"])
    ad_content_budget = ad_declared - ad_n_special
    return [
        {
            "label": "SGuard native",
            "tokenizer": sg_tok,
            "policy": InputPolicy(
                name="native", model_id=SGUARD_MODEL_ID, model_role=schema.ROLE_TEXT_SAFETY,
                add_special_tokens=False, cap=None, cap_kind="none",
                declared_max_length=SGUARD_NATIVE_CONTEXT, special_tokens_reserved=0),
            "max_length_effective": SGUARD_NATIVE_CONTEXT,
            "max_length_source": "model config max_position_embeddings",
            "is_sguard": True,
        },
        {
            "label": f"SGuard constrained_{EXPERIMENTAL_TOKEN_CAP}",
            "tokenizer": sg_tok,
            "policy": InputPolicy(
                name=f"constrained_{EXPERIMENTAL_TOKEN_CAP}", model_id=SGUARD_MODEL_ID,
                model_role=schema.ROLE_TEXT_SAFETY, add_special_tokens=False,
                cap=EXPERIMENTAL_TOKEN_CAP, cap_kind="user_content",
                declared_max_length=SGUARD_NATIVE_CONTEXT, special_tokens_reserved=0),
            "max_length_effective": EXPERIMENTAL_TOKEN_CAP,
            "max_length_source": "experimental cap on user content (not native limit)",
            "is_sguard": True,
        },
        {
            "label": "AltDiffusion native",
            "tokenizer": ad_tok,
            "policy": InputPolicy(
                name="native", model_id=ALTDIFF_MODEL_ID, model_role=schema.ROLE_GENERATOR,
                add_special_tokens=False, cap=ad_content_budget, cap_kind="native",
                declared_max_length=ad_declared,
                special_tokens_reserved=ad_n_special),
            "max_length_effective": ad_declared,
            "max_length_source": (
                f"tokenizer.model_max_length {ad_declared} "
                f"- {ad_n_special} special tokens = {ad_content_budget} content budget"),
            "is_sguard": False,
        },
    ]


def tokenizer_meta(tok: Any, revision: str) -> dict[str, Any]:
    return {
        "tokenizer_class": tok.__class__.__name__,
        "tokenizer_revision": revision,
        "padding_side": tok.padding_side,
        "truncation_side": tok.truncation_side,
        "special_token_count": len(getattr(tok, "all_special_ids", []) or []),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="432 전체 분석 (기본은 pilot concept 만)")
    ap.add_argument("--overwrite", action="store_true",
                    help="기존 결과 파일을 덮어쓴다")
    ap.add_argument("--prompts", default=None,
                    help="입력 prompts CSV. 기본은 benchmarks/prompts/prompts.csv. "
                         "벤치마크 변형(prompts_77.csv 등)을 비교할 때 쓴다.")
    args = ap.parse_args()

    prompts_path = Path(args.prompts) if args.prompts else PROMPTS_CSV
    if not prompts_path.is_absolute():
        prompts_path = (REPO / prompts_path).resolve()
    if not prompts_path.exists():
        log.error("입력 파일이 없습니다: %s", prompts_path)
        return 1
    # 변형 파일은 출력 이름에 표시해 기본 결과를 덮어쓰지 않게 한다.
    #   prompts.csv -> ""   prompts_77.csv -> "_77"   prompts_127.csv -> "_127"
    variant = prompts_path.stem.replace("prompts", "")

    mode = "full" if args.full else "pilot"
    suffix = variant + ("" if args.full else "_pilot")
    out_csv = OUT_DIR / f"tokenization_results{suffix}.csv"
    out_meta = OUT_DIR / f"run_metadata{suffix}.json"

    if out_csv.exists() and not args.overwrite:
        log.error("이미 존재합니다: %s  (덮어쓰려면 --overwrite)", out_csv)
        return 1

    with open(prompts_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if mode == "pilot":
        rows = [r for r in rows if r["concept_id"] in PILOT_CONCEPTS]
    log.info("mode=%s  프롬프트 %d개", mode, len(rows))
    if not rows:
        log.error("대상 프롬프트가 없습니다")
        return 1

    from transformers import AutoTokenizer
    import transformers

    sg_tok = AutoTokenizer.from_pretrained(SGUARD_MODEL_ID, revision=SGUARD_REVISION)
    ad_tok = AutoTokenizer.from_pretrained(ALTDIFF_MODEL_ID, revision=ALTDIFF_REVISION,
                                           subfolder=ALTDIFF_TOKENIZER_SUBFOLDER)
    revisions = {
        SGUARD_MODEL_ID: resolve_revision(SGUARD_MODEL_ID),
        ALTDIFF_MODEL_ID: resolve_revision(ALTDIFF_MODEL_ID, ALTDIFF_TOKENIZER_SUBFOLDER),
    }
    conditions = build_conditions(sg_tok, ad_tok)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 토큰 배열은 CSV 셀이 아니라 조건별 JSONL 로 분리 저장한다 (§46)
    log_files = {}
    for cond in conditions:
        pol = cond["policy"]
        name = f"{'sguard' if cond['is_sguard'] else 'altdiff'}_{pol.name}{suffix}.jsonl"
        log_files[cond["label"]] = (TOKEN_LOG_DIR / name).open("w", encoding="utf-8")

    out_rows: list[dict[str, Any]] = []
    n_error = 0
    t_start = time.time()

    for r in rows:
        raw, key = r["raw_prompt"], r["key_expression"]
        for cond in conditions:
            tok, pol = cond["tokenizer"], cond["policy"]
            t0 = time.perf_counter()
            res: KeySpanResult = analyze_key_span(
                tok, raw, key, pol, prompt_id=r["prompt_id"],
                max_length_source=cond["max_length_source"])
            runtime_ms = (time.perf_counter() - t0) * 1000

            if res.analysis_status != STATUS_OK:
                n_error += 1
                log.error("%s / %s : %s", r["prompt_id"], cond["label"], res.error_message)

            d = asdict(res)
            d.pop("token_rows", None)
            d.update({
                "concept_id": r["concept_id"],
                "safety_label": r["safety_label"],
                "rarity_label": r["rarity_label"],
                "hazard_category": r["hazard_category"],
                "sguard_category": r["sguard_category"],
                "length_level": r["length_level"],
                "position_level": r["position_level"],
                "key_expression": key,
                "max_length_effective": cond["max_length_effective"],
                "max_length_source": cond["max_length_source"],
                # 팀 스키마(TokCols)가 요구하는 컬럼이라 남긴다. 다만 벽시계 값이므로
                # 이 CSV 는 재생성해도 바이트가 달라진다 — 최신 여부를 확인할 때는
                # 파일 해시가 아니라 runtime_ms 를 뺀 나머지 55개 컬럼을 비교할 것.
                "runtime_ms": round(runtime_ms, 3),
            })
            d.update(tokenizer_meta(tok, revisions[pol.model_id]))

            if cond["is_sguard"]:
                fmt = sguard_formatted_tokens(sg_tok, raw)
                d["formatted_pretrunc_tokens"] = fmt
                d["template_overhead_tokens"] = fmt - res.total_tokens_pretrunc
            else:
                d["formatted_pretrunc_tokens"] = ""
                d["template_overhead_tokens"] = ""

            fh = log_files[cond["label"]]
            d["token_ids_path"] = str(Path(fh.name).relative_to(REPO)).replace("\\", "/")
            fh.write(json.dumps({
                "prompt_id": r["prompt_id"],
                "model_id": pol.model_id,
                "input_policy": pol.name,
                "tokens": res.token_rows,
            }, ensure_ascii=False) + "\n")

            out_rows.append({k: d.get(k, "") for k in FIELDNAMES})

    for fh in log_files.values():
        fh.close()
    elapsed = time.time() - t_start

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(out_rows)

    meta = {
        "mode": mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "n_prompts": len(rows),
        "n_conditions": len(conditions),
        "n_rows": len(out_rows),
        "n_error_rows": n_error,
        "elapsed_seconds": round(elapsed, 2),
        "pilot_concepts": PILOT_CONCEPTS if mode == "pilot" else None,
        "experimental_token_cap": EXPERIMENTAL_TOKEN_CAP,
        "experimental_token_cap_note":
            "연구가 정의한 user content 예산. SGuard native limit 이 아님.",
        "sguard_native_context": SGUARD_NATIVE_CONTEXT,
        "model_revisions": revisions,
        "conditions": [
            {"label": c["label"], "model_id": c["policy"].model_id,
             "input_policy": c["policy"].name,
             "add_special_tokens": c["policy"].add_special_tokens,
             "cap": c["policy"].cap, "cap_kind": c["policy"].cap_kind,
             "max_length_source": c["max_length_source"]}
            for c in conditions
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "transformers": transformers.__version__,
        },
        "inputs": {"prompts_csv": str(prompts_path.relative_to(REPO)).replace("\\", "/")},
    }
    for m in ("tokenizers", "sentencepiece", "pandas", "numpy", "torch", "diffusers"):
        try:
            meta["environment"][m] = __import__(m).__version__
        except Exception:
            meta["environment"][m] = None
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 78}")
    print(f"  mode          : {mode}")
    print(f"  프롬프트       : {len(rows)}  x  조건 {len(conditions)}  =  {len(out_rows)} 행")
    print(f"  error 행       : {n_error}")
    print(f"  소요           : {elapsed:.1f}s")
    print(f"  결과 CSV       : {out_csv.relative_to(REPO)}")
    print(f"  실행 메타데이터 : {out_meta.relative_to(REPO)}")
    print(f"  토큰 로그       : {TOKEN_LOG_DIR.relative_to(REPO)}/  ({len(log_files)} 파일)")
    print(f"{'=' * 78}")
    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())
