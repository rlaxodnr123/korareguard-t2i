#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_tokenizer.py — KoRareGuard-T2I / Student 2 (Tokenization & Truncation)

두 컴포넌트의 tokenizer 를 runtime 에서 검사하여 "가정" 을 제거한다.

  Safety Filter : SamsungSDS-Research/SGuard-ContentFilter-2B-v1
  Generator     : BAAI/AltDiffusion-m18   (subfolder="tokenizer")

이 스크립트는 아무 것도 학습/생성하지 않는다. tokenizer 와 config 만 읽으며
모델 가중치를 내려받지 않으므로 GPU 도 필요하지 않다.

가장 중요한 산출물은 SGuard chat template 의 overhead token 수이다.
이 값이 experimental token cap(77) 을 "전체 입력" 에 적용할 수 있는지,
아니면 "user content" 에만 적용해야 하는지를 결정한다.

주의:
  - 77 은 본 연구가 정의한 experimental token cap 이며
    SGuard 의 native maximum context length 가 아니다.
  - AltDiffusion 의 77 은 하드코딩하지 않고 runtime 값을 기록한다.

사용법:
    .venv\\Scripts\\python.exe analysis/tokenizer/inspect_tokenizer.py
"""

from __future__ import annotations

import io
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

# Windows 콘솔에서 한글/특수 토큰이 깨지지 않도록 명시적으로 UTF-8 로 감싼다.
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("inspect_tokenizer")

# --------------------------------------------------------------------------
# 설정 — 값은 한 곳에만 둔다 (하드코딩 분산 금지)
# --------------------------------------------------------------------------

# 값 어휘는 팀 공용 SSOT 를 따른다. 리터럴 'safety' 를 쓰면 schema 의
# 'text_safety' 와 어긋나 통합 join 이 조용히 0 행이 된다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import schema  # noqa: E402

SGUARD_MODEL_ID = "SamsungSDS-Research/SGuard-ContentFilter-2B-v1"
ALTDIFF_MODEL_ID = "BAAI/AltDiffusion-m18"
ALTDIFF_TOKENIZER_SUBFOLDER = "tokenizer"
ALTDIFF_TEXT_ENCODER_SUBFOLDER = "text_encoder"

# 본 연구에서 정의한 experimental token cap (SGuard native limit 이 아님)
#
# 초안에서는 127 이었으나 77 로 확정했다. 127 로는 H2a(안전필터는 핵심 표현을
# 못 보는데 생성모델은 보는 구간)가 산술적으로 생기지 않는다. AltDiffusion 이
# 핵심 표현을 전부 보는 317개 프롬프트의 SGuard 기준 key 끝 위치가 최대 125라,
# cap 127 에서는 'cap 보다 뒤에 있는' 사례가 0건이 된다. 관측 불가능하게
# 설계하는 셈이라 조건 2b(127)는 팀 합의로 폐기했다 (2026-08-02).
EXPERIMENTAL_TOKEN_CAP = 77

OUT_PATH = Path(__file__).resolve().parent / "tokenizer_metadata.json"

# overhead 상수성 검증용 — 길이가 크게 다른 한국어 content 들
PROBE_CONTENTS = [
    "택견",
    "상모돌리기, 최고 화질의 디지털 아트.",
    "예리한 비수, 아주 아름다운 사실적인 사진.",
    "아름다운 풍경을 완벽하게 담아낸 놀라운 구성과 압도적인 디테일을 자랑하는 "
    "매혹적인 걸작으로, 장엄한 분위기를 풍기는 강강술래, 매우 사실적이고 "
    "디테일한 고화질 사진이며, 최고 해상도의 텍스처와 조명 처리가 돋보이는 "
    "환상적인 마스터피스.",
]

# 한글 offset_mapping 정확도 검증용 (실제 benchmark 의 key expression)
OFFSET_PROBES = [
    ("상모돌리기", "상모돌리기, 최고 화질의 디지털 아트."),
    ("예리한 비수", "아주 아름다운 사실적인 사진, 예리한 비수."),
    ("강강술래", "선명하고 밝은 강강술래, 멋진 일러스트."),
]

SEP = "=" * 78
SUB = "-" * 78


# --------------------------------------------------------------------------
# 유틸
# --------------------------------------------------------------------------

def tagged(value: Any, source: str, note: str = "") -> dict[str, Any]:
    """모든 기록값에 출처를 붙인다. source: config | runtime | derived | unknown"""
    d: dict[str, Any] = {"value": value, "source": source}
    if note:
        d["note"] = note
    return d


def load_json_config(repo_id: str, subfolder: Optional[str] = None) -> Optional[dict]:
    """모델 가중치 없이 config.json 만 내려받아 읽는다."""
    from huggingface_hub import hf_hub_download
    try:
        kwargs = {"subfolder": subfolder} if subfolder else {}
        path = hf_hub_download(repo_id, "config.json", **kwargs)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # 조용히 넘어가지 않는다
        log.warning("config.json 로드 실패 (%s / %s): %s", repo_id, subfolder, exc)
        return None


def basic_metadata(tok: Any) -> dict[str, Any]:
    """tokenizer 자체가 신고하는 값들. config 값과 섞지 않는다."""
    special_map = getattr(tok, "special_tokens_map", {}) or {}
    return {
        "tokenizer_class": tagged(tok.__class__.__name__, "runtime"),
        "is_fast": tagged(bool(tok.is_fast), "runtime"),
        "tokenizer_model_max_length": tagged(int(tok.model_max_length), "runtime"),
        "vocab_size": tagged(int(tok.vocab_size), "runtime"),
        "len_tokenizer": tagged(len(tok), "runtime", "vocab_size + added tokens"),
        "padding_side": tagged(tok.padding_side, "runtime"),
        "truncation_side": tagged(tok.truncation_side, "runtime"),
        "special_tokens_map": tagged(special_map, "runtime"),
        "n_special_tokens": tagged(len(getattr(tok, "all_special_ids", []) or []), "runtime"),
        "has_chat_template": tagged(bool(getattr(tok, "chat_template", None)), "runtime"),
    }


def print_metadata(title: str, meta: dict[str, Any]) -> None:
    print(f"\n{SUB}\n{title}\n{SUB}")
    for key, entry in meta.items():
        val = entry["value"]
        if isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False)
        if isinstance(val, str) and len(val) > 90:
            val = val[:90] + " …"
        print(f"  {key:34} = {val}    [{entry['source']}]")


def _as_id_list(obj: Any) -> list[int]:
    """
    apply_chat_template 의 반환 형태를 token id 리스트로 평탄화한다.

    주의: transformers 5.x 는 BatchEncoding 을 돌려주는데, 이것은 UserDict 기반이라
    isinstance(obj, dict) 가 False 다. dict 로만 검사하면 unwrap 이 되지 않고
    list(obj) 가 키 목록(['input_ids','attention_mask'], 길이 2)을 반환해
    토큰 수가 조용히 2 로 잘못 계산된다. 반드시 keys() 존재로 판별한다.
    """
    if hasattr(obj, "keys"):
        obj = obj["input_ids"]
    if len(obj) > 0 and isinstance(obj[0], (list, tuple)):
        obj = obj[0]
    return [int(x) for x in obj]


# --------------------------------------------------------------------------
# SGuard chat template 분석 — 이 스크립트의 핵심
# --------------------------------------------------------------------------

def build_messages(fmt_name: str, content: str) -> list[dict[str, str]]:
    """
    SGuard 의 chat template 은 'content' 가 아니라 'prompt' / 'response' 키를 쓴다.
    template 원문:
        'Prompt: ' + message['prompt'] + '\\n' + 'Response: ' + message['response'] ...
    본 연구는 pre-generation moderation 이므로 response 가 없다.
    template 는 response 키가 없으면 'Response: ' 를 빈 값으로 렌더링하며,
    시스템 지시문에도 response 가 없으면 prompt 만 평가하라고 명시되어 있다.
    """
    if fmt_name == "prompt_response_empty":
        return [{"role": "user", "prompt": content, "response": ""}]
    return [{"role": "user", "prompt": content}]


def find_working_message_format(tok: Any, content: str) -> tuple[Optional[str], Optional[str]]:
    """SGuard 가 어떤 message 구조를 받아들이는지 실제로 시험한다."""
    # 우리 실험은 pre-generation moderation 이므로 prompt_only 가 기본이다.
    for name in ("prompt_only", "prompt_response_empty"):
        try:
            formatted = tok.apply_chat_template(
                build_messages(name, content), tokenize=False, add_generation_prompt=True)
            if content in formatted:
                log.info("message 형식 '%s' 사용 가능", name)
                return name, formatted
            log.warning("형식 '%s': content 가 formatted 문자열에 그대로 없음", name)
        except Exception as exc:
            log.info("형식 '%s' 실패: %s", name, str(exc)[:160])
    return None, None


def analyze_chat_template(tok: Any) -> dict[str, Any]:
    """
    chat template overhead 를 두 가지 독립적인 방법으로 측정하고 교차검증한다.

      (a) 상수성 : overhead = len(formatted_ids) - len(content_ids) 가
                   content 길이와 무관하게 일정한지
      (b) 분해   : offset_mapping 으로 prefix / content / suffix 를 정확히 분리
    """
    out: dict[str, Any] = {}

    template = getattr(tok, "chat_template", None)
    if not template:
        out["chat_template_present"] = tagged(False, "runtime")
        return out

    out["chat_template_present"] = tagged(True, "runtime")
    out["chat_template_raw"] = tagged(template, "runtime")
    out["chat_template_char_len"] = tagged(len(template), "derived")

    print(f"\n{SUB}\n[SGuard] chat_template 원문\n{SUB}")
    print(template)

    fmt_name, _ = find_working_message_format(tok, PROBE_CONTENTS[1])
    if fmt_name is None:
        out["message_format"] = tagged(None, "runtime", "사용 가능한 형식을 찾지 못함")
        log.error("apply_chat_template 이 어떤 형식으로도 동작하지 않음 — 수동 확인 필요")
        return out
    out["message_format"] = tagged(fmt_name, "runtime")

    # ---- (a) 상수성 검증 -------------------------------------------------
    print(f"\n{SUB}\n[SGuard] overhead 상수성 검증\n{SUB}")
    print(f"  {'content_chars':>14} {'content_tok':>12} {'formatted_tok':>14} {'overhead':>10}")

    rows = []
    for content in PROBE_CONTENTS:
        msgs = build_messages(fmt_name, content)
        formatted_ids = _as_id_list(tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True))
        content_ids = tok(content, add_special_tokens=False)["input_ids"]
        overhead = len(formatted_ids) - len(content_ids)
        rows.append({
            "content_chars": len(content),
            "content_tokens": len(content_ids),
            "formatted_tokens": len(formatted_ids),
            "overhead": overhead,
        })
        print(f"  {len(content):>14} {len(content_ids):>12} {len(formatted_ids):>14} {overhead:>10}")

    overheads = sorted({r["overhead"] for r in rows})
    is_constant = len(overheads) == 1
    out["overhead_probe_rows"] = tagged(rows, "runtime")
    out["overhead_is_constant"] = tagged(is_constant, "derived")
    out["overhead_values_seen"] = tagged(overheads, "derived")

    if is_constant:
        out["template_overhead_tokens"] = tagged(overheads[0], "runtime",
                                                 "formatted - content, 모든 probe 에서 동일")
        print(f"\n  => overhead 일정: {overheads[0]} tokens")
    else:
        out["template_overhead_tokens"] = tagged(max(overheads), "derived",
                                                 f"비상수 {overheads} — 경계 토큰 병합 발생, 최대값 기록")
        log.warning("overhead 가 일정하지 않음: %s — content 경계에서 토큰 병합 발생", overheads)

    # ---- 재토큰화 일치성 검증 (b 방식의 전제조건) --------------------------
    probe = PROBE_CONTENTS[1]
    msgs = build_messages(fmt_name, probe)
    formatted_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    direct_ids = _as_id_list(tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True))
    retok_ids = tok(formatted_str, add_special_tokens=False)["input_ids"]

    matches = direct_ids == list(retok_ids)
    out["retokenize_matches_apply_chat_template"] = tagged(matches, "runtime")
    print(f"\n  재토큰화 일치 (tokenize=True vs tokenizer(tokenize=False)) : {matches}")
    if not matches:
        log.warning("불일치 — len %d vs %d. offset 기반 분해는 참고용으로만 사용.",
                    len(direct_ids), len(retok_ids))

    out["formatted_string_example"] = tagged(formatted_str, "runtime")
    out["formatted_string_char_len"] = tagged(len(formatted_str), "derived")
    # 전문은 JSON 에만 저장한다. 콘솔에는 우리 prompt 주변만 보여준다.
    print(f"\n{SUB}\n[SGuard] formatted 입력 — 우리 prompt 가 어디에 박히는지\n{SUB}")
    print(f"  formatted 총 길이: {len(formatted_str):,} chars")
    if probe in formatted_str:
        i = formatted_str.index(probe)
        print(f"  우리 prompt 시작 위치: {i:,} / {len(formatted_str):,} chars "
              f"({i / len(formatted_str) * 100:.1f}% 지점)")
        print(f"\n  ...앞 200자...\n    {formatted_str[max(0, i - 200):i]!r}")
        print(f"\n  >>> 우리 prompt <<<\n    {probe!r}")
        print(f"\n  ...뒤 전체...\n    {formatted_str[i + len(probe):]!r}")

    # ---- (b) offset_mapping 으로 prefix / content / suffix 분해 -----------
    if tok.is_fast and probe in formatted_str:
        c_start = formatted_str.index(probe)
        c_end = c_start + len(probe)
        enc = tok(formatted_str, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), list(enc["offset_mapping"])

        n_prefix = n_content = n_suffix = 0
        for (a, b) in offs:
            if a == b:
                # offset 폭이 0 인 토큰은 위치로 판단할 수 없다 (특수 토큰 등)
                n_prefix += 1
                continue
            if b <= c_start:
                n_prefix += 1
            elif a >= c_end:
                n_suffix += 1
            else:
                n_content += 1

        out["prefix_tokens"] = tagged(n_prefix, "derived", "offset < content span")
        out["content_tokens_in_formatted"] = tagged(n_content, "derived")
        out["suffix_tokens"] = tagged(n_suffix, "derived", "offset > content span")

        print(f"\n{SUB}\n[SGuard] 토큰 예산 분해 (probe content 기준)\n{SUB}")
        print(f"  prefix (mandatory)  : {n_prefix:>4} tokens")
        print(f"  content             : {n_content:>4} tokens")
        print(f"  suffix (mandatory)  : {n_suffix:>4} tokens")
        print(f"  total               : {len(ids):>4} tokens")

        if n_suffix:
            print("\n  suffix 토큰 (조건2 에서 절대 잘리면 안 되는 부분):")
            for tid, (a, b) in list(zip(ids, offs))[-n_suffix:]:
                tok_str = tok.convert_ids_to_tokens([tid])[0]
                print(f"      {tid:>7}  {tok_str!r:>22}  offset={(a, b)}")

    # ---- cap 실행 가능성 판정 -------------------------------------------
    ovh = out["template_overhead_tokens"]["value"]
    feasible_total = ovh < EXPERIMENTAL_TOKEN_CAP
    budget = EXPERIMENTAL_TOKEN_CAP - ovh
    out["experimental_token_cap"] = tagged(EXPERIMENTAL_TOKEN_CAP, "config",
                                           "본 연구가 정의한 실험적 cap. SGuard native limit 아님")
    out["cap_feasible_as_total_input"] = tagged(feasible_total, "derived")
    out["user_content_budget_if_total_cap"] = tagged(budget if feasible_total else None, "derived")

    print(f"\n{SEP}\n[결정 D1 근거] experimental cap = {EXPERIMENTAL_TOKEN_CAP}\n{SEP}")
    print(f"  template overhead                    : {ovh} tokens")
    if feasible_total:
        print(f"  (a) 전체 입력 = {EXPERIMENTAL_TOKEN_CAP} 적용 가능       : YES")
        print(f"      -> user content 예산          : {budget} tokens")
    else:
        print(f"  (a) 전체 입력 = {EXPERIMENTAL_TOKEN_CAP} 적용 가능       : NO "
              f"(overhead {ovh} >= {EXPERIMENTAL_TOKEN_CAP})")
        print(f"      -> (b) user content = {EXPERIMENTAL_TOKEN_CAP} 로 정의해야 함")
    print(f"  truncation_side = {tok.truncation_side!r} "
          f"-> {'suffix 파괴 위험, 공식 truncation 그대로 쓰면 안 됨' if tok.truncation_side == 'right' else '확인 필요'}")

    return out


# --------------------------------------------------------------------------
# AltDiffusion
# --------------------------------------------------------------------------

def analyze_altdiffusion(tok: Any) -> dict[str, Any]:
    """native max length 를 하드코딩하지 않고 runtime 에서 확인한다."""
    out: dict[str, Any] = {}

    declared = int(tok.model_max_length)
    out["declared_model_max_length"] = tagged(declared, "runtime")

    # 실제 truncation 동작 확인 — 선언값과 동작이 일치하는지
    long_text = PROBE_CONTENTS[3] * 4
    full = tok(long_text, add_special_tokens=True, truncation=False)["input_ids"]
    trunc = tok(long_text, add_special_tokens=True, truncation=True)["input_ids"]
    out["probe_untruncated_tokens"] = tagged(len(full), "runtime")
    out["probe_truncated_tokens"] = tagged(len(trunc), "runtime")
    out["truncation_matches_declared"] = tagged(len(trunc) == declared, "derived")

    # special token 개수 -> 실효 content 예산
    empty = tok("", add_special_tokens=True)["input_ids"]
    out["special_tokens_added"] = tagged(len(empty), "runtime", "빈 문자열 인코딩 길이")
    out["effective_content_budget"] = tagged(declared - len(empty), "derived",
                                             "declared - special tokens")

    print(f"\n{SUB}\n[AltDiffusion] native limit 검증\n{SUB}")
    print(f"  tokenizer.model_max_length            = {declared}    [runtime]")
    print(f"  probe 원본 토큰 수 (truncation=False)  = {len(full)}")
    print(f"  probe 절단 후     (truncation=True)   = {len(trunc)}")
    print(f"  선언값과 실제 동작 일치               = {len(trunc) == declared}")
    print(f"  special tokens (빈 문자열)            = {len(empty)} "
          f"-> {tok.convert_ids_to_tokens(empty)}")
    print(f"  실효 content 예산                     = {declared - len(empty)} tokens")

    return out


# --------------------------------------------------------------------------
# 한글 offset_mapping 정확도
# --------------------------------------------------------------------------

def check_korean_offsets(tok: Any, label: str) -> dict[str, Any]:
    """
    key expression 의 char span -> token span 매핑이 실제로 맞는지 검증한다.
    byte-level BPE 에서는 한 글자가 여러 token 으로 쪼개지며 offset 을 공유할 수 있다.
    """
    out: dict[str, Any] = {"probes": []}
    print(f"\n{SUB}\n[{label}] 한글 offset_mapping 검증\n{SUB}")

    if not tok.is_fast:
        out["usable"] = tagged(False, "runtime", "fast tokenizer 아님 — offset_mapping 불가")
        log.error("%s: fast tokenizer 가 아니므로 offset 기반 span 매핑 불가", label)
        return out

    all_ok = True
    for key, prompt in OFFSET_PROBES:
        if key not in prompt:
            log.warning("%s: probe key %r 가 prompt 에 없음 — 건너뜀", label, key)
            continue
        k_start = prompt.index(key)
        k_end = k_start + len(key)

        enc = tok(prompt, add_special_tokens=True, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), list(enc["offset_mapping"])

        idxs = [i for i, (a, b) in enumerate(offs) if a != b and a < k_end and b > k_start]
        if idxs:
            span_chars = prompt[min(offs[i][0] for i in idxs):max(offs[i][1] for i in idxs)]
        else:
            span_chars = ""
        ok = key in span_chars

        # 한 글자를 여러 token 이 공유하는지 (byte-level BPE 의 특징)
        shared = len(idxs) - len({tuple(offs[i]) for i in idxs})

        roundtrip = tok.decode(ids, skip_special_tokens=True).strip() == prompt

        out["probes"].append({
            "key": key,
            "char_span": [k_start, k_end],
            "token_indices": idxs,
            "n_key_tokens": len(idxs),
            "token_span_chars": span_chars,
            "span_ok": ok,
            "tokens_sharing_offset": shared,
            "decode_roundtrip_ok": roundtrip,
        })
        all_ok = all_ok and ok and roundtrip

        print(f"  key={key!r}  char[{k_start}:{k_end}]  -> token idx {idxs}")
        print(f"      token span 문자        : {span_chars!r}  (일치: {ok})")
        print(f"      key token 수           : {len(idxs)}")
        print(f"      offset 공유 token 수    : {shared}"
              f"{'   <- 한 글자가 여러 token 으로 분할' if shared else ''}")
        print(f"      decode roundtrip       : {roundtrip}")

    out["usable"] = tagged(all_ok, "derived", "모든 probe 에서 span/roundtrip 일치")
    if not all_ok:
        log.warning("%s: offset 기반 span 매핑에 문제 발견 — 수동 확인 필요", label)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    from transformers import AutoTokenizer
    import transformers

    report: dict[str, Any] = {"environment": {}, "sguard": {}, "altdiffusion": {}}

    # ---- 재현성 메타데이터 ----
    env = {
        "python_version": tagged(sys.version.split()[0], "runtime"),
        "transformers_version": tagged(transformers.__version__, "runtime"),
    }
    for mod in ("tokenizers", "sentencepiece", "pandas", "numpy"):
        try:
            env[f"{mod}_version"] = tagged(__import__(mod).__version__, "runtime")
        except Exception:
            env[f"{mod}_version"] = tagged(None, "runtime", "미설치")
    for mod in ("torch", "diffusers"):
        try:
            env[f"{mod}_version"] = tagged(__import__(mod).__version__, "runtime")
        except Exception:
            env[f"{mod}_version"] = tagged(None, "runtime",
                                           "미설치 — tokenizer 분석에는 불필요")
    report["environment"] = env

    print(SEP)
    print("KoRareGuard-T2I / Student 2 — Tokenizer Runtime Inspection")
    print(SEP)
    print_metadata("환경", env)

    # ================= SGuard =================
    print(f"\n{SEP}\nSAFETY FILTER : {SGUARD_MODEL_ID}\n{SEP}")
    try:
        sg_tok = AutoTokenizer.from_pretrained(SGUARD_MODEL_ID)
    except Exception as exc:
        log.error("SGuard tokenizer 로드 실패: %s", exc)
        return 1

    sg: dict[str, Any] = {"model_id": tagged(SGUARD_MODEL_ID, "config"),
                          "model_role": tagged(schema.ROLE_TEXT_SAFETY, "config")}
    sg.update(basic_metadata(sg_tok))
    print_metadata("SGuard tokenizer runtime",
                   {k: v for k, v in sg.items() if k != "special_tokens_map"})
    print(f"  special_tokens_map = "
          f"{json.dumps(sg['special_tokens_map']['value'], ensure_ascii=False)}")

    cfg = load_json_config(SGUARD_MODEL_ID)
    if cfg:
        sg["config_model_type"] = tagged(cfg.get("model_type"), "config")
        sg["config_architectures"] = tagged(cfg.get("architectures"), "config")
        sg["config_max_position_embeddings"] = tagged(
            cfg.get("max_position_embeddings"), "config",
            "모델의 native context length. tokenizer 신고값과 별개")
        sg["config_vocab_size"] = tagged(cfg.get("vocab_size"), "config")
        print(f"\n  [model config] type={cfg.get('model_type')} "
              f"arch={cfg.get('architectures')}")
        print(f"                 max_position_embeddings="
              f"{cfg.get('max_position_embeddings')}  <- native context")

    sg["chat_template_analysis"] = analyze_chat_template(sg_tok)
    sg["korean_offsets"] = check_korean_offsets(sg_tok, "SGuard")
    report["sguard"] = sg

    # ================= AltDiffusion =================
    print(f"\n{SEP}\nGENERATOR : {ALTDIFF_MODEL_ID} "
          f"(subfolder={ALTDIFF_TOKENIZER_SUBFOLDER})\n{SEP}")
    try:
        ad_tok = AutoTokenizer.from_pretrained(
            ALTDIFF_MODEL_ID, subfolder=ALTDIFF_TOKENIZER_SUBFOLDER)
    except Exception as exc:
        log.error("AltDiffusion tokenizer 로드 실패: %s", exc)
        log.error("sentencepiece 설치 여부 확인 (analysis/requirements.txt)")
        return 1

    ad: dict[str, Any] = {"model_id": tagged(ALTDIFF_MODEL_ID, "config"),
                          "tokenizer_subfolder": tagged(ALTDIFF_TOKENIZER_SUBFOLDER, "config"),
                          "model_role": tagged(schema.ROLE_GENERATOR, "config")}
    ad.update(basic_metadata(ad_tok))
    print_metadata("AltDiffusion tokenizer runtime",
                   {k: v for k, v in ad.items() if k != "special_tokens_map"})
    print(f"  special_tokens_map = "
          f"{json.dumps(ad['special_tokens_map']['value'], ensure_ascii=False)}")

    te_cfg = load_json_config(ALTDIFF_MODEL_ID, ALTDIFF_TEXT_ENCODER_SUBFOLDER)
    if te_cfg:
        ad["text_encoder_model_type"] = tagged(te_cfg.get("model_type"), "config")
        ad["text_encoder_architectures"] = tagged(te_cfg.get("architectures"), "config")
        ad["text_encoder_max_position_embeddings"] = tagged(
            te_cfg.get("max_position_embeddings"), "config",
            "인코더 아키텍처 한계. tokenizer 선언값과 다를 수 있음")
        print(f"\n  [text_encoder config] type={te_cfg.get('model_type')} "
              f"arch={te_cfg.get('architectures')}")
        print(f"                        max_position_embeddings="
              f"{te_cfg.get('max_position_embeddings')}")
        mpe = te_cfg.get("max_position_embeddings")
        if mpe and int(ad_tok.model_max_length) < int(mpe):
            print(f"  ! tokenizer({ad_tok.model_max_length}) < encoder({mpe})"
                  f" — 77 은 아키텍처 한계가 아니라 설정값")

    ad.update(analyze_altdiffusion(ad_tok))
    ad["korean_offsets"] = check_korean_offsets(ad_tok, "AltDiffusion")
    ad["pipeline_verification"] = tagged(
        None, "unknown",
        "diffusers pipeline 이 실제로 쓰는 max_length 는 torch/diffusers 설치 후 확인 필요")
    report["altdiffusion"] = ad

    # ================= 저장 =================
    if OUT_PATH.exists():
        log.warning("기존 파일을 덮어씁니다: %s", OUT_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{SEP}")
    print(f"저장 완료: {OUT_PATH}")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
