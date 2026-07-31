#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_sguard_behavior.py — KoRareGuard-T2I / Student 2 / PILOT GATE 0a

SGuard 를 실모델로 돌려서 네 가지를 확인한다.

  1. 모델이 실제로 라벨 전용 토큰(49159~49168)을 생성하는가
     -> tokenizer 가 그렇게 인코딩한다는 것까지만 확인돼 있었다.
        모델이 글자 단위로 뱉으면 logit 기반 unsafe_score 설계가 성립하지 않는다.

  2. response="" 에서 정상 동작하는가
     -> chat template 의 시스템 지시문에 다음 두 문장이 함께 들어 있다.
          "If the Response is missing or says 'None', evaluate the Prompt alone"
          "Do not assign risk levels based solely on the Prompt"
        본 연구는 pre-generation moderation 이라 response 가 항상 비어 있다.
        이 조건에서 전부 safe 로 판정되면 연구 설계 자체가 흔들린다. (Limitation 5)

  3. safe / unsafe 를 구분하는가

  4. (보너스) user content 를 자르면 판정이 뒤집히는가 — H1 메커니즘 미리보기

또한 build_sguard_input() 의 token splice 구현을 여기서 검증한다.
잘린 content 를 decode 해서 template 에 다시 넣으면 byte-level BPE 경계에서
U+FFFD 가 생겨 288개 중 67개(23.3%)에서 token id 가 달라진다. 그래서
decode 를 거치지 않고 token id 수준에서 prefix + content[:budget] + suffix 로 잇는다.

주의: 최초 실행 시 SGuard 가중치(약 5GB)를 내려받는다. CPU 로도 돌지만 느리다.
연구 안전: 안전 분류기를 벤치마크 프롬프트에 돌리는 것으로, 유해 콘텐츠를
생성하지 않는다. 출력은 카테고리별 safe/unsafe 라벨뿐이다.

사용법:
    .venv\\Scripts\\python.exe analysis/tokenizer/verify_sguard_behavior.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Optional

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
PROMPTS_CSV = REPO / "benchmarks" / "prompts" / "prompts.csv"
OUT_PATH = Path(__file__).resolve().parent / "sguard_behavior_gate.json"

MODEL_ID = "SamsungSDS-Research/SGuard-ContentFilter-2B-v1"
REVISION = "870ae18c091f06f8f96e4119051f4cd063c83481"
EXPERIMENTAL_TOKEN_CAP = 77
CATEGORIES = ("crime", "manipulation", "privacy", "sexual", "violence")

# analysis/tokenizer/sguard_label_tokens.py 로 조사한 값
LABEL_TOKEN_IDS = {
    "crime":        {"safe": 49159, "unsafe": 49160},
    "manipulation": {"safe": 49161, "unsafe": 49162},
    "privacy":      {"safe": 49163, "unsafe": 49164},
    "sexual":       {"safe": 49165, "unsafe": 49166},
    "violence":     {"safe": 49167, "unsafe": 49168},
}
ALL_LABEL_IDS = {v for d in LABEL_TOKEN_IDS.values() for v in d.values()}

# front 위치 = 핵심 표현이 절단과 무관하게 항상 보임 -> 모델 판정력 자체를 본다
GATE_PROMPTS = [
    "SAFE_CULT_01_RARE_SHORT_FRONT",       # safe  / rare   상모돌리기
    "SAFE_CULT_01_COMMON_SHORT_FRONT",     # safe  / common
    "UNSAFE_VIOL_13_COMMON_SHORT_FRONT",   # unsafe/ common 위협적인 칼
    "UNSAFE_VIOL_13_RARE_SHORT_FRONT",     # unsafe/ rare   예리한 비수
    "UNSAFE_CRIM_24_COMMON_SHORT_FRONT",   # unsafe/ common 방화
    "UNSAFE_NSFW_18_COMMON_SHORT_FRONT",   # unsafe/ common 선정성
]
# 절단이 판정을 뒤집는지 보는 쌍 (key 가 back -> constrained_77 에서 사라짐)
TRUNCATION_PROBE = "UNSAFE_VIOL_13_RARE_NEAR_LIMIT_BACK"

SEP = "=" * 92
report: dict[str, Any] = {"model_id": MODEL_ID, "revision": REVISION, "steps": {}}


def step(name: str, ok: bool, detail: str = "") -> None:
    report["steps"][name] = {"ok": ok, "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# build_sguard_input — 학생 2·4 가 공유해야 하는 함수의 초안
# ---------------------------------------------------------------------------
def build_sguard_input(tok: Any, raw_prompt: str,
                       content_budget: Optional[int] = None,
                       response: str = "") -> dict[str, Any]:
    """SGuard 에 실제로 들어갈 token id 열을 만든다.

    cap 은 user content 토큰에만 적용한다. chat template 의 prefix/suffix 는
    그대로 보존한다 (truncation_side='right' 라 tokenizer 내장 truncation 을
    쓰면 suffix 가 파괴된다).

    decode 를 거치지 않는 것이 핵심이다. 전체 문자열을 한 번만 토큰화한 뒤
    그 결과를 잘라 붙이므로 U+FFFD 도 생기지 않고 경계 병합 동작도 보존된다.
    """
    msgs = [{"role": "user", "prompt": raw_prompt, "response": response}]
    formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    n_occur = formatted.count(raw_prompt)
    if n_occur != 1:
        raise ValueError(
            f"formatted 안에서 raw_prompt 가 {n_occur}회 등장 — content 구간을 특정할 수 없다")

    enc = tok(formatted, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offs = [tuple(o) for o in enc["offset_mapping"]]

    c_start = formatted.index(raw_prompt)
    c_end = c_start + len(raw_prompt)
    idx = [i for i, (a, b) in enumerate(offs) if a != b and a < c_end and b > c_start]
    if not idx:
        raise ValueError("content 구간에 대응하는 토큰이 없다")

    s, e = min(idx), max(idx)
    prefix, content, suffix = ids[:s], ids[s:e + 1], ids[e + 1:]
    used = content if content_budget is None else content[:content_budget]

    return {
        "token_ids": prefix + used + suffix,
        "n_prefix": len(prefix),
        "n_content_pretrunc": len(content),
        "n_content_used": len(used),
        "n_suffix": len(suffix),
        "content_truncated": len(used) < len(content),
        "content_budget": content_budget,
    }


def parse_generated(tok: Any, gen_ids: list[int]) -> dict[str, Any]:
    """생성 토큰을 라벨 토큰 기준으로 해석한다. 문자열 파싱은 대조용."""
    label_hits = []
    for tid in gen_ids:
        for cat, d in LABEL_TOKEN_IDS.items():
            if tid == d["safe"]:
                label_hits.append((cat, "safe", tid))
            elif tid == d["unsafe"]:
                label_hits.append((cat, "unsafe", tid))
    cats = {c: v for c, v, _ in label_hits}
    return {
        "gen_ids": gen_ids,
        "gen_text": tok.decode(gen_ids, skip_special_tokens=True),
        "n_gen": len(gen_ids),
        "n_label_tokens": len(label_hits),
        "categories_from_tokens": cats,
        "decision": "unsafe" if any(v == "unsafe" for v in cats.values()) else "safe",
        "all_gen_are_label_tokens": bool(gen_ids) and all(
            t in ALL_LABEL_IDS for t in gen_ids),
    }


def run_one(model, tok, torch, raw_prompt: str, budget=None, response="") -> dict[str, Any]:
    built = build_sguard_input(tok, raw_prompt, budget, response)
    ids = torch.tensor([built["token_ids"]])
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=ids,
                             attention_mask=torch.ones_like(ids),
                             max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = out[0, ids.shape[1]:].tolist()
    res = parse_generated(tok, gen)
    res.update(built)
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(PROMPTS_CSV, encoding="utf-8-sig", newline="") as f:
        prompts = {r["prompt_id"]: r for r in csv.DictReader(f)}

    print(SEP)
    print("PILOT GATE 0a — SGuard 실모델 동작 검증")
    print(SEP)
    print(f"  torch {torch.__version__}  |  최초 실행이면 가중치 약 5GB 다운로드")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=torch.float32, low_cpu_mem_usage=True)
        model.eval()
        step("모델 로드", True, f"{type(model).__name__}")
    except Exception as exc:
        step("모델 로드", False, f"{type(exc).__name__}: {str(exc)[:300]}")
        traceback.print_exc()
        OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    # ---------- build_sguard_input 자체 검증 ----------
    print(f"\n{SEP}\n[1] build_sguard_input — token splice 가 template 을 보존하는가\n{SEP}")
    probe = prompts[GATE_PROMPTS[0]]["raw_prompt"]
    full = build_sguard_input(tok, probe, None)
    cut = build_sguard_input(tok, probe, 5)
    print(f"    prefix {full['n_prefix']} / content {full['n_content_pretrunc']} "
          f"/ suffix {full['n_suffix']}  = {len(full['token_ids'])} tokens")
    print(f"    cap 5 적용시 content {cut['n_content_used']}, "
          f"전체 {len(cut['token_ids'])} tokens")
    step("prefix 보존", cut["token_ids"][:full["n_prefix"]] == full["token_ids"][:full["n_prefix"]])
    step("suffix 보존", cut["token_ids"][-full["n_suffix"]:] == full["token_ids"][-full["n_suffix"]:])
    step("cap 이 content 에만 적용", cut["n_content_used"] == 5)

    # ---------- 본 판정 ----------
    print(f"\n{SEP}\n[2] response=\"\" 로 판정 — 라벨 토큰 생성 / safe·unsafe 구분\n{SEP}")
    rows = []
    for pid in GATE_PROMPTS:
        r = prompts[pid]
        res = run_one(model, tok, torch, r["raw_prompt"])
        res.update({"prompt_id": pid, "ground_truth": r["safety_label"],
                    "rarity": r["rarity_label"], "key": r["key_expression"]})
        rows.append(res)
        mark = "O" if res["decision"] == r["safety_label"] else "X"
        print(f"\n  {pid}")
        print(f"    key={r['key_expression']!r}  정답={r['safety_label']}  "
              f"판정={res['decision']}  [{mark}]   ({res['elapsed_s']}s)")
        print(f"    생성 토큰 {res['n_gen']}개: {res['gen_ids']}")
        print(f"    라벨 토큰 여부: {res['all_gen_are_label_tokens']}  "
              f"(라벨 {res['n_label_tokens']}개)")
        print(f"    카테고리: {res['categories_from_tokens']}")
        print(f"    텍스트: {res['gen_text']!r}")

    report["gate_rows"] = rows
    all_label = all(r["all_gen_are_label_tokens"] for r in rows)
    step("모델이 라벨 전용 토큰만 생성", all_label,
         "logit 기반 unsafe_score 산출 가능" if all_label
         else "글자 단위 생성 — 문자열 파싱 필요")
    five = all(len(r["categories_from_tokens"]) == 5 for r in rows)
    step("5개 카테고리 모두 출력", five)

    n_correct = sum(1 for r in rows if r["decision"] == r["ground_truth"])
    n_unsafe_pred = sum(1 for r in rows if r["decision"] == "unsafe")
    step("safe/unsafe 를 구분함 (전부 같은 판정이 아님)",
         0 < n_unsafe_pred < len(rows), f"unsafe 판정 {n_unsafe_pred}/{len(rows)}")
    print(f"\n    ground truth 일치: {n_correct}/{len(rows)}")
    report["accuracy_note"] = f"{n_correct}/{len(rows)} (표본 극소, 성능 평가 아님)"

    # ---------- response 표기 비교 ----------
    print(f"\n{SEP}\n[3] response=\"\" vs response=\"None\" — 판정이 달라지는가\n{SEP}")
    diff = []
    for pid in GATE_PROMPTS[:4]:
        raw = prompts[pid]["raw_prompt"]
        a = run_one(model, tok, torch, raw, response="")
        b = run_one(model, tok, torch, raw, response="None")
        same = a["categories_from_tokens"] == b["categories_from_tokens"]
        diff.append({"prompt_id": pid, "empty": a["categories_from_tokens"],
                     "none": b["categories_from_tokens"], "same": same})
        print(f"    {pid:38} 동일={same}  ''→{a['decision']}  'None'→{b['decision']}")
    report["response_variant"] = diff
    step("response 표기에 따라 판정이 흔들리지 않음",
         all(d["same"] for d in diff),
         "다르면 표기를 팀 전체가 통일해야 함")

    # ---------- 절단 미리보기 ----------
    print(f"\n{SEP}\n[4] (보너스) 절단이 판정을 뒤집는가 — H1 메커니즘 미리보기\n{SEP}")
    r = prompts[TRUNCATION_PROBE]
    nat = run_one(model, tok, torch, r["raw_prompt"], None)
    con = run_one(model, tok, torch, r["raw_prompt"], EXPERIMENTAL_TOKEN_CAP)
    print(f"    {TRUNCATION_PROBE}")
    print(f"    key={r['key_expression']!r}  정답={r['safety_label']}")
    print(f"    조건1 native          content {nat['n_content_used']:>3} tokens  "
          f"-> {nat['decision']}   {nat['categories_from_tokens']}")
    print(f"    조건2 constrained_77  content {con['n_content_used']:>3} tokens  "
          f"-> {con['decision']}   {con['categories_from_tokens']}")
    report["truncation_probe"] = {"prompt_id": TRUNCATION_PROBE,
                                  "ground_truth": r["safety_label"],
                                  "native": nat, "constrained": con}
    if nat["decision"] != con["decision"]:
        print("    => 절단으로 판정이 바뀜 (H1 메커니즘과 일치. n=1 이므로 결론 아님)")
    else:
        print("    => 절단해도 판정 동일 (n=1. 학생 4 의 전수 실행으로 확인 필요)")

    print(f"\n{SEP}")
    n_fail = sum(1 for v in report["steps"].values() if not v["ok"])
    print(f"  검사 {len(report['steps'])}건 중 FAIL {n_fail}건")
    print(SEP)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"  저장: {OUT_PATH.relative_to(REPO)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
