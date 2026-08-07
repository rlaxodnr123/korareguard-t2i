#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sguard_label_tokens.py — KoRareGuard-T2I / Student 2

SGuard 의 added token(= base vocab 을 넘는 id)을 전수 조사해서
카테고리별 safe/unsafe 라벨 토큰 표를 만든다.

왜 필요한가:
  SGuard 는 고정 5줄(Crime/Manipulation/Privacy/Sexual/Violence)을 출력하는데,
  각 줄이 vocab 에 전용 단일 토큰으로 들어가 있다. 이 id 를 알면
  학생 4 가 unsafe_score 를 문자열 파싱이 아니라 logit 비교로 산출할 수 있다.
  (같은 위치에서 safe 토큰과 unsafe 토큰의 logit 을 직접 비교)

주의:
  token id 는 revision 에 종속된다. config 의 revision 이 바뀌면 재실행할 것.

사용법:
    .venv\\Scripts\\python.exe analysis/tokenizer/sguard_label_tokens.py
"""

from __future__ import annotations

import io
import json
import re
import sys
import warnings
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import config  # noqa: E402  — 모델 id / revision 의 팀 공용 SSOT

SGUARD_MODEL_ID = config.SGUARD_MODEL_ID
SGUARD_REVISION = config.SGUARD_REVISION
CATEGORIES = ("Crime", "Manipulation", "Privacy", "Sexual", "Violence")
OUT_PATH = Path(__file__).resolve().parent / "sguard_label_tokens.json"

SEP = "=" * 86


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SGUARD_MODEL_ID, revision=SGUARD_REVISION)
    base_vocab = tok.vocab_size
    total = len(tok)

    print(SEP)
    print(f"SGuard added tokens   base vocab {base_vocab} / total {total} "
          f"-> added {total - base_vocab}")
    print(SEP)
    print(f"  {'id':>6}  {'token':<34} {'decode':<26}")
    added = []
    for tid in range(base_vocab, total):
        piece = tok.convert_ids_to_tokens([tid])[0]
        dec = tok.decode([tid])
        added.append({"id": tid, "token": piece, "decoded": dec})
        print(f"  {tid:>6}  {piece!r:<34} {dec!r:<26}")

    # ---- 카테고리 x safe/unsafe 표 구성 -------------------------------------
    print(f"\n{SEP}")
    print("카테고리별 safe / unsafe 라벨 토큰")
    print(SEP)
    table: dict[str, dict[str, int | None]] = {c.lower(): {"safe": None, "unsafe": None}
                                               for c in CATEGORIES}
    pat = re.compile(r"^(" + "|".join(CATEGORIES) + r"):\s*(safe|unsafe)\s*$")
    for a in added:
        m = pat.match(a["decoded"].strip())
        if m:
            table[m.group(1).lower()][m.group(2)] = a["id"]

    print(f"  {'category':<14} {'safe id':>9} {'unsafe id':>11}   비고")
    missing = []
    for c in CATEGORIES:
        k = c.lower()
        s, u = table[k]["safe"], table[k]["unsafe"]
        note = ""
        if s is None or u is None:
            note = "  <- 미발견"
            missing.append(k)
        print(f"  {c:<14} {str(s):>9} {str(u):>11}{note}")

    # ---- 실제 출력이 몇 토큰인가 -------------------------------------------
    print(f"\n{SEP}")
    print("실제 5줄 출력이 몇 토큰으로 인코딩되는가 (logit 위치 파악용)")
    print(SEP)
    samples = {
        "전부 safe": "\n".join(f"{c}: safe" for c in CATEGORIES),
        "예시(Sexual unsafe)": (
            "Crime: safe\nManipulation: safe\nPrivacy: safe\n"
            "Sexual: unsafe\nViolence: safe"),
        "전부 unsafe": "\n".join(f"{c}: unsafe" for c in CATEGORIES),
    }
    encodings = {}
    for name, text in samples.items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        pieces = tok.convert_ids_to_tokens(ids)
        encodings[name] = {"n_tokens": len(ids), "ids": list(ids)}
        print(f"\n  [{name}]  {len(ids)} tokens")
        for i, (tid, p) in enumerate(zip(ids, pieces)):
            flag = "  <- 라벨 토큰" if tid >= base_vocab else ""
            print(f"      {i}: {tid:>6}  {p!r}{flag}")

    # ---- config 붙여넣기용 스니펫 ------------------------------------------
    print(f"\n{SEP}")
    print("config.py 붙여넣기용")
    print(SEP)
    snippet = ["# SGuard 라벨 토큰 id (revision 종속 — revision 변경 시 재조사)",
               "# analysis/tokenizer/sguard_label_tokens.py 로 생성",
               "SGUARD_LABEL_TOKEN_IDS = {"]
    for c in CATEGORIES:
        k = c.lower()
        snippet.append(f'    "{k}": {{"safe": {table[k]["safe"]}, '
                       f'"unsafe": {table[k]["unsafe"]}}},')
    snippet.append("}")
    print("\n".join(snippet))

    payload = {
        "model_id": SGUARD_MODEL_ID,
        "base_vocab_size": base_vocab,
        "total_vocab_size": total,
        "added_tokens": added,
        "label_token_ids": table,
        "missing_categories": missing,
        "output_encodings": encodings,
        "note": "token id 는 model revision 에 종속된다. revision 변경 시 재실행할 것.",
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n저장: {OUT_PATH}")
    if missing:
        print(f"경고: safe/unsafe 짝을 못 찾은 카테고리 {missing} — "
              f"logit 비교 방식을 쓰려면 추가 확인 필요")
    return 0


if __name__ == "__main__":
    sys.exit(main())
