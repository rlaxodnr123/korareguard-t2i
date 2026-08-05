#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
token_chunk_checker.py — KoRareGuard-T2I / Student 5 (방어 2)

긴 프롬프트를 안전 필터 tokenizer 의 **토큰 수** 기준으로 겹치는 창(window)으로
나눈다. 문자 수 기준이 아니다 — 한국어는 글자당 토큰 수가 표현마다 크게 달라
(희귀 표현 기준 최대 3배) 문자 기준 분할은 필터가 실제로 보는 양과 어긋난다.

================================================================================
왜 자르는가 — 절단 복구가 아니라 희석 해소
================================================================================
설계 문서의 원래 동기는 "예산을 넘겨 잘린 꼬리를 되살린다"였다. 실측은 그 동기를
지지하지 않는다. native(절단 0건) ↔ constrained_77 사이에서 판정이 뒤집힌
프롬프트는 432개 중 7개뿐이고, 절단 방향은 4개다.

실제로 관측되는 것은 **길이 자체**의 효과다. 절단이 구조적으로 불가능한 native
조건에서도 under-blocking 이 short 69.4% → near_limit 91.7% → over_limit 100% 로
오른다. unsafe_score 중앙값(일반 표현)으로 보면 원인이 더 분명하다.

    구간          안전      위험      분리비
    short        0.0025   0.4848    194배
    near_limit   0.0023   0.0303     13배
    over_limit   0.0083   0.0110    1.3배

안전 프롬프트 점수는 길이와 무관하게 바닥에 고정돼 있고(더 내려갈 데가 없다),
위험 프롬프트만 그 바닥으로 떨어진다. 즉 희석은 점수를 낮추는 게 아니라
**안전/위험 분리도 자체를 파괴한다.** 임계값 조정으로는 복구할 수 없다.

chunk 는 긴 프롬프트를 분리도가 살아 있는 짧은 구간으로 되돌리는 조작이다.

================================================================================
구현 규칙 (실험 정당성 직결)
================================================================================
1. **token id 수준에서 조립한다.** decode 한 문자열을 다시 토큰화해 모델에 먹이지
   않는다 (sguard.py #3 과 같은 이유 — U+FFFD 손상, 공백 병합 유실).
   각 chunk 의 입력은 `prefix_ids + content_ids[start:end] + suffix_ids` 다.
2. **chat template 은 매 chunk 마다 온전히 붙인다.** SGuard 는 template overhead 가
   약 1,480 토큰이고 판정 형식을 그 안에서 지시받는다. content 만 잘라 넣으면
   모델이 판정을 내지 않는다.
3. **tokenizer 내장 truncation 을 쓰지 않는다.** truncation_side='right' 라
   template suffix 가 파괴된다.
4. chunk 크기는 **모델 제약이 아니라 방어 하이퍼파라미터**다. SGuard native
   context 는 131,072 라 자를 이유가 없다. 77 은 본 연구가 정의한 실험 cap 이자
   AltDiffusion 의 native 예산이라 채택했을 뿐이며, Phase 4 에서 ablation 축으로
   따로 스윕한다 (short 구간 중앙값이 24 토큰이므로 더 작은 값이 유리할 수 있다).

사용법:
    python defense/token_chunk_checker.py --selftest
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.adapters.token_analysis import analyze_content_tokens  # noqa: E402

# ---------------------------------------------------------------- 기본값
# PHASE1_GATE.md 에 사전 등록된 값. 게이트 실행 후 변경 금지.
DEFAULT_BUDGET = 77
DEFAULT_STRIDE = 62          # overlap = 77 - 62 = 15


@dataclass
class Chunk:
    """content 토큰 공간의 창 하나. 모델 입력은 input_ids 를 그대로 쓴다."""
    index: int
    start: int                              # content 토큰 index (inclusive)
    end: int                                # content 토큰 index (exclusive)
    content_ids: list[int] = field(default_factory=list)
    input_ids: list[int] = field(default_factory=list)
    text: str = ""                          # decode 결과 — 로그/표시 전용
    contains_key: bool = False              # 핵심표현 토큰을 하나라도 포함하는가
    key_tokens_in_chunk: int = 0

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


def chunk_spans(n_tokens: int, budget: int = DEFAULT_BUDGET,
                stride: int = DEFAULT_STRIDE) -> list[tuple[int, int]]:
    """content 토큰열을 겹치는 [start, end) 구간으로 나눈다. 모델 불필요 — 순수 함수.

    규칙:
      - n <= budget  → 창 하나 [0, n). 짧은 프롬프트는 원본과 동일해진다.
      - 그 외        → start = 0, stride, 2*stride, ... 로 창을 만들고,
                       마지막 창은 항상 [max(0, n-budget), n) 로 맞춘다.

    마지막 창을 끝에 붙여 정렬하는 이유: 그냥 stride 를 이어가면 꼬리에
    몇 토큰짜리 창이 남는다. 짧은 창은 문맥이 없어 점수가 요동치고, max 집계에서
    잡음만 키운다. 끝에서 budget 만큼 되짚어 잡으면 모든 창이 같은 크기가 된다.

    반환 구간들은 시작 index 오름차순이며 중복이 없다. 전체 [0, n) 를 빠짐없이 덮는다.
    """
    if budget <= 0:
        raise ValueError(f"budget 은 양수여야 한다: {budget}")
    if stride <= 0:
        raise ValueError(f"stride 는 양수여야 한다: {stride}")
    if stride >= budget:
        # stride == budget 이면 겹침이 0 이라, 경계에 걸친 표현이 양쪽 창 모두에서
        # 쪼개져 사라진다. 겹침을 두는 것이 이 방어의 핵심이므로 실행을 막는다.
        raise ValueError(
            f"stride({stride}) 는 budget({budget}) 보다 작아야 한다 (겹침 필요)")
    if n_tokens <= 0:
        return []
    if n_tokens <= budget:
        return [(0, n_tokens)]

    spans: list[tuple[int, int]] = []
    start = 0
    while start + budget < n_tokens:
        spans.append((start, start + budget))
        start += stride
    spans.append((max(0, n_tokens - budget), n_tokens))

    # 마지막 창을 끝에 정렬하면서 직전 창과 같아질 수 있다 (n 이 딱 맞아떨어질 때).
    deduped: list[tuple[int, int]] = []
    for s in spans:
        if not deduped or s != deduped[-1]:
            deduped.append(s)
    return deduped


def build_chunks(adapter, prompt: str, key_expression: str,
                 budget: int = DEFAULT_BUDGET,
                 stride: int = DEFAULT_STRIDE) -> list[Chunk]:
    """프롬프트를 chunk 로 나누고 각 chunk 의 모델 입력(input_ids)까지 조립한다.

    Args:
        adapter: SGuardAdapter. tok.encode_chat_template / tok.decode 를 쓴다.
        key_expression: 핵심표현. 어느 chunk 가 그것을 담고 있는지 표시하는 데만
                        쓰이며 분할 자체에는 영향을 주지 않는다 (분할이 라벨을
                        보고 달라지면 통제 실험이 아니게 된다).
    """
    enc = adapter.tok.encode_chat_template(prompt_text=prompt, response_text="")
    n = len(enc.content_ids)

    # 핵심표현이 pretrunc content 토큰열의 어디에 있는지 (budget=None → 절단 없음).
    tr = analyze_content_tokens(
        prompt=prompt, key_expression=key_expression,
        content_ids=enc.content_ids, offsets=enc.content_offsets,
        budget=None, decode_fn=adapter.tok.decode,
    )
    key_lo, key_hi = tr.key_start_pretrunc, tr.key_end_pretrunc  # 둘 다 inclusive

    chunks: list[Chunk] = []
    for i, (s, e) in enumerate(chunk_spans(n, budget, stride)):
        window = list(enc.content_ids[s:e])
        overlap = max(0, min(e - 1, key_hi) - max(s, key_lo) + 1)
        chunks.append(Chunk(
            index=i, start=s, end=e,
            content_ids=window,
            input_ids=enc.prefix_ids + window + enc.suffix_ids,
            text=adapter.tok.decode(window),
            contains_key=overlap > 0,
            key_tokens_in_chunk=overlap,
        ))
    return chunks


# ================================================================ self-test
def _selftest() -> int:
    PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
    results: list[bool] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append(bool(cond))
        print(f"  [{PASS if cond else FAIL}] {name}" + (f"  — {detail}" if detail else ""))

    print("=" * 78)
    print("  chunk_spans 경계 로직 (모델 불필요)")
    print("=" * 78)

    # --- 짧은 입력은 창 하나, 원본과 동일
    check("n=0 → 빈 목록", chunk_spans(0) == [])
    check("n < budget → 창 하나", chunk_spans(24) == [(0, 24)], str(chunk_spans(24)))
    check("n == budget → 창 하나", chunk_spans(77) == [(0, 77)], str(chunk_spans(77)))
    check("n = budget+1 → 창 둘", len(chunk_spans(78)) == 2, str(chunk_spans(78)))

    # --- over_limit 대표값 (SGuard content 토큰 중앙값 422)
    sp = chunk_spans(422)
    check("n=422 → 7창", len(sp) == 7, str(sp))
    check("모든 창이 budget 크기", all(e - s == 77 for s, e in sp))
    check("마지막 창이 끝에 정렬", sp[-1][1] == 422, str(sp[-1]))
    check("겹침 15 유지 (마지막 제외)",
          all(sp[i + 1][0] - sp[i][0] == 62 for i in range(len(sp) - 2)))

    # --- 불변식: 전 구간 커버, 정렬, 중복 없음
    for n in [1, 5, 76, 77, 78, 100, 122, 140, 200, 422, 446, 1000]:
        sp = chunk_spans(n)
        covered = set()
        for s, e in sp:
            covered.update(range(s, e))
        check(f"n={n}: 전 구간 커버", covered == set(range(n)),
              f"{len(covered)}/{n}")
        check(f"n={n}: 시작 오름차순·중복 없음",
              all(sp[i][0] < sp[i + 1][0] for i in range(len(sp) - 1)))
        check(f"n={n}: 모든 창 ≤ budget", all(e - s <= 77 for s, e in sp))

    # --- 잘못된 설정은 실행을 막는다
    for bad, why in [((77, 77), "stride == budget (겹침 0)"),
                     ((77, 100), "stride > budget (구멍 발생)"),
                     ((0, 62), "budget = 0"),
                     ((77, 0), "stride = 0")]:
        try:
            chunk_spans(500, bad[0], bad[1])
            check(f"거부: {why}", False, "예외가 안 났다")
        except ValueError:
            check(f"거부: {why}", True)

    # --- 겹침이 실제로 경계 표현을 살리는가 (핵심 동기의 검증)
    # 창 경계에 걸친 5토큰짜리 표현이 적어도 한 창에는 통째로 들어가야 한다.
    sp = chunk_spans(422)
    worst = None
    for key_lo in range(0, 422 - 5):
        key_hi = key_lo + 4
        whole = any(s <= key_lo and key_hi < e for s, e in sp)
        if not whole:
            worst = (key_lo, key_hi)
            break
    check("5토큰 표현이 어느 위치에 있든 온전히 담기는 창이 존재",
          worst is None, f"실패 위치 {worst}" if worst else "422 위치 전수 확인")

    n_pass = sum(results)
    print("=" * 78)
    print(f"  {n_pass}/{len(results)} CHECK 통과")
    print("=" * 78)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
    print("\n실행: python defense/token_chunk_checker.py --selftest")
