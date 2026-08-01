#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
key_span.py — KoRareGuard-T2I / Student 2

raw_prompt 안의 key_expression 이 각 컴포넌트의 토큰 공간에서 어디에 놓이고,
input policy 를 적용했을 때 얼마나 살아남는지를 계산한다.

이 모듈은 나중에 adapters 의 tokenize() 구현체가 된다.
학생 3·4 가 실제 모델에 넣는 입력과 동일한 토큰을 봐야 하므로
여기서 임의의 토크나이저를 추측해서 부르지 않는다. tokenizer 는 주입받는다.

핵심 개념 구분:
  prompt_truncated  : 프롬프트 전체가 잘렸는가
  key_visibility    : 핵심 표현 자체가 잘렸는가
  -> 이 둘은 다르다. 프롬프트가 잘려도 key 가 앞쪽이면 full 일 수 있다.

byte-level BPE(SGuard) 주의:
  한글 1 글자가 여러 token 으로 쪼개지며 같은 char offset 을 공유한다.
  절단이 그 사이를 지나가면 글자가 '반쪽' 만 남는다.
  따라서 retention 을 토큰 기준과 글자 기준 두 가지로 계산한다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

# 값 어휘는 팀 공용 SSOT(src/common/schema.py)에서 가져온다.
# 여기에 리터럴을 다시 적으면 두 곳이 조용히 어긋난다.
# (실제로 model_role 을 'safety' 로 쓰다가 schema 의 'text_safety' 와 어긋나
#  통합 join 이 0건이 되는 문제가 있었다)
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.common import schema  # noqa: E402

VISIBILITY_FULL = schema.VISIBILITY_FULL
VISIBILITY_PARTIAL = schema.VISIBILITY_PARTIAL
VISIBILITY_NONE = schema.VISIBILITY_NONE

STATUS_OK = schema.STATUS_OK
STATUS_ERROR = schema.STATUS_ERROR

# InputPolicy.model_role 에 넣을 값. 리터럴 'safety' 를 쓰면 schema 의 'text_safety'
# 와 어긋나 통합 join 이 0 행이 된다. 실제로 한 번 겪었으므로 여기서 재수출한다.
ROLE_TEXT_SAFETY = schema.ROLE_TEXT_SAFETY
ROLE_GENERATOR = schema.ROLE_GENERATOR


@dataclass(frozen=True)
class InputPolicy:
    """
    한 컴포넌트가 입력을 처리하는 방식.

    tokenizer 가 같아도 policy 가 다르면 다른 조건이다.
    (SGuard native 와 SGuard constrained_77 은 같은 tokenizer, 다른 policy)

    중요: cap 은 항상 **content 토큰 예산**이다. special token 이나 chat template 을
    포함한 전체 입력 길이가 아니다. 두 컴포넌트 모두 content 공간에서 계산해야
    위치·retention 지표를 서로 비교할 수 있다.

      SGuard      content 앞뒤에 chat template 1,480 토큰이 붙는다.
                  cap 은 template 밖의 user content 에만 적용.
      AltDiffusion tokenizer(padding='max_length', max_length=77, truncation=True) 는
                  special token 자리(<s>, </s> 2개)를 먼저 확보한 뒤 content 를 자른다.
                  따라서 content 예산은 77 이 아니라 75 다.
    """
    name: str                      # native | constrained_77 | ...
    model_id: str
    model_role: str                # safety | generator
    add_special_tokens: bool       # content 공간 계산이므로 원칙적으로 False
    cap: Optional[int] = None      # content 토큰 예산. None 이면 절단 없음
    cap_kind: str = "none"         # none | user_content | native
    declared_max_length: Optional[int] = None   # 컴포넌트가 신고하는 한계 (77 / 131072)
    special_tokens_reserved: int = 0            # declared - cap 을 설명하는 값
    note: str = ""

    @property
    def experimental_token_cap(self) -> Optional[int]:
        """연구가 임의로 정한 cap 만. 모델 native limit 은 여기 넣지 않는다."""
        return self.cap if self.cap_kind == "user_content" else None


@dataclass
class KeySpanResult:
    # 식별
    prompt_id: str = ""
    model_id: str = ""
    model_role: str = ""
    input_policy: str = ""
    experimental_token_cap: Optional[int] = None
    content_token_budget: Optional[int] = None   # 모델이 실제로 읽는 content 토큰 수
    declared_max_length: Optional[int] = None    # 컴포넌트 신고 한계
    special_tokens_reserved: int = 0
    max_length_effective: Optional[int] = None
    max_length_source: str = ""

    # 원본
    character_count: int = 0
    key_character_count: int = 0
    key_char_start: int = -1
    key_char_end: int = -1

    # PASS 1 — pre-truncation
    total_tokens_pretrunc: int = 0
    key_token_count_original: int = 0
    key_start_pretrunc: int = -1
    key_end_pretrunc: int = -1
    key_start_ratio: float = 0.0
    key_center_ratio: float = 0.0
    key_end_ratio: float = 0.0
    key_tokens_per_character: float = 0.0

    # PASS 2 — actual used input
    total_tokens_used: int = 0
    prompt_truncated: bool = False
    # 실제 사용된 토큰열에서 살아남은 key span. 전부 잘렸으면 -1.
    # (pretrunc span 과 다르다. 앞에서부터 자르므로 index 자체는 같지만
    #  '남은 것' 만 담는다는 점이 다르다)
    key_start_token: int = -1
    key_end_token: int = -1
    key_tokens_retained: int = 0
    key_retention_ratio: float = 0.0
    key_chars_retained: int = 0
    key_chars_covered: int = 0
    key_chars_uncovered: int = 0
    key_retention_ratio_char: float = 0.0
    key_visibility: str = VISIBILITY_NONE
    decoded_used_input: str = ""
    removed_text: str = ""

    # byte-level BPE 진단
    tokens_sharing_char_offset: int = 0
    key_split_mid_character: bool = False

    # 신뢰성
    analysis_status: str = STATUS_OK
    error_message: str = ""

    # 사람 검증용 (CSV 에는 넣지 않는다)
    token_rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("token_rows", None)
        return d


def find_char_span(text: str, key: str) -> tuple[int, int, Optional[str]]:
    """key 의 문자 span. 못 찾거나 여러 번 나오면 조용히 넘어가지 않고 사유를 반환한다."""
    n = text.count(key)
    if n == 0:
        return -1, -1, f"key_expression not found in raw_prompt: {key!r}"
    if n > 1:
        return -1, -1, f"key_expression occurs {n} times (ambiguous span): {key!r}"
    s = text.index(key)
    return s, s + len(key), None


def _visibility_from_ratio(ratio: float) -> str:
    if ratio >= 1.0:
        return VISIBILITY_FULL
    if ratio > 0.0:
        return VISIBILITY_PARTIAL
    return VISIBILITY_NONE


def analyze_key_span(
    tokenizer: Any,
    raw_prompt: str,
    key_expression: str,
    policy: InputPolicy,
    prompt_id: str = "",
    max_length_source: str = "",
) -> KeySpanResult:
    """
    두 단계로 나눠 계산한다.
      PASS 1 : 절단 없는 전체 시퀀스에서 key 의 토큰 span
      PASS 2 : policy.cap 을 적용한 실제 입력에서 얼마나 남았는지
    """
    res = KeySpanResult(
        prompt_id=prompt_id,
        model_id=policy.model_id,
        model_role=policy.model_role,
        input_policy=policy.name,
        experimental_token_cap=policy.experimental_token_cap,
        content_token_budget=policy.cap,
        declared_max_length=policy.declared_max_length,
        special_tokens_reserved=policy.special_tokens_reserved,
        max_length_effective=policy.cap,
        max_length_source=max_length_source or policy.cap_kind,
        character_count=len(raw_prompt),
        key_character_count=len(key_expression),
    )

    if not tokenizer.is_fast:
        res.analysis_status = STATUS_ERROR
        res.error_message = "tokenizer is not fast; offset_mapping unavailable"
        return res

    c_start, c_end, err = find_char_span(raw_prompt, key_expression)
    if err:
        res.analysis_status = STATUS_ERROR
        res.error_message = err
        return res
    res.key_char_start, res.key_char_end = c_start, c_end

    # ---------- PASS 1 ----------
    enc = tokenizer(
        raw_prompt,
        add_special_tokens=policy.add_special_tokens,
        return_offsets_mapping=True,
    )
    ids = list(enc["input_ids"])
    offs = [tuple(o) for o in enc["offset_mapping"]]
    res.total_tokens_pretrunc = len(ids)

    # offset 폭이 0 인 토큰(특수 토큰 등)은 key 와 겹칠 수 없다
    key_idx = [i for i, (a, b) in enumerate(offs)
               if a != b and a < c_end and b > c_start]

    if not key_idx:
        res.analysis_status = STATUS_ERROR
        res.error_message = "no token overlaps the key character span"
        return res

    res.key_start_pretrunc = min(key_idx)
    res.key_end_pretrunc = max(key_idx)
    res.key_token_count_original = len(key_idx)
    res.tokens_sharing_char_offset = len(key_idx) - len({offs[i] for i in key_idx})

    if res.key_start_pretrunc > res.key_end_pretrunc:
        res.analysis_status = STATUS_ERROR
        res.error_message = "key_start > key_end"
        return res

    n = res.total_tokens_pretrunc
    # end 는 시작 index 가 아니라 "마지막 key 토큰까지 포함한" 소비량이므로 +1 한다.
    # key 가 맨 끝 토큰이면 end_ratio 가 정확히 1.0 이 되어야 한다.
    # center 는 그 정규화 구간 [start_ratio, end_ratio] 의 중점으로 정의한다.
    # (팀 어댑터 analyze_content_tokens 와 같은 정의여야 한다 — 예전에 여기만
    #  inclusive index 의 중점을 써서 432행 전부가 1/(2n) 만큼 어긋났다)
    res.key_start_ratio = res.key_start_pretrunc / n
    res.key_end_ratio = (res.key_end_pretrunc + 1) / n
    res.key_center_ratio = (res.key_start_ratio + res.key_end_ratio) / 2
    if res.key_character_count:
        res.key_tokens_per_character = res.key_token_count_original / res.key_character_count

    # ---------- PASS 2 ----------
    cutoff = policy.cap if policy.cap is not None else n
    cutoff = min(cutoff, n)
    res.total_tokens_used = cutoff
    res.prompt_truncated = cutoff < n

    retained_idx = [i for i in key_idx if i < cutoff]
    res.key_tokens_retained = len(retained_idx)
    res.key_retention_ratio = len(retained_idx) / len(key_idx)
    if retained_idx:
        res.key_start_token = retained_idx[0]
        res.key_end_token = retained_idx[-1]

    # 글자 단위 retention: 그 글자를 덮는 토큰이 전부 남아야 인정한다.
    # (byte-level BPE 에서 한 글자가 여러 토큰으로 쪼개지는 경우 대비)
    #
    # 주의: 어떤 토큰도 덮지 않는 문자가 있다. SentencePiece(AltCLIP)는 '▁' 마커가
    # 선행 공백을 표현하되 offset 에는 포함시키지 않으므로, key 안의 공백이
    # 어떤 토큰 offset 에도 잡히지 않는다. 그 공백은 decode 하면 그대로 복원되므로
    # 손실이 아니라 애초에 추적 대상이 아니다. 분자·분모 양쪽에서 제외한다.
    #
    # 세 컬럼의 의미 (팀 SSOT schema.py 정의를 따른다):
    #   key_chars_covered   토큰 offset 이 닿은 key 글자 수            = 비율의 분모
    #   key_chars_retained  그중 절단되지 않고 온전히 남은 글자 수      = 비율의 분자
    #   key_chars_uncovered covered - retained. 즉 "토큰이 닿았는데 잘린" 글자 수
    #
    # uncovered 는 이름과 달리 "아무 토큰도 안 닿은 글자" 가 아니다. 그 값이 필요하면
    # key_character_count - key_chars_covered 로 언제든 구할 수 있으므로 정보 손실은
    # 없다. 팀 어댑터(analyze_content_tokens)와 정의를 하나로 맞추기 위한 합의다.
    chars_ok = 0
    chars_covered = 0
    split_mid_char = False
    for c in range(c_start, c_end):
        covering = [i for i in key_idx if offs[i][0] <= c < offs[i][1]]
        if not covering:
            continue
        chars_covered += 1
        kept = [i for i in covering if i < cutoff]
        if len(kept) == len(covering):
            chars_ok += 1
        elif kept:
            split_mid_char = True          # 일부만 남음 = 글자가 반쪽
    res.key_chars_retained = chars_ok
    res.key_chars_covered = chars_covered
    res.key_chars_uncovered = chars_covered - chars_ok
    res.key_retention_ratio_char = chars_ok / chars_covered if chars_covered else 0.0
    res.key_split_mid_character = split_mid_char

    res.key_visibility = _visibility_from_ratio(res.key_retention_ratio)

    res.decoded_used_input = tokenizer.decode(ids[:cutoff], skip_special_tokens=True)
    res.removed_text = tokenizer.decode(ids[cutoff:], skip_special_tokens=True) if res.prompt_truncated else ""

    # ---------- 사람 검증용 토큰 표 ----------
    key_set = set(key_idx)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for i, (tid, (a, b)) in enumerate(zip(ids, offs)):
        res.token_rows.append({
            "idx": i,
            "token_id": tid,
            "token": tokenizer.convert_ids_to_tokens([tid])[0],
            "offset": (a, b),
            "text": raw_prompt[a:b] if a != b else "",
            "is_special": tid in special_ids,
            "overlaps_key": i in key_set,
            "retained": i < cutoff,
        })

    _validate(res)
    return res


def _validate(res: KeySpanResult) -> None:
    """§41 검증. 실패해도 행을 버리지 않고 사유를 기록한다."""
    problems = []
    if not (0.0 <= res.key_retention_ratio <= 1.0):
        problems.append(f"key_retention_ratio out of range: {res.key_retention_ratio}")
    if not (0.0 <= res.key_retention_ratio_char <= 1.0):
        problems.append(f"key_retention_ratio_char out of range: {res.key_retention_ratio_char}")
    if res.key_visibility != _visibility_from_ratio(res.key_retention_ratio):
        problems.append("key_visibility inconsistent with key_retention_ratio")
    # 토큰이 전부 남았는데 글자가 덜 남았다면 계산이 틀린 것이다.
    # (글자 중간이 잘린 경우는 예외 — 그때는 토큰도 일부만 남는다)
    if (res.key_retention_ratio >= 1.0
            and not res.key_split_mid_character
            and res.key_retention_ratio_char < 1.0):
        problems.append(
            f"all key tokens retained but ratio_char={res.key_retention_ratio_char:.3f} < 1")
    if res.max_length_effective is not None and res.total_tokens_used > res.max_length_effective:
        problems.append(
            f"total_tokens_used {res.total_tokens_used} exceeds cap {res.max_length_effective}")
    if res.key_start_pretrunc > res.key_end_pretrunc:
        problems.append("key_start > key_end")
    if res.key_end_pretrunc >= res.total_tokens_pretrunc:
        problems.append("key span outside sequence")
    if problems:
        res.analysis_status = STATUS_ERROR
        res.error_message = "; ".join(problems)
