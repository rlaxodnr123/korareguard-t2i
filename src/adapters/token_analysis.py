"""content 토큰 공간에서의 절단·핵심표현 가시성 분석 — 단일 출처.

이 모듈이 실험 정당성의 핵심이다:
- 학생 2 의 tokenize() 기록과 학생 4 의 predict() 실제 입력이 이 함수 하나를 공유한다.
- cap 은 항상 "user content 토큰"에만 적용된다. (SGuard: template 밖 / AltDiff: special token 밖)
- tokenizer 내장 truncation(truncation=True, max_length=...)은 절대 쓰지 않는다.
  SGuard 는 truncation_side='right' 라 template suffix 가 파괴되기 때문이다.

용어:
- content tokens : special token / chat template 을 제외한, user 프롬프트 자체의 토큰열
- budget         : content tokens 중 모델이 실제로 보는 앞쪽 개수 (None = 전부)
"""
from dataclasses import dataclass, field

from ..common import schema


class KeyExpressionNotFound(Exception):
    pass


@dataclass
class TokenizationResult:
    """tokenization_results.csv 한 행에 대응 (조건 라벨 등 프롬프트 메타는 호출부가 채움)."""
    # [PASS 1] 절단 전
    total_tokens_pretrunc: int = 0
    key_token_count_original: int = 0
    key_start_pretrunc: int = -1        # content 토큰 index (inclusive)
    key_end_pretrunc: int = -1          # content 토큰 index (inclusive)
    # [PASS 2] 절단 후 실제 사용
    total_tokens_used: int = 0
    prompt_truncated: bool = False
    key_start_token: int = -1           # 사용 토큰열 내 index, 미보존 시 -1
    key_end_token: int = -1
    key_tokens_retained: int = 0
    key_retention_ratio: float = 0.0
    key_visibility: str = schema.VISIBILITY_NONE
    # [위치] — pretrunc content 토큰열 기준 상대 위치 (0=맨앞, 1=맨뒤)
    key_start_ratio: float = -1.0
    key_center_ratio: float = -1.0
    key_end_ratio: float = -1.0
    # [정규화]
    key_tokens_per_character: float = 0.0
    # [유효입력]
    decoded_used_input: str = ""
    removed_text: str = ""
    # 내부 전달용 (CSV 직접 기록 X)
    used_content_ids: list = field(default_factory=list)


def _find_key_char_span(prompt: str, key_expression: str) -> tuple[int, int]:
    start = prompt.find(key_expression)
    if start < 0:
        raise KeyExpressionNotFound(
            f"key expression {key_expression!r} not found in prompt"
        )
    return start, start + len(key_expression)


def _key_token_indices(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    """char span 과 겹치는 content 토큰 index 목록.

    토큰 i 의 char 구간 [s,e) 가 key 구간 [ks,ke) 와 한 글자라도 겹치면 key 토큰.
    (SentencePiece 가 key 경계와 어긋나게 분절해도 보수적으로 포함)
    """
    ks, ke = span
    return [i for i, (s, e) in enumerate(offsets) if s < ke and e > ks and e > s]


def analyze_content_tokens(
    prompt: str,
    key_expression: str,
    content_ids: list[int],
    offsets: list[tuple[int, int]],
    budget: int | None,
    decode_fn,
) -> TokenizationResult:
    """content 토큰열에 budget 을 적용했을 때의 핵심표현 가시성을 계산한다.

    Args:
        content_ids: special/template 제외 content 토큰 id 열
        offsets:     각 content 토큰의 (char_start, char_end), content_ids 와 동일 길이
        budget:      모델이 보는 content 토큰 수. None → 전부 (native)
        decode_fn:   list[int] -> str
    """
    assert len(content_ids) == len(offsets), "ids/offsets 길이 불일치"
    r = TokenizationResult()

    # ---------------- PASS 1: 절단 전
    n = len(content_ids)
    r.total_tokens_pretrunc = n
    span = _find_key_char_span(prompt, key_expression)
    key_idx = _key_token_indices(offsets, span)
    if not key_idx:
        raise KeyExpressionNotFound(
            f"key expression {key_expression!r} maps to zero tokens"
        )
    r.key_token_count_original = len(key_idx)
    r.key_start_pretrunc = key_idx[0]
    r.key_end_pretrunc = key_idx[-1]
    if n > 1:
        r.key_start_ratio = key_idx[0] / (n - 1)
        r.key_end_ratio = key_idx[-1] / (n - 1)
        r.key_center_ratio = (r.key_start_ratio + r.key_end_ratio) / 2
    else:
        r.key_start_ratio = r.key_center_ratio = r.key_end_ratio = 0.0
    r.key_tokens_per_character = len(key_idx) / max(len(key_expression), 1)

    # ---------------- PASS 2: budget 적용 (앞쪽 budget 개만 사용)
    if budget is None or budget >= n:
        used_n = n
        r.prompt_truncated = False
    else:
        used_n = budget
        r.prompt_truncated = True
    r.total_tokens_used = used_n
    r.used_content_ids = list(content_ids[:used_n])
    r.decoded_used_input = decode_fn(content_ids[:used_n])
    r.removed_text = decode_fn(content_ids[used_n:]) if used_n < n else ""

    retained = [i for i in key_idx if i < used_n]
    r.key_tokens_retained = len(retained)
    r.key_retention_ratio = len(retained) / len(key_idx)
    r.key_visibility = schema.visibility_from_ratio(r.key_retention_ratio)
    if retained:
        r.key_start_token = retained[0]
        r.key_end_token = retained[-1]
    # 주의: prompt_truncated 와 key_visibility 는 별개다.
    # 프롬프트가 잘렸어도(back 이 아닌 key 라면) key 는 full 일 수 있다.
    return r
