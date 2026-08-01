"""text_safety adapter 공용 인터페이스.

PreparedInput/SafetyResult 는 SGuardAdapter(및 향후 추가될 다른 텍스트 안전성
어댑터)가 반환하는 결과의 고정된 모양이다. TextSafetyAdapter 는 그 어댑터들이
반드시 구현해야 하는 메서드를 강제한다. 로직은 없다 — sguard.py 가 실제로
채워서 반환하는 값의 계약(contract)만 여기 있다.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..token_analysis import TokenizationResult


@dataclass
class PreparedInput:
    """prepare_input() 의 반환값. tokenize()/predict() 가 이 하나를 공유한다."""
    input_policy: str
    experimental_token_cap: int | None
    used_content_text: str
    used_content_ids: list[int]
    formatted_text: str
    formatted_input_ids: list[int]
    template_overhead_tokens: int
    total_input_tokens_estimate: int
    tokenization: TokenizationResult


@dataclass
class SafetyResult:
    """predict() 의 반환값. safety_results.csv 한 행에 대응한다.
    prompt_id/runtime_ms/error_type 은 어댑터가 모르는 값이라 호출부(harness)가 채운다."""
    prompt_id: str
    input_policy: str
    model_id: str
    categories: dict[str, str]
    decision: str
    unsafe_score: float | None
    raw_output: str
    runtime_ms: float = 0.0
    error_type: str = ""  # "" = 에러 없음 (GenerationOutput 과 동일 관례)


class TextSafetyAdapter(ABC):
    """텍스트 안전성(pre-generation moderation) 어댑터의 최소 계약."""
    model_id: str
    revision: str

    @abstractmethod
    def prepare_input(self, prompt: str, key_expression: str,
                       input_policy: str) -> PreparedInput: ...

    @abstractmethod
    def tokenize(self, prompt: str, key_expression: str,
                 input_policy: str) -> TokenizationResult: ...

    @abstractmethod
    def predict(self, prompt: str, key_expression: str,
                input_policy: str) -> SafetyResult: ...
