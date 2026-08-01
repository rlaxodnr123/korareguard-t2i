"""generator adapter 공용 인터페이스.

GenerationOutput 은 generate() 가 반환하는 결과의 고정된 모양이고,
GeneratorAdapter 는 이미지 생성 어댑터(AltDiffusionAdapter 등)가 반드시
구현해야 하는 메서드를 강제한다.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..token_analysis import TokenizationResult


@dataclass
class GenerationOutput:
    """generate() 의 반환값. generation_results.csv 한 행에 대응한다.
    image_path/error_type 은 성공/실패에 따라 호출부에서 사후 대입한다."""
    prompt_id: str
    generator_id: str
    seed: int
    image_path: str = ""
    error_type: str = ""  # "" = 에러 없음 (None 아님 — CSV/비교 시 빈 문자열 관례를 따름)
    runtime_ms: float = 0.0


class GeneratorAdapter(ABC):
    """이미지 생성 어댑터의 최소 계약."""
    model_id: str
    revision: str
    content_budget: int

    @abstractmethod
    def tokenize(self, prompt: str, key_expression: str) -> TokenizationResult: ...

    @abstractmethod
    def generate(self, prompt: str, seed: int, out_path: str) -> GenerationOutput: ...
