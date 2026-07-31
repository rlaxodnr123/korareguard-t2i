"""BAAI/AltDiffusion-m18 adapter.

실측 사실:
- tokenizer: XLM-R SentencePiece, model_max_length=77 (special 2개 포함 → content 예산 75)
- text_encoder max_position 514 → 77 은 아키텍처 한계가 아니라 설정값
- 내장 safety_checker 는 OFF (이미지 안전 필터 미사용, 사람 평가로 대체)

policy 는 native 고정: content 예산 75 는 experimental cap 이 아니라 tokenizer 설정에서
유도된 native limit 이다. (schema 의 max_length_source='tokenizer_config')
"""
from typing import Protocol

from ...common import config, schema
from ..token_analysis import TokenizationResult, analyze_content_tokens
from .base import GenerationOutput, GeneratorAdapter


class AltDiffTokenizerBackend(Protocol):
    tokenizer_class: str
    revision: str
    padding_side: str
    truncation_side: str

    def encode_content(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """add_special_tokens=False."""
        ...
    def decode(self, ids: list[int]) -> str: ...


class AltDiffPipelineBackend(Protocol):
    def generate_image(self, prompt: str, seed: int, out_path: str) -> None: ...


class AltDiffusionAdapter(GeneratorAdapter):
    model_id = config.ALTDIFF_MODEL_ID
    revision = config.ALTDIFF_REVISION
    content_budget = config.ALTDIFF_CONTENT_BUDGET  # 75 = 77 - special 2

    def __init__(self, tokenizer: AltDiffTokenizerBackend,
                 pipeline: AltDiffPipelineBackend | None = None):
        self.tok = tokenizer
        self.pipe = pipeline

    def tokenize(self, prompt: str, key_expression: str) -> TokenizationResult:
        content_ids, offsets = self.tok.encode_content(prompt)
        return analyze_content_tokens(
            prompt=prompt, key_expression=key_expression,
            content_ids=content_ids, offsets=offsets,
            budget=self.content_budget, decode_fn=self.tok.decode,
        )

    def generate(self, prompt: str, seed: int, out_path: str) -> GenerationOutput:
        if self.pipe is None:
            raise RuntimeError("pipeline backend 미주입 — generate 불가")
        out = GenerationOutput(prompt_id="", generator_id=self.model_id, seed=seed)
        try:
            self.pipe.generate_image(prompt, seed, out_path)
            out.image_path = out_path
        except Exception as e:  # noqa: BLE001 — 실패도 행으로 기록해야 함
            out.error_type = type(e).__name__
        return out

    # tokenization_results.csv 메타 필드용
    def policy_meta(self) -> dict:
        return {
            schema.TokCols.INPUT_POLICY: schema.POLICY_NATIVE,
            schema.TokCols.EXPERIMENTAL_TOKEN_CAP: None,
            schema.TokCols.MAX_LENGTH_EFFECTIVE: config.ALTDIFF_MODEL_MAX_LENGTH,
            schema.TokCols.MAX_LENGTH_SOURCE: "tokenizer_config",
        }


def load_real_altdiffusion_adapter(device: str = "cuda") -> AltDiffusionAdapter:
    """실제 diffusers pipeline 로더 (lazy import).

    PILOT GATE 0b: full pipeline 로드가 여기서 실패하면 m9 폴백을 결정하고
    config.ALTDIFF_FALLBACK_* 를 pin 한 뒤 이 함수의 model_id 를 교체한다.
    """
    from diffusers import AltDiffusionPipeline  # lazy
    from transformers import XLMRobertaTokenizer  # noqa: F401 — 로드 확인용

    pipe = AltDiffusionPipeline.from_pretrained(
        config.ALTDIFF_MODEL_ID, revision=config.ALTDIFF_REVISION,
        safety_checker=None,  # 내장 필터 OFF (설계 문서 섹션 1)
        requires_safety_checker=False,
    ).to(device)
    hf_tok = pipe.tokenizer

    class _RealTok:
        tokenizer_class = type(hf_tok).__name__
        revision = config.ALTDIFF_REVISION
        padding_side = hf_tok.padding_side
        truncation_side = hf_tok.truncation_side

        def encode_content(self, text):
            enc = hf_tok(text, add_special_tokens=False, return_offsets_mapping=True)
            return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

        def decode(self, ids):
            return hf_tok.decode(ids)

    class _RealPipe:
        def generate_image(self, prompt, seed, out_path):
            import torch
            g = torch.Generator(device=device).manual_seed(seed)
            image = pipe(prompt, generator=g).images[0]
            image.save(out_path)

    return AltDiffusionAdapter(_RealTok(), _RealPipe())
