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
                 pipeline: AltDiffPipelineBackend | None = None,
                 input_ids_dump_path: str | None = None,
                 input_ids_dump_n: int = 10):
        self.tok = tokenizer
        self.pipe = pipeline
        # 학생3(#6b): "생성기가 이 프롬프트를 어디까지 읽었는지" 실제로 tokenizer 에
        # 들어간 input_ids 를 첫 N개만 파일로 남긴다. 그냥 generate() 를 돌리기만 하면 됨.
        self._dump_path = input_ids_dump_path
        self._dump_n = input_ids_dump_n
        self._dump_written = 0

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
        if self._dump_path and self._dump_written < self._dump_n:
            self._dump_input_ids(prompt)
        out = GenerationOutput(prompt_id="", generator_id=self.model_id, seed=seed)
        try:
            self.pipe.generate_image(prompt, seed, out_path)
            out.image_path = out_path
        except Exception as e:  # noqa: BLE001 — 실패도 행으로 기록해야 함
            out.error_type = type(e).__name__
        return out

    def _dump_input_ids(self, prompt: str) -> None:
        """실제 pipe.tokenizer 가 본 input_ids 를 JSONL 로 append (학생2 분석 대조용)."""
        debug_fn = getattr(self.pipe, "debug_input_ids", None)
        if debug_fn is None:
            return  # mock 등 디버그 미지원 backend 는 조용히 skip
        ids = debug_fn(prompt)
        import json
        import os
        os.makedirs(os.path.dirname(os.path.abspath(self._dump_path)) or ".", exist_ok=True)
        with open(self._dump_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "index": self._dump_written,
                "prompt_preview": prompt[:80],
                "input_ids": ids,
            }, ensure_ascii=False) + "\n")
        self._dump_written += 1

    # tokenization_results.csv 메타 필드용
    def policy_meta(self) -> dict:
        return {
            schema.TokCols.INPUT_POLICY: schema.POLICY_NATIVE,
            schema.TokCols.EXPERIMENTAL_TOKEN_CAP: None,
            schema.TokCols.MAX_LENGTH_EFFECTIVE: config.ALTDIFF_MODEL_MAX_LENGTH,
            schema.TokCols.MAX_LENGTH_SOURCE: "tokenizer_config",
        }


def load_real_altdiffusion_adapter(
    device: str | None = None,
    input_ids_dump_path: str | None = "input_ids_sample.jsonl",
    input_ids_dump_n: int = 10,
) -> AltDiffusionAdapter:
    """실제 diffusers pipeline 로더 (lazy import).

    device 미지정 시 CUDA 가능 여부로 자동 결정 (GPU 없는 로컬에서도 로드 자체는
    가능해야 함 — sguard.py 의 동일 이슈와 같은 이유).

    m18 의 model_index.json 은 diffusers 0.8.0.dev0 시절 파일이라 text_encoder 를
    "alt_diffusion" 모듈로 지정하는데, 이후 diffusers 가 그 모듈을
    pipelines/alt_diffusion → pipelines/deprecated/alt_diffusion 로 옮기면서
    AltDiffusionPipeline.from_pretrained(model_id) 의 경로 해석이 ValueError 로 실패한다
    (m18 은 2022년 이후 업데이트가 없어 그대로임 — m9 도 동일 시기 파일이라 폴백도
    같은 이유로 실패할 가능성이 높다, Limitation 4).

    component 를 직접 조립해 model_index.json 해석을 건너뛴다.
    실측 검증 완료(#1): 파이프라인 조립·encode_prompt·768x768 실제 이미지 생성 모두 OK,
    text_encoder 의 roberta.pooler 가중치 미존재 경고는 무해함 확인.
    → m18 그대로 사용 가능. m9 폴백은 불필요.

    PILOT GATE 0b: 그럼에도 이 조립이 실패하면 m9 폴백을 결정하고
    config.ALTDIFF_FALLBACK_* 를 pin 한 뒤 이 함수의 model_id 를 교체한다
    (그때도 from_pretrained 가 아니라 이 방식의 component 조립을 먼저 시도할 것).
    """
    import torch  # lazy
    from diffusers import (  # lazy
        AltDiffusionPipeline, AutoencoderKL, PNDMScheduler, UNet2DConditionModel,
    )
    from diffusers.pipelines.deprecated.alt_diffusion.modeling_roberta_series import (
        RobertaSeriesModelWithTransformation,
    )
    from transformers import XLMRobertaTokenizer

    import os
    if input_ids_dump_path and os.path.exists(input_ids_dump_path):
        try:
            os.remove(input_ids_dump_path)
        except Exception:
            pass

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"


    mid, rev = config.ALTDIFF_MODEL_ID, config.ALTDIFF_REVISION
    pipe = AltDiffusionPipeline(
        vae=AutoencoderKL.from_pretrained(mid, subfolder="vae", revision=rev),
        text_encoder=RobertaSeriesModelWithTransformation.from_pretrained(
            mid, subfolder="text_encoder", revision=rev),
        tokenizer=XLMRobertaTokenizer.from_pretrained(
            mid, subfolder="tokenizer", revision=rev),
        unet=UNet2DConditionModel.from_pretrained(mid, subfolder="unet", revision=rev),
        scheduler=PNDMScheduler.from_pretrained(mid, subfolder="scheduler", revision=rev),
        safety_checker=None,  # 내장 필터 OFF (설계 문서 섹션 1)
        feature_extractor=None,
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

        def debug_input_ids(self, prompt):
            # diffusers 가 실제로 호출하는 방식과 동일하게 재현 (#6b).
            enc = pipe.tokenizer(prompt, padding="max_length",
                                 max_length=config.ALTDIFF_MODEL_MAX_LENGTH,
                                 truncation=True)
            return list(enc["input_ids"])

    return AltDiffusionAdapter(_RealTok(), _RealPipe(),
                               input_ids_dump_path=input_ids_dump_path,
                               input_ids_dump_n=input_ids_dump_n)
