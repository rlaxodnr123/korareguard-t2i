"""SGuard-ContentFilter-2B-v1 adapter.

실험 정당성에 직결되는 구현 규칙:
1. cap 은 user content 토큰에만 적용한다. tokenizer 의 truncation=True, max_length 는
   절대 쓰지 않는다 (truncation_side='right' → template suffix 파괴, 설계 문서 섹션 3).
2. chat template 의 message 키는 `prompt` / `response` 다 (`content` 아님).
3. pre-generation moderation 이므로 response="" 로 넣는다.
   빈 response 동작은 PILOT GATE 0a 에서 경험적으로 검증한다 (Limitation 5).
4. 출력은 고정 5줄 (Crime/Manipulation/Privacy/Sexual/Violence) 이며 단일 카테고리가 아니다.
   safe = 5개 전부 safe.
"""
from typing import Protocol

from ...common import config, schema
from ..token_analysis import TokenizationResult, analyze_content_tokens
from .base import PreparedInput, SafetyResult, TextSafetyAdapter


class SGuardTokenizerBackend(Protocol):
    """실제 HF tokenizer 또는 시뮬레이션 mock 이 구현해야 하는 최소 인터페이스."""
    tokenizer_class: str
    revision: str
    padding_side: str
    truncation_side: str

    def encode_content(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """add_special_tokens=False. (ids, char offsets) 반환."""
        ...
    def decode(self, ids: list[int]) -> str: ...
    def apply_chat_template(self, prompt_text: str, response_text: str) -> str:
        """키는 반드시 config.SGUARD_MESSAGE_KEY_PROMPT / _RESPONSE."""
        ...
    def count_tokens(self, formatted_text: str) -> int: ...


class SGuardModelBackend(Protocol):
    def generate(self, formatted_text: str) -> str:
        """고정 5줄 raw output 반환."""
        ...
    def label_logits(self, formatted_text: str) -> dict[str, float] | None:
        """카테고리별 unsafe 라벨 토큰 logit. 미구현 시 None (unsafe_score 산출: 김태욱)."""
        ...


_POLICY_TO_BUDGET: dict[str, int | None] = {
    schema.POLICY_NATIVE: None,
    schema.POLICY_CONSTRAINED_77: config.CAP_CONSTRAINED_77,
    schema.POLICY_CONSTRAINED_127: config.CAP_CONSTRAINED_127,
}


class SGuardAdapter(TextSafetyAdapter):
    model_id = config.SGUARD_MODEL_ID
    revision = config.SGUARD_REVISION

    def __init__(self, tokenizer: SGuardTokenizerBackend,
                 model: SGuardModelBackend | None = None):
        # 실측 사실과 다르면 실행 자체를 막는다 (revision 이 바뀐 것)
        if tokenizer.truncation_side != config.SGUARD_TRUNCATION_SIDE:
            raise RuntimeError(
                f"SGuard truncation_side 실측값({config.SGUARD_TRUNCATION_SIDE}) 과 "
                f"불일치: {tokenizer.truncation_side}. revision 확인 필요."
            )
        self.tok = tokenizer
        self.model = model

    # ------------------------------------------------------------ 공유 경로
    def prepare_input(self, prompt: str, key_expression: str,
                      input_policy: str) -> PreparedInput:
        if input_policy not in _POLICY_TO_BUDGET:
            raise ValueError(f"unknown input_policy: {input_policy}")
        budget = _POLICY_TO_BUDGET[input_policy]

        content_ids, offsets = self.tok.encode_content(prompt)
        tr: TokenizationResult = analyze_content_tokens(
            prompt=prompt, key_expression=key_expression,
            content_ids=content_ids, offsets=offsets,
            budget=budget, decode_fn=self.tok.decode,
        )

        # 절단된 content 를 template 에 넣는다. response 는 비움 (pre-generation).
        formatted = self.tok.apply_chat_template(
            prompt_text=tr.decoded_used_input, response_text="",
        )
        # 방어적 검증: cap 을 전체 입력에 잘못 적용했다면 formatted 가 overhead 보다
        # 짧아지는 파국이 생긴다. template 이 보존됐는지 하한으로 확인.
        formatted_tokens = self.tok.count_tokens(formatted)
        if formatted_tokens < config.SGUARD_TEMPLATE_OVERHEAD_TOKENS:
            raise RuntimeError(
                "formatted input 이 template overhead(1,480)보다 짧음 — "
                "template 파괴 절단이 의심됨. tokenizer truncation 사용 여부 점검."
            )

        return PreparedInput(
            input_policy=input_policy,
            experimental_token_cap=budget,
            used_content_text=tr.decoded_used_input,
            used_content_ids=tr.used_content_ids,
            formatted_text=formatted,
            template_overhead_tokens=config.SGUARD_TEMPLATE_OVERHEAD_TOKENS,
            total_input_tokens_estimate=(
                config.SGUARD_TEMPLATE_OVERHEAD_TOKENS + tr.total_tokens_used
            ),
            tokenization=tr,
        )

    # ------------------------------------------------------------ 학생 2
    def tokenize(self, prompt: str, key_expression: str,
                 input_policy: str) -> TokenizationResult:
        return self.prepare_input(prompt, key_expression, input_policy).tokenization

    def formatted_pretrunc_tokens(self, prompt: str) -> int:
        """PASS1 보조 필드: 전체 프롬프트 기준 template 포함 토큰 수."""
        formatted = self.tok.apply_chat_template(prompt_text=prompt, response_text="")
        return self.tok.count_tokens(formatted)

    # ------------------------------------------------------------ 학생 4
    def predict(self, prompt: str, key_expression: str,
                input_policy: str) -> SafetyResult:
        if self.model is None:
            raise RuntimeError("model backend 미주입 — predict 불가")
        prep = self.prepare_input(prompt, key_expression, input_policy)
        raw = self.model.generate(prep.formatted_text)
        categories = parse_sguard_output(raw)
        decision = (schema.UNSAFE
                    if any(v == schema.UNSAFE for v in categories.values())
                    else schema.SAFE)
        logits = self.model.label_logits(prep.formatted_text)
        unsafe_score = max(logits.values()) if logits else None
        return SafetyResult(
            prompt_id="",  # 호출부(run_text_safety)가 채움
            input_policy=input_policy,
            model_id=self.model_id,
            categories=categories,
            decision=decision,
            unsafe_score=unsafe_score,
            raw_output=raw,
        )


def parse_sguard_output(raw: str) -> dict[str, str]:
    """고정 5줄 출력 파싱. 5줄·5카테고리·값 {safe,unsafe} 가 아니면 실패.

    'Safe' 는 SGuard 카테고리가 아니다. 단일 카테고리 출력 가정 금지.
    """
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    parsed: dict[str, str] = {}
    for ln in lines:
        if ":" not in ln:
            raise ValueError(f"unparseable SGuard line: {ln!r}")
        name, _, value = ln.partition(":")
        name, value = name.strip().lower(), value.strip().lower()
        if name not in config.SGUARD_CATEGORIES:
            raise ValueError(f"unknown SGuard category: {name!r}")
        if value not in (schema.SAFE, schema.UNSAFE):
            raise ValueError(f"invalid SGuard value for {name}: {value!r}")
        if name in parsed:
            raise ValueError(f"duplicate SGuard category: {name}")
        parsed[name] = value
    missing = set(config.SGUARD_CATEGORIES) - set(parsed)
    if missing:
        raise ValueError(f"missing SGuard categories: {sorted(missing)}")
    return parsed


# ---------------------------------------------------------------- 실모델 로더 (lazy)
def load_real_sguard_adapter(device: str = "cuda") -> SGuardAdapter:
    """실제 HF 모델용 로더. transformers 는 여기서만 lazy import.

    PILOT GATE 0a 에서 이 로더로 빈-response 동작을 먼저 검증한 뒤 대량 실행할 것.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

    hf_tok = AutoTokenizer.from_pretrained(
        config.SGUARD_MODEL_ID, revision=config.SGUARD_REVISION)
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.SGUARD_MODEL_ID, revision=config.SGUARD_REVISION).to(device)

    class _RealTok:
        tokenizer_class = type(hf_tok).__name__
        revision = config.SGUARD_REVISION
        padding_side = hf_tok.padding_side
        truncation_side = hf_tok.truncation_side

        def encode_content(self, text):
            enc = hf_tok(text, add_special_tokens=False, return_offsets_mapping=True)
            return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

        def decode(self, ids):
            return hf_tok.decode(ids)

        def apply_chat_template(self, prompt_text, response_text):
            msgs = [{config.SGUARD_MESSAGE_KEY_PROMPT: prompt_text,
                     config.SGUARD_MESSAGE_KEY_RESPONSE: response_text}]
            return hf_tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)

        def count_tokens(self, formatted_text):
            return len(hf_tok(formatted_text, add_special_tokens=False)["input_ids"])

    class _RealModel:
        def generate(self, formatted_text):
            import torch
            inputs = hf_tok(formatted_text, return_tensors="pt",
                            add_special_tokens=False).to(device)
            with torch.no_grad():
                out = hf_model.generate(**inputs, max_new_tokens=32, do_sample=False)
            return hf_tok.decode(out[0, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)

        def label_logits(self, formatted_text):
            return None  # unsafe_score 산출 방법: 김태욱 확정 후 구현 (#6)

    return SGuardAdapter(_RealTok(), _RealModel())
