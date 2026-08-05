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
from dataclasses import dataclass
from typing import Protocol

from ...common import config, schema
from ..token_analysis import TokenizationResult, analyze_content_tokens
from .base import PreparedInput, SafetyResult, TextSafetyAdapter


@dataclass
class ChatTemplateEncoding:
    """prompt_text(미절단) 전체를 template 에 넣어 '한 번만' 토큰화한 결과.

    decode 를 거치지 않고 offset 으로 prefix/content/suffix 경계를 나눈 것이므로,
    U+FFFD 손상이나 'Prompt: ' 뒤 공백 병합 같은 경계 토큰화 동작 유실이 없다 (#3).
    content_offsets 는 content_ids 와 길이가 같고, prompt(원문 content_text) 기준
    char offset 이다 — token_analysis.analyze_content_tokens 에 그대로 넘길 수 있다.
    """
    prefix_ids: list[int]
    content_ids: list[int]
    content_offsets: list[tuple[int, int]]
    suffix_ids: list[int]


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
        """키는 반드시 config.SGUARD_MESSAGE_KEY_PROMPT / _RESPONSE.
        formatted_pretrunc_tokens() 처럼 '전체(미절단) 프롬프트'의 길이만 셀 때 쓴다.
        절단된/디코드된 문자열을 다시 이걸로 감싸 모델 입력을 만들지 않는다 (#3)."""
        ...
    def encode_chat_template(self, prompt_text: str, response_text: str) -> ChatTemplateEncoding:
        """prompt_text(미절단) 전체를 template 에 넣어 한 번만 토큰화하고
        prefix/content/suffix 로 분리해 반환한다. 실제 모델 입력(input_ids)은
        이 결과의 prefix_ids + content_ids[:budget] + suffix_ids 로 조립한다."""
        ...
    def count_tokens(self, formatted_text: str) -> int: ...


class SGuardModelBackend(Protocol):
    def generate(self, input_ids: list[int]) -> str:
        """고정 5줄 raw output 반환. formatted_text 문자열이 아니라 조립된
        input_ids 를 직접 받는다 — 문자열 재-토큰화로 인한 불일치를 원천 차단 (#3)."""
        ...
    def label_logits(self, input_ids: list[int]) -> dict[str, float] | None:
        """카테고리별 unsafe 라벨 토큰의 p_unsafe (safe/unsafe 두 후보만 softmax 정규화).
        미구현 시 None (unsafe_score 산출 방법: #5)."""
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

        # 전체(미절단) prompt 를 template 에 넣어 '한 번만' 토큰화하고 offset 으로
        # prefix/content/suffix 를 나눈다 (#3). content_ids/offsets 는 encode_content
        # 단독 호출이 아니라 이 template-내장 토큰화에서 나온 것이므로, 실제로
        # 모델이 받는 input_ids 와 분석에 쓰이는 토큰이 완전히 같은 소스다.
        enc = self.tok.encode_chat_template(prompt_text=prompt, response_text="")
        tr: TokenizationResult = analyze_content_tokens(
            prompt=prompt, key_expression=key_expression,
            content_ids=enc.content_ids, offsets=enc.content_offsets,
            budget=budget, decode_fn=self.tok.decode,
        )

        # decode 를 거치지 않고 token id 수준에서 이어붙인다 — U+FFFD 손상,
        # 'Prompt: ' 뒤 공백 병합 유실 등 문자열 왕복으로 생기는 불일치를 원천 차단.
        input_ids = enc.prefix_ids + tr.used_content_ids + enc.suffix_ids
        formatted_tokens = len(input_ids)
        # 방어적 검증: cap 을 전체 입력에 잘못 적용했다면 formatted 가 overhead 보다
        # 짧아지는 파국이 생긴다. template 이 보존됐는지 하한으로 확인.
        if formatted_tokens < config.SGUARD_TEMPLATE_OVERHEAD_TOKENS:
            raise RuntimeError(
                "formatted input 이 template overhead(1,480)보다 짧음 — "
                "template 파괴 절단이 의심됨. tokenizer truncation 사용 여부 점검."
            )
        # decode 는 로그/표시용으로만 쓴다 — 이 문자열을 다시 토큰화해 모델에 먹이지 않는다.
        formatted_text = self.tok.decode(input_ids)

        return PreparedInput(
            input_policy=input_policy,
            experimental_token_cap=budget,
            used_content_text=tr.decoded_used_input,
            used_content_ids=tr.used_content_ids,
            formatted_text=formatted_text,
            formatted_input_ids=input_ids,
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
        raw = self.model.generate(prep.formatted_input_ids)
        categories = parse_sguard_output(raw)
        decision = (schema.UNSAFE
                    if any(v == schema.UNSAFE for v in categories.values())
                    else schema.SAFE)
        # p_unsafe (카테고리별, safe/unsafe 두 후보 softmax 정규화). 가장 위험한
        # 카테고리 기준으로 종합 점수를 낸다 (decision 이 "하나라도 unsafe"인 것과 동일 기준).
        unsafe_probs = self.model.label_logits(prep.formatted_input_ids)
        unsafe_score = max(unsafe_probs.values()) if unsafe_probs else None
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
def _build_real_tokenizer(hf_tok):
    """HF tokenizer 를 SGuardTokenizerBackend 로 감싼다.

    load_real_sguard_adapter(모델 포함) 와 load_sguard_tokenizer_adapter(토크나이저만)
    가 이 하나를 공유한다 — 두 경로가 갈라지면 "분석이 본 토큰"과 "모델이 먹은 토큰"이
    어긋나고, 그건 이 저장소에서 이미 한 번 112건 불일치로 겪은 실패 유형이다
    (analysis/tokenizer/adapter_agreement_report.md).
    """

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
            # 'role' 이 없으면 chat template 이 message['role'] 참조에서
            # UndefinedError 로 실패한다 (실모델 PILOT GATE 0a 에서 확인, #4).
            msgs = [{"role": "user",
                     config.SGUARD_MESSAGE_KEY_PROMPT: prompt_text,
                     config.SGUARD_MESSAGE_KEY_RESPONSE: response_text}]
            return hf_tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)

        def encode_chat_template(self, prompt_text, response_text):
            formatted = self.apply_chat_template(prompt_text, response_text)
            content_start = formatted.find(prompt_text)
            if content_start < 0:
                raise RuntimeError(
                    "prompt_text 가 formatted template 안에서 발견되지 않음 — "
                    "apply_chat_template 이 content 를 변형함(escape 등). 가정 위반."
                )
            content_end = content_start + len(prompt_text)
            enc = hf_tok(formatted, add_special_tokens=False, return_offsets_mapping=True)
            ids = list(enc["input_ids"])
            offsets = [tuple(o) for o in enc["offset_mapping"]]
            prefix_ids, content_ids, content_offsets, suffix_ids = [], [], [], []
            for tid, (s, e) in zip(ids, offsets):
                if e <= content_start:
                    prefix_ids.append(tid)
                elif s >= content_end:
                    suffix_ids.append(tid)
                else:
                    # 경계 토큰(template 문자와 content 문자가 한 토큰으로 병합된 경우)도
                    # 보수적으로 content 로 취급 — _key_token_indices 와 동일한 원칙.
                    content_ids.append(tid)
                    content_offsets.append((s - content_start, e - content_start))
            return ChatTemplateEncoding(prefix_ids=prefix_ids, content_ids=content_ids,
                                        content_offsets=content_offsets, suffix_ids=suffix_ids)

        def count_tokens(self, formatted_text):
            return len(hf_tok(formatted_text, add_special_tokens=False)["input_ids"])

    return _RealTok()


def load_sguard_tokenizer_adapter() -> SGuardAdapter:
    """토크나이저만 로드한 어댑터 (model=None). 가중치 5GB 를 받지 않는다.

    용도: chunk 경계 조립, 토큰 수 계산, prepare_input 검증처럼 추론이 필요 없는 작업.
    predict() 를 부르면 RuntimeError 로 막힌다 — 추론이 필요하면
    load_real_sguard_adapter() 를 쓴다.

    GPU 없는 로컬에서 방어 파이프라인의 토큰 처리 경로를 검증하기 위한 것이다.
    """
    from transformers import AutoTokenizer  # lazy

    hf_tok = AutoTokenizer.from_pretrained(
        config.SGUARD_MODEL_ID, revision=config.SGUARD_REVISION)
    return SGuardAdapter(_build_real_tokenizer(hf_tok), model=None)


def load_real_sguard_adapter(device: str | None = None) -> SGuardAdapter:
    """실제 HF 모델용 로더. transformers 는 여기서만 lazy import.

    device 미지정 시 CUDA 가능 여부로 자동 결정 — "cuda" 하드코딩이면 GPU 없는
    로컬(CPU-only torch)에서 로드 자체가 AssertionError 로 막힌다.
    PILOT GATE 0a 에서 이 로더로 빈-response 동작을 먼저 검증한 뒤 대량 실행할 것.
    """
    import torch  # lazy
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_tok = AutoTokenizer.from_pretrained(
        config.SGUARD_MODEL_ID, revision=config.SGUARD_REVISION)
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.SGUARD_MODEL_ID, revision=config.SGUARD_REVISION).to(device)

    class _RealModel:
        def generate(self, input_ids):
            import torch
            ids_t = torch.tensor([input_ids], device=device)
            attn_t = torch.ones_like(ids_t)
            with torch.no_grad():
                # PILOT GATE 0a 실측: EOS 를 내지 않고 계속 생성하므로 5(카테고리 수)로
                # 명시적으로 자른다. 6 이상을 주면 raw 뒤에 6번째 토큰이 붙어
                # parse_sguard_output 이 5번째 줄 값 파싱에서 실패할 수 있다.
                out = hf_model.generate(input_ids=ids_t, attention_mask=attn_t,
                                        max_new_tokens=len(config.SGUARD_CATEGORIES),
                                        do_sample=False)
            # skip_special_tokens=False 여야 한다. 판정 라벨 5종(id 49159~49168)은
            # base vocab(49152) 바깥의 added token 이라 True 로 디코드하면 전부 벗겨져
            # 빈 문자열이 나오고, parse_sguard_output 이 모든 행에서 실패한다.
            # 실측 근거: analysis/tokenizer/sguard_behavior_gate.json 의 gen_text 8행 전부 "".
            # (게이트는 문자열이 아니라 토큰 id 로 판정을 읽어서 통과했다.)
            # 라벨 토큰의 표면 문자열이 "Crime: safe\n" … "Violence: safe" 이므로
            # False 로 디코드하면 parse_sguard_output 이 기대하는 5줄이 그대로 나온다.
            return hf_tok.decode(out[0, ids_t.shape[1]:], skip_special_tokens=False)

        def label_logits(self, input_ids):
            # 5줄 출력 = 정확히 5토큰, 한 줄=한 토큰, 순서 고정(#5).
            # 스텝 k 의 safe/unsafe 두 후보 logit만 놓고 softmax 정규화한다
            # (전체 vocab softmax 아님) → p_unsafe.
            import torch
            ids_t = torch.tensor([input_ids], device=device)
            attn_t = torch.ones_like(ids_t)
            with torch.no_grad():
                out = hf_model.generate(input_ids=ids_t, attention_mask=attn_t,
                                        max_new_tokens=len(config.SGUARD_CATEGORIES),
                                        do_sample=False,
                                        output_scores=True, return_dict_in_generate=True)
            scores = out.scores  # tuple, 스텝별 [1, vocab] logits
            result = {}
            for step, category in enumerate(config.SGUARD_CATEGORIES):
                if step >= len(scores):
                    break  # 생성이 짧게 끝났으면(=포맷 이상) 그 이후 카테고리는 비움
                token_ids = config.SGUARD_LABEL_TOKEN_IDS[category]
                step_logits = scores[step][0]
                two = torch.stack([step_logits[token_ids["safe"]],
                                   step_logits[token_ids["unsafe"]]])
                probs = torch.softmax(two, dim=0)
                result[category] = probs[1].item()
            return result or None

    return SGuardAdapter(_build_real_tokenizer(hf_tok), _RealModel())
