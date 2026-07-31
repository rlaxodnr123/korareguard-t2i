"""시뮬레이션용 mock 백엔드.

실측 사실을 구조적으로 재현한다 (수치 자체가 아니라 '메커니즘'을 재현하는 것이 목적):
- SGuard mock  : 2 chars/token  ┐
- AltDiff mock : 3 chars/token  ┘ → 토큰비 1.5 (실측 1.58 근사, 방향성 동일)
- SGuard template overhead 1,480 (prefix 1,416 + suffix 64), truncation_side='right'
- SGuard mock 모델: '보이는' content 에 트리거 표현이 있으면 해당 카테고리 unsafe
  → 절단으로 트리거가 사라지면 under-blocking 이 기계적으로 발생 (H1 메커니즘)
"""
from src.common import config
from src.adapters.text_safety.sguard import ChatTemplateEncoding


# mock 토큰 id 를 (정수 base + 텍스트조각) 로 인코딩해 decode 를 stateless 로 만든다.
# 실제 HF tokenizer.decode 는 encode 호출 이력에 의존하지 않으므로, mock 도 그래야
# 시뮬레이션이 실제 동작과 어긋나지 않는다.
class _Tok(int):
    """정수처럼 동작하지만 원문 조각을 실어나르는 토큰 id."""
    piece: str
    def __new__(cls, value: int, piece: str):
        obj = super().__new__(cls, value)
        obj.piece = piece
        return obj


def _fixed_chunk_encode(text: str, chunk: int):
    """chunk 글자당 1토큰. id 는 조각을 실은 _Tok, offsets 는 정확한 char span."""
    ids, offsets = [], []
    for s in range(0, len(text), chunk):
        e = min(s + chunk, len(text))
        ids.append(_Tok(s, text[s:e]))
        offsets.append((s, e))
    return ids, offsets


class _MockTokBase:
    padding_side = "left"

    def __init__(self, chunk: int):
        self.chunk = chunk

    def encode_content(self, text):
        return _fixed_chunk_encode(text, self.chunk)

    def decode(self, ids):
        # stateless: 각 토큰이 자기 조각을 안다. plain int 가 섞이면 명시적으로 실패.
        pieces = []
        for t in ids:
            if not isinstance(t, _Tok):
                raise TypeError("mock decode 는 encode_content 가 만든 토큰만 받는다")
            pieces.append(t.piece)
        return "".join(pieces)


class MockSGuardTokenizer(_MockTokBase):
    tokenizer_class = "MockGraniteTokenizer"
    revision = config.SGUARD_REVISION
    truncation_side = config.SGUARD_TRUNCATION_SIDE  # 'right'

    PREFIX_MARK = "<<SGUARD_TAXONOMY_PREFIX>>"
    SUFFIX_MARK = "<<SGUARD_GEN_SUFFIX>>"

    def __init__(self):
        super().__init__(chunk=2)  # 2 chars/token

    def apply_chat_template(self, prompt_text, response_text):
        assert isinstance(prompt_text, str) and isinstance(response_text, str)
        return (f"{self.PREFIX_MARK}{prompt_text}"
                f"<RESPONSE:{response_text}>{self.SUFFIX_MARK}")

    def encode_chat_template(self, prompt_text, response_text):
        formatted = self.apply_chat_template(prompt_text, response_text)
        content_start = formatted.find(prompt_text)
        content_end = content_start + len(prompt_text)
        ids, offsets = _fixed_chunk_encode(formatted, self.chunk)
        prefix_ids, content_ids, content_offsets, suffix_ids = [], [], [], []
        for tid, (s, e) in zip(ids, offsets):
            if e <= content_start:
                prefix_ids.append(tid)
            elif s >= content_end:
                suffix_ids.append(tid)
            else:
                content_ids.append(tid)
                content_offsets.append((s - content_start, e - content_start))
        return ChatTemplateEncoding(prefix_ids=prefix_ids, content_ids=content_ids,
                                    content_offsets=content_offsets, suffix_ids=suffix_ids)

    def count_tokens(self, formatted_text):
        if (self.PREFIX_MARK not in formatted_text
                or self.SUFFIX_MARK not in formatted_text):
            # template 이 파괴된 상태 — 남은 텍스트만 토큰화됐다고 가정
            return -(-len(formatted_text) // self.chunk)
        body = formatted_text.replace(self.PREFIX_MARK, "").replace(self.SUFFIX_MARK, "")
        body = body.split("<RESPONSE:")[0]
        content_tokens = -(-len(body) // self.chunk)
        return config.SGUARD_TEMPLATE_OVERHEAD_TOKENS + content_tokens

    # --- 잘못된 사용법 재현용: tokenizer 내장 우절단을 전체 입력에 적용하면?
    def naive_truncate_formatted(self, formatted_text, max_length: int) -> str:
        """truncation=True, max_length=cap 을 흉내낸다 (앞에서 max_length 토큰만 유지).

        truncation_side='right' → 뒤가 잘림 → suffix 파괴.
        prefix 1,416 ≫ 77 이므로 taxonomy 내부에서 잘려 프롬프트에 도달조차 못 한다.
        """
        keep_chars = max_length * self.chunk
        return formatted_text[:keep_chars]


class MockAltDiffTokenizer(_MockTokBase):
    tokenizer_class = "MockXLMRobertaTokenizer"
    revision = config.ALTDIFF_REVISION
    truncation_side = "right"

    def __init__(self):
        super().__init__(chunk=3)  # 3 chars/token → SGuard 대비 1.5배


class MockSGuardModel:
    """트리거 사전 기반 rule 모델. '보이는' content 로만 판정한다."""

    def __init__(self, tokenizer: MockSGuardTokenizer,
                 triggers: dict[str, list[str]]):
        self.tok = tokenizer
        self.triggers = triggers  # {"violence": ["폭탄제조", ...], ...}

    def _visible_content(self, formatted_text: str) -> str:
        body = formatted_text.replace(self.tok.PREFIX_MARK, "")
        body = body.replace(self.tok.SUFFIX_MARK, "")
        return body.split("<RESPONSE:")[0]

    def generate(self, input_ids: list) -> str:
        # mock decode 는 무손실(char 경계로만 자름)이라 실모델과 달리 round-trip 안전.
        formatted_text = self.tok.decode(input_ids)
        visible = self._visible_content(formatted_text)
        lines = []
        for cat in ("Crime", "Manipulation", "Privacy", "Sexual", "Violence"):
            hit = any(t in visible for t in self.triggers.get(cat.lower(), []))
            lines.append(f"{cat}: {'unsafe' if hit else 'safe'}")
        return "\n".join(lines)

    def label_logits(self, input_ids: list):
        return None


class MockAltDiffPipeline:
    """이미지 대신 '보이는 텍스트'를 기록 — concept 렌더링 여부를 시뮬레이션."""

    def __init__(self, tokenizer: MockAltDiffTokenizer):
        self.tok = tokenizer
        self.log: dict[str, str] = {}

    def generate_image(self, prompt: str, seed: int, out_path: str) -> None:
        ids, _ = self.tok.encode_content(prompt)
        visible = self.tok.decode(ids[:config.ALTDIFF_CONTENT_BUDGET])
        self.log[out_path] = visible
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(visible)
