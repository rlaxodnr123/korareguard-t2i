"""KoRareGuard-T2I 공용 설정.

설계 문서 섹션 2 "실측 확인된 모델 사실"을 상수로 고정한다.
학생 2·3·4 는 이 파일 외의 곳에서 모델 ID / revision / cap 을 하드코딩하지 않는다.
"""

# ---------------------------------------------------------------- 모델 (revision freeze)
SGUARD_MODEL_ID = "SamsungSDS-Research/SGuard-ContentFilter-2B-v1"
SGUARD_REVISION = "870ae18c091f06f8f96e4119051f4cd063c83481"

ALTDIFF_MODEL_ID = "BAAI/AltDiffusion-m18"
ALTDIFF_REVISION = "b3c429d6aa48346ab421ea314375d4a10b9b66e9"

# m9 폴백 (Limitation 10 / PILOT GATE 0b 에서 full pipeline 로드 실패 시에만 교체)
ALTDIFF_FALLBACK_MODEL_ID = "BAAI/AltDiffusion-m9"
ALTDIFF_FALLBACK_REVISION = None  # 폴백 확정 시 pin

# ---------------------------------------------------------------- 실측 사실 (config/runtime 검증값)
SGUARD_NATIVE_CONTEXT = 131_072          # config 실측
SGUARD_TEMPLATE_PREFIX_TOKENS = 1_416    # runtime 실측
SGUARD_TEMPLATE_SUFFIX_TOKENS = 64       # runtime 실측
SGUARD_TEMPLATE_OVERHEAD_TOKENS = (      # = 1,480
    SGUARD_TEMPLATE_PREFIX_TOKENS + SGUARD_TEMPLATE_SUFFIX_TOKENS
)
# SGuard chat template 은 message 키로 `prompt` / `response` 를 쓴다 (`content` 아님).
SGUARD_MESSAGE_KEY_PROMPT = "prompt"
SGUARD_MESSAGE_KEY_RESPONSE = "response"

# tokenizer 실측: truncation_side='right'
#   → truncation=True, max_length=cap 사용 시 template suffix 가 파괴됨.
#   → 절대 tokenizer 내장 truncation 으로 cap 을 적용하지 않는다. (adapter 가 user content 만 자름)
SGUARD_TRUNCATION_SIDE = "right"

ALTDIFF_MODEL_MAX_LENGTH = 77            # tokenizer 설정값 (아키텍처 한계 514 와 별개)
ALTDIFF_NUM_SPECIAL_TOKENS = 2           # <s> ... </s>
ALTDIFF_CONTENT_BUDGET = ALTDIFF_MODEL_MAX_LENGTH - ALTDIFF_NUM_SPECIAL_TOKENS  # = 75

# SGuard/AltDiff 토큰 비율 (432행 실측, 논문 Limitation 3 서술용 참고값)
SGUARD_TO_ALTDIFF_TOKEN_RATIO_MEAN = 1.58
SGUARD_TO_ALTDIFF_TOKEN_RATIO_MEDIAN = 1.60

# ---------------------------------------------------------------- 조건 / cap
# cap 은 "전체 입력"이 아니라 "user content 토큰 예산"이다 (overhead 1,480 ≫ 77).
CAP_CONSTRAINED_77 = 77
CAP_CONSTRAINED_127 = 127                # 조건 2b (선택)

# ---------------------------------------------------------------- SGuard 출력 사양
SGUARD_CATEGORIES = ("crime", "manipulation", "privacy", "sexual", "violence")
# 라벨이 전용 단일 토큰이므로 문자열 파싱 대신 토큰 logit 점수화가 가능하다.
# (id 는 revision 고정 하에서만 유효. PILOT GATE 에서 재확인할 것.)
SGUARD_SAFE_LABEL_TOKEN_IDS = {
    "crime": 49159,          # 'Crime: safe\n'
    "manipulation": 49161,   # 'Manipulation: safe\n'
    "privacy": 49163,        # 'Privacy: safe\n'
}
SGUARD_UNSAFE_LABEL_TOKEN_IDS = {
    "sexual": 49166,         # 'Sexual: unsafe\n'
}
SGUARD_LAST_LINE_TOKEN_IDS = {
    "violence_safe": 49167,  # 'Violence: safe'
}

# ---------------------------------------------------------------- 벤치마크 규모
N_PROMPTS = 432
