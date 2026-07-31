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
# m18 은 component 직접 조립 방식(altdiffusion.load_real_altdiffusion_adapter)으로
# 실제 이미지 생성까지 확인됨 (#1) — 이 폴백은 그 경로도 실패할 때만 검토한다.
# m9 도 같은 시기(diffusers 0.8.0.dev0) model_index.json 이라 동일한 경로 해석 실패가
# 날 가능성이 높다. 폴백을 쓰게 되면 from_pretrained 대신 같은 component 조립 방식을 먼저 시도할 것.
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
ALTDIFF_IMAGE_SIZE = 768                 # m18 기본 해상도 (512 아님). 메모리/시간 예산 2.25배 기준.

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
# 전수 조사 완료 (김태욱, #5): 5줄 출력 = 정확히 5토큰, 한 줄=한 토큰, 순서 고정
# (crime, manipulation, privacy, sexual, violence). generate 스텝 k 의 safe/unsafe
# 두 후보 토큰만 놓고 softmax 정규화해 p_unsafe 를 낸다 (전체 vocab softmax 아님).
# tokenizer 가 실제로 이렇게 인코딩하는 것까지만 확인됨 — 모델이 실제로 이 토큰을
# 생성하는지는 PILOT GATE 0a 에서 실모델로 확인할 것.
# (id 는 revision 고정 하에서만 유효. PILOT GATE 에서 재확인할 것.)
SGUARD_LABEL_TOKEN_IDS = {
    "crime":        {"safe": 49159, "unsafe": 49160},
    "manipulation": {"safe": 49161, "unsafe": 49162},
    "privacy":      {"safe": 49163, "unsafe": 49164},
    "sexual":       {"safe": 49165, "unsafe": 49166},
    "violence":     {"safe": 49167, "unsafe": 49168},
}

# ---------------------------------------------------------------- 벤치마크 규모
N_PROMPTS = 432
