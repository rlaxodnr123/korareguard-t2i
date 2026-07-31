"""컬럼명·값 어휘의 단일 출처(SSOT).

학생 2·3·4 는 CSV 컬럼명과 카테고리 값을 문자열 리터럴로 쓰지 않고
반드시 이 모듈의 상수를 import 한다.
→ `key_visibility` vs `visibility` 같은 불일치가 구조적으로 불가능해진다.
"""

# ---------------------------------------------------------------- input_policy 값
POLICY_NATIVE = "native"
POLICY_CONSTRAINED_77 = "constrained_77"
POLICY_CONSTRAINED_127 = "constrained_127"
ALL_POLICIES = (POLICY_NATIVE, POLICY_CONSTRAINED_77, POLICY_CONSTRAINED_127)

# ---------------------------------------------------------------- key_visibility 값
VISIBILITY_FULL = "full"
VISIBILITY_PARTIAL = "partial"
VISIBILITY_NONE = "none"
ALL_VISIBILITY = (VISIBILITY_FULL, VISIBILITY_PARTIAL, VISIBILITY_NONE)


def visibility_from_ratio(key_retention_ratio: float) -> str:
    """key_retention_ratio → key_visibility 파생 규칙 (설계 문서 섹션 8).

    1 → full, 0 < r < 1 → partial, 0 → none. 다른 이름/규칙 혼용 금지.
    """
    if key_retention_ratio >= 1.0:
        return VISIBILITY_FULL
    if key_retention_ratio > 0.0:
        return VISIBILITY_PARTIAL
    return VISIBILITY_NONE


# ---------------------------------------------------------------- model_role 값
ROLE_TEXT_SAFETY = "text_safety"
ROLE_GENERATOR = "generator"

# ---------------------------------------------------------------- 안전 판정 값
SAFE = "safe"
UNSAFE = "unsafe"

# ---------------------------------------------------------------- 조건 라벨 값 (prompts.csv)
POSITION_FRONT, POSITION_MIDDLE, POSITION_BACK = "front", "middle", "back"
LENGTH_SHORT, LENGTH_NEAR_LIMIT, LENGTH_OVER_LIMIT = "short", "near_limit", "over_limit"
RARITY_COMMON, RARITY_RARE = "common", "rare"

# ---------------------------------------------------------------- analysis_status 값
STATUS_OK = "ok"
STATUS_ERROR = "error"

# ================================================================ 컬럼명
# --- tokenization_results.csv (학생 2) — PK = prompt_id × model_id × input_policy
class TokCols:
    # [키]
    PROMPT_ID = "prompt_id"
    MODEL_ID = "model_id"
    MODEL_ROLE = "model_role"
    INPUT_POLICY = "input_policy"
    # [정책]
    EXPERIMENTAL_TOKEN_CAP = "experimental_token_cap"
    MAX_LENGTH_EFFECTIVE = "max_length_effective"
    MAX_LENGTH_SOURCE = "max_length_source"
    # [조건]
    SAFETY_LABEL = "safety_label"
    RARITY_LABEL = "rarity_label"
    LENGTH_LEVEL = "length_level"
    POSITION_LEVEL = "position_level"
    KEY_EXPRESSION = "key_expression"
    # [원본]
    CHARACTER_COUNT = "character_count"
    KEY_CHARACTER_COUNT = "key_character_count"
    # [PASS 1] — 절단 전
    TOTAL_TOKENS_PRETRUNC = "total_tokens_pretrunc"
    KEY_TOKEN_COUNT_ORIGINAL = "key_token_count_original"
    KEY_START_PRETRUNC = "key_start_pretrunc"
    KEY_END_PRETRUNC = "key_end_pretrunc"
    # [PASS 2] — 절단 후(실제 사용)
    TOTAL_TOKENS_USED = "total_tokens_used"
    PROMPT_TRUNCATED = "prompt_truncated"
    KEY_START_TOKEN = "key_start_token"
    KEY_END_TOKEN = "key_end_token"
    KEY_TOKENS_RETAINED = "key_tokens_retained"
    KEY_RETENTION_RATIO = "key_retention_ratio"
    KEY_VISIBILITY = "key_visibility"
    # [위치]
    KEY_START_RATIO = "key_start_ratio"
    KEY_CENTER_RATIO = "key_center_ratio"
    KEY_END_RATIO = "key_end_ratio"
    # [정규화]
    KEY_TOKENS_PER_CHARACTER = "key_tokens_per_character"
    # [유효입력]
    DECODED_USED_INPUT = "decoded_used_input"
    REMOVED_TEXT = "removed_text"
    # [SGuard]
    TEMPLATE_OVERHEAD_TOKENS = "template_overhead_tokens"
    FORMATTED_PRETRUNC_TOKENS = "formatted_pretrunc_tokens"
    # [메타]
    TOKENIZER_CLASS = "tokenizer_class"
    TOKENIZER_REVISION = "tokenizer_revision"
    PADDING_SIDE = "padding_side"
    TRUNCATION_SIDE = "truncation_side"
    # [신뢰성]
    ANALYSIS_STATUS = "analysis_status"
    ERROR_MESSAGE = "error_message"
    TOKEN_IDS_PATH = "token_ids_path"
    RUNTIME_MS = "runtime_ms"


TOKENIZATION_COLUMNS = [
    v for k, v in vars(TokCols).items() if not k.startswith("_")
]


# --- safety_results.csv (학생 4) — PK = prompt_id × input_policy
class SafetyCols:
    PROMPT_ID = "prompt_id"
    INPUT_POLICY = "input_policy"
    MODEL_ID = "model_id"
    SG_CRIME = "sg_crime"
    SG_MANIPULATION = "sg_manipulation"
    SG_PRIVACY = "sg_privacy"
    SG_SEXUAL = "sg_sexual"
    SG_VIOLENCE = "sg_violence"
    DECISION = "decision"            # unsafe if any of the 5 is unsafe
    UNSAFE_SCORE = "unsafe_score"    # 라벨 토큰 logit 기반 (산출 방법 문서화: 김태욱)
    RAW_OUTPUT = "raw_output"
    RUNTIME_MS = "runtime_ms"
    ERROR_TYPE = "error_type"


SAFETY_COLUMNS = [v for k, v in vars(SafetyCols).items() if not k.startswith("_")]
SG_CATEGORY_COLUMNS = [
    SafetyCols.SG_CRIME, SafetyCols.SG_MANIPULATION, SafetyCols.SG_PRIVACY,
    SafetyCols.SG_SEXUAL, SafetyCols.SG_VIOLENCE,
]


# --- generation_results.csv (학생 3) — PK = generation_id
class GenCols:
    GENERATION_ID = "generation_id"
    PROMPT_ID = "prompt_id"
    GENERATOR_ID = "generator_id"
    SEED = "seed"
    IMAGE_PATH = "image_path"
    RUNTIME_MS = "runtime_ms"
    ERROR_TYPE = "error_type"


GENERATION_COLUMNS = [v for k, v in vars(GenCols).items() if not k.startswith("_")]


# --- image_labels.csv (학생 3, 사람 평가)
class ImgCols:
    GENERATION_ID = "generation_id"
    CONCEPT_PRESENT = "concept_present"          # utility (H2b outcome)
    IMAGE_SAFETY_LABEL = "image_safety_label"    # 생성 이미지 자체의 안전성
    ANNOTATOR_1 = "annotator_1"
    ANNOTATOR_2 = "annotator_2"
    FINAL_LABEL = "final_label"
    DISAGREEMENT_RESOLVED_BY = "disagreement_resolved_by"


IMAGE_LABEL_COLUMNS = [v for k, v in vars(ImgCols).items() if not k.startswith("_")]
