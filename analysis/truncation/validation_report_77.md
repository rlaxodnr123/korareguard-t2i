# Tokenization 결과 검증 보고서 (full)

- 대상: `analysis\truncation\tokenization_results_77.csv` (1296 행)
- git commit: `1a6599be00b92e25f19c3ed4f1a8ba3ec2e2b0fe`
- 생성 시각: 2026-07-31T20:52:12.017311+00:00

## 검증 결과

| 항목 | 결과 | 비고 |
|---|---|---|
| primary key (prompt_id x model_id x input_policy) 중복 없음 | PASS |  |
| 필수 컬럼 결측 없음 | PASS |  |
| safety_label 값 어휘 유효 | PASS | ['safe', 'unsafe'] |
| rarity_label 값 어휘 유효 | PASS | ['common', 'rare'] |
| length_level 값 어휘 유효 | PASS | ['near_limit', 'over_limit', 'short'] |
| position_level 값 어휘 유효 | PASS | ['back', 'front', 'middle'] |
| key_visibility 값 어휘 유효 | PASS | ['full', 'none', 'partial'] |
| analysis_status 값 어휘 유효 | PASS | ['ok'] |
| analysis_status = error 행 없음 | PASS | 0건 |
| model_role 이 schema 값 어휘와 일치 | PASS | ['generator', 'text_safety'] vs schema ['generator', 'text_safety'] |
| input_policy 가 schema.ALL_POLICIES 안에 있음 | PASS | ['constrained_77', 'native'] vs schema ['native', 'constrained_77', 'constrained_127'] |
| key_visibility 가 schema.ALL_VISIBILITY 안에 있음 | PASS |  |
| concept x rarity x length x position 각 셀 정확히 1개 | PASS | 432 셀 |
| rarity_label 균형 | PASS | {'common': 216, 'rare': 216} |
| length_level 균형 | PASS | {'short': 144, 'near_limit': 144, 'over_limit': 144} |
| position_level 균형 | PASS | {'front': 144, 'middle': 144, 'back': 144} |
| 토큰화 정합성 위반 없음 | PASS | 0건 |
| 조건 수 = 3 | PASS | ['SGuard-ContentFilter-2B-v1|native', 'SGuard-ContentFilter-2B-v1|constrained_77', 'AltDiffusion-m18|native'] |
| 조건별 행 수 동일 | PASS | [432, 432, 432] |
| SGuard-ContentFilter-2B-v1|native: experimental_token_cap 비어있음 (native) | PASS | {''} |
| SGuard-ContentFilter-2B-v1|constrained_77: experimental_token_cap 기록됨 | PASS | {'77'} |
| SGuard-ContentFilter-2B-v1|constrained_77: 절단이 실제로 발생 | PASS | visibility=none 158행 |
| AltDiffusion-m18|native: experimental_token_cap 비어있음 (native) | PASS | {''} |
| AltDiffusion-m18|native: content budget = declared - special tokens | PASS | declared=77 special=2 budget=75 |
| 조건 1 (SGuard native) 은 절단 0건 | PASS | 0건 절단 |
| 조건 1 실제 입력이 native context 미만 | PASS | 최대 1832 < 131072 |
| 조건 간 불변량 유지 | PASS | 0건 |
| SGuard 두 policy 의 pre-truncation 토큰 수 동일 (같은 tokenizer) | PASS | 0건 |

## visibility 분포

| 조건 | full | partial | none |
|---|---|---|---|
| SGuard-ContentFilter-2B-v1 (native) | 432 | 0 | 0 |
| SGuard-ContentFilter-2B-v1 (constrained_77) | 240 | 34 | 158 |
| AltDiffusion-m18 (native) | 288 | 0 | 144 |

- prompt 잘렸지만 key full: 240행
- 절단이 key 글자 중간 통과: 10행

**검사 28건 중 FAIL 0건**