# Tokenization 결과 검증 보고서 (pilot)

- 대상: `analysis\truncation\tokenization_results_pilot.csv` (270 행)
- git commit: `a17bd28a7f1c6e90a1629ec3c893c1e487e90397`
- 생성 시각: 2026-07-30T15:51:08.956131+00:00

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
| concept x rarity x length x position 각 셀 정확히 1개 | PASS | 90 셀 |
| rarity_label 균형 | PASS | {'common': 45, 'rare': 45} |
| length_level 균형 | PASS | {'short': 30, 'near_limit': 30, 'over_limit': 30} |
| position_level 균형 | PASS | {'front': 30, 'middle': 30, 'back': 30} |
| 토큰화 정합성 위반 없음 | PASS | 0건 |
| 조건 수 = 3 | PASS | ['SGuard-ContentFilter-2B-v1|native', 'SGuard-ContentFilter-2B-v1|constrained_77', 'AltDiffusion-m18|native'] |
| 조건별 행 수 동일 | PASS | [90, 90, 90] |
| SGuard-ContentFilter-2B-v1|native: experimental_token_cap 비어있음 (native) | PASS | {''} |
| SGuard-ContentFilter-2B-v1|constrained_77: experimental_token_cap 기록됨 | PASS | {'77'} |
| SGuard-ContentFilter-2B-v1|constrained_77: 절단이 실제로 발생 | PASS | visibility=none 30행 |
| AltDiffusion-m18|native: experimental_token_cap 비어있음 (native) | PASS | {''} |
| 조건 1 (SGuard native) 은 절단 0건 | PASS | 0건 절단 |
| 조건 1 실제 입력이 native context 미만 | PASS | 최대 1926 < 131072 |
| 조건 간 불변량 유지 | PASS | 0건 |
| SGuard 두 policy 의 pre-truncation 토큰 수 동일 (같은 tokenizer) | PASS | 0건 |

## visibility 분포

| 조건 | full | partial | none |
|---|---|---|---|
| SGuard-ContentFilter-2B-v1 (native) | 90 | 0 | 0 |
| SGuard-ContentFilter-2B-v1 (constrained_77) | 59 | 1 | 30 |
| AltDiffusion-m18 (native) | 68 | 2 | 20 |

- prompt 잘렸지만 key full: 55행
- 절단이 key 글자 중간 통과: 1행

**검사 24건 중 FAIL 0건**