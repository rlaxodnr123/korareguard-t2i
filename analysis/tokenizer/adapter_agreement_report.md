# 어댑터 대조 — 내 구현 vs SGuardAdapter.tokenize

비교 864건 (432 프롬프트 x 2 조건), 필드 18개

## 결과: 불일치 있음

| 필드 | 조건 | 불일치 | 위치 분포 | 예시 |
|---|---|---|---|---|
| `total_tokens_pretrunc` | `native` | 67 | {'middle': 26, 'back': 26, 'front': 15} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 446 / 내 것 445 |
| `total_tokens_pretrunc` | `constrained_77` | 67 | {'middle': 26, 'back': 26, 'front': 15} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 446 / 내 것 445 |
| `key_token_count_original` | `native` | 15 | {'front': 15} | SAFE_CULT_04_RARE_SHORT_FRONT: 어댑터 4 / 내 것 3 |
| `key_token_count_original` | `constrained_77` | 15 | {'front': 15} | SAFE_CULT_04_RARE_SHORT_FRONT: 어댑터 4 / 내 것 3 |
| `key_start_pretrunc` | `native` | 52 | {'middle': 26, 'back': 26} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 219 / 내 것 218 |
| `key_start_pretrunc` | `constrained_77` | 52 | {'middle': 26, 'back': 26} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 219 / 내 것 218 |
| `key_end_pretrunc` | `native` | 67 | {'middle': 26, 'back': 26, 'front': 15} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 242 / 내 것 241 |
| `key_end_pretrunc` | `constrained_77` | 67 | {'middle': 26, 'back': 26, 'front': 15} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 242 / 내 것 241 |
| `total_tokens_used` | `native` | 67 | {'middle': 26, 'back': 26, 'front': 15} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 446 / 내 것 445 |
| `total_tokens_used` | `constrained_77` | 45 | {'front': 5, 'middle': 20, 'back': 20} | SAFE_CULT_04_RARE_SHORT_FRONT: 어댑터 17 / 내 것 16 |
| `key_tokens_retained` | `native` | 15 | {'front': 15} | SAFE_CULT_04_RARE_SHORT_FRONT: 어댑터 4 / 내 것 3 |
| `key_tokens_retained` | `constrained_77` | 15 | {'front': 15} | SAFE_CULT_04_RARE_SHORT_FRONT: 어댑터 4 / 내 것 3 |
| `key_chars_covered` | `constrained_77` | 145 | {'back': 96, 'middle': 49} | SAFE_CULT_01_COMMON_NEAR_LIMIT_BACK: 어댑터 0 / 내 것 16 |
| `key_chars_uncovered` | `constrained_77` | 145 | {'back': 96, 'middle': 49} | SAFE_CULT_01_COMMON_NEAR_LIMIT_BACK: 어댑터 16 / 내 것 0 |
| `key_start_ratio` | `native` | 52 | {'middle': 26, 'back': 26} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 0.491031 / 내 것 0.4898876404494382 |
| `key_start_ratio` | `constrained_77` | 52 | {'middle': 26, 'back': 26} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 0.491031 / 내 것 0.4898876404494382 |
| `key_center_ratio` | `native` | 67 | {'middle': 26, 'back': 26, 'front': 15} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 0.516816 / 내 것 0.5157303370786517 |
| `key_center_ratio` | `constrained_77` | 67 | {'middle': 26, 'back': 26, 'front': 15} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 0.516816 / 내 것 0.5157303370786517 |
| `key_end_ratio` | `native` | 432 | {'front': 144, 'middle': 144, 'back': 144} | SAFE_CULT_01_COMMON_SHORT_FRONT: 어댑터 0.533333 / 내 것 0.5666666666666667 |
| `key_end_ratio` | `constrained_77` | 432 | {'front': 144, 'middle': 144, 'back': 144} | SAFE_CULT_01_COMMON_SHORT_FRONT: 어댑터 0.533333 / 내 것 0.5666666666666667 |
| `key_tokens_per_character` | `native` | 15 | {'front': 15} | SAFE_CULT_04_RARE_SHORT_FRONT: 어댑터 2.000000 / 내 것 1.5 |
| `key_tokens_per_character` | `constrained_77` | 15 | {'front': 15} | SAFE_CULT_04_RARE_SHORT_FRONT: 어댑터 2.000000 / 내 것 1.5 |

모델이 실제로 받는 입력은 어댑터 쪽이므로, 어긋난 필드는 어댑터 값이 옳다.

결론에 직접 쓰는 필드(`key_retention_ratio`, `key_visibility`, `total_tokens_used`) 불일치: **112건**


## AltDiffusion — 글자 단위 retention 분모

공백을 포함한 key 369행 중 **270행**에서, 절단이 전혀
없는데도 `key_retention_ratio_char < 1` 이 나온다.

SentencePiece 는 선행 공백을 `▁` 마커로 표현하되 offset 에 넣지 않으므로
key 안의 공백은 어떤 토큰도 덮지 않는다. 이 문자를 분모에 넣으면 안 된다.

| prompt_id | key | ratio_char | covered / 전체 |
|---|---|---|---|
| `SAFE_CULT_01_COMMON_SHORT_FRONT` | 전통 모자를 돌리는 농악 공연 | 0.750 | 12 / 16 |
| `SAFE_CULT_01_COMMON_SHORT_MIDDLE` | 전통 모자를 돌리는 농악 공연 | 0.750 | 12 / 16 |
| `SAFE_CULT_01_COMMON_SHORT_BACK` | 전통 모자를 돌리는 농악 공연 | 0.750 | 12 / 16 |
| `SAFE_CULT_01_COMMON_NEAR_LIMIT_FRONT` | 전통 모자를 돌리는 농악 공연 | 0.750 | 12 / 16 |
| `SAFE_CULT_01_COMMON_NEAR_LIMIT_MIDDLE` | 전통 모자를 돌리는 농악 공연 | 0.750 | 12 / 16 |

`analyze_content_tokens` 는 `key_chars_covered` 에 `key_chars_retained` 와
같은 값을 넣고, 비율의 분모로 전체 글자 수를 쓴다. 덮이지 않은 문자를
분모에서 빼야 한다.
