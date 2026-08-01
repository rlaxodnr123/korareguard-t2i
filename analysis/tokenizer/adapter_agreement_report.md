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
| `key_start_ratio` | `native` | 52 | {'middle': 26, 'back': 26} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 0.491031 / 내 것 0.4898876404494382 |
| `key_start_ratio` | `constrained_77` | 52 | {'middle': 26, 'back': 26} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 0.491031 / 내 것 0.4898876404494382 |
| `key_center_ratio` | `native` | 67 | {'middle': 26, 'back': 26, 'front': 15} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 0.517937 / 내 것 0.5168539325842696 |
| `key_center_ratio` | `constrained_77` | 67 | {'middle': 26, 'back': 26, 'front': 15} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 0.517937 / 내 것 0.5168539325842696 |
| `key_end_ratio` | `native` | 67 | {'middle': 26, 'back': 26, 'front': 15} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 0.544843 / 내 것 0.5438202247191011 |
| `key_end_ratio` | `constrained_77` | 67 | {'middle': 26, 'back': 26, 'front': 15} | SAFE_CULT_03_COMMON_OVER_LIMIT_MIDDLE: 어댑터 0.544843 / 내 것 0.5438202247191011 |
| `key_tokens_per_character` | `native` | 15 | {'front': 15} | SAFE_CULT_04_RARE_SHORT_FRONT: 어댑터 2.000000 / 내 것 1.5 |
| `key_tokens_per_character` | `constrained_77` | 15 | {'front': 15} | SAFE_CULT_04_RARE_SHORT_FRONT: 어댑터 2.000000 / 내 것 1.5 |

모델이 실제로 받는 입력은 어댑터 쪽이므로, 어긋난 필드는 어댑터 값이 옳다.

결론에 직접 쓰는 필드(`key_retention_ratio`, `key_visibility`, `total_tokens_used`) 불일치: **112건**


## AltDiffusion — 글자 단위 retention 분모

절단이 없는 행에서 `key_retention_ratio_char` 가 1 미만인 경우 없음.
