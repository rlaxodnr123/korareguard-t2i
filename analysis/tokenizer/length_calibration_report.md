# PHASE 3 — Length Calibration & Signal Preview

- 프롬프트 432개 × 3조건 = 1296행, error 0건, 5.1초
- SGuard experimental token cap: **77** (user content budget, native limit 아님)
- AltDiffusion native max length: **77** (runtime)

## 1. length_level 별 절단 전 토큰 분포

### SGuard native (cap=None)

| level | n | min | p25 | median | mean | p75 | max |
|---|---|---|---|---|---|---|---|
| short | 144 | 17 | 21 | 23 | 24.5 | 27 | 41 |
| near_limit | 144 | 109 | 117 | 122 | 122.1 | 126 | 140 |
| over_limit | 144 | 400 | 411 | 424 | 421.3 | 429 | 445 |

### SGuard constrained_77 (cap=77)

| level | n | min | p25 | median | mean | p75 | max |
|---|---|---|---|---|---|---|---|
| short | 144 | 17 | 21 | 23 | 24.5 | 27 | 41 |
| near_limit | 144 | 109 | 117 | 122 | 122.1 | 126 | 140 |
| over_limit | 144 | 400 | 411 | 424 | 421.3 | 429 | 445 |

### AltDiffusion native (cap=77)

| level | n | min | p25 | median | mean | p75 | max |
|---|---|---|---|---|---|---|---|
| short | 144 | 11 | 15 | 17 | 17.7 | 20 | 28 |
| near_limit | 144 | 67 | 74 | 79 | 78.6 | 83 | 92 |
| over_limit | 144 | 263 | 266 | 269 | 269.3 | 271 | 281 |

## 2. cap 초과율

| level | SGuard native | SGuard@77 | AltDiffusion@77 |
|---|---|---|---|
| short |   0/144 (  0.0%) |   0/144 (  0.0%) |   0/144 (  0.0%) |
| near_limit |   0/144 (  0.0%) | 144/144 (100.0%) |  85/144 ( 59.0%) |
| over_limit |   0/144 (  0.0%) | 144/144 (100.0%) | 144/144 (100.0%) |

## 3. SGuard 실제 전체 입력 (chat template 포함)

| level | min | median | max | native context 131,072 초과 |
|---|---|---|---|---|
| short | 1497 | 1503 | 1520 | 0/144 |
| near_limit | 1589 | 1602 | 1620 | 0/144 |
| over_limit | 1881 | 1904 | 1926 | 0/144 |

- template overhead: min 1479 / median 1480 / max 1481

## 4. key_visibility 분포 (length × position)

### SGuard native

| level | position | full | partial | none |
|---|---|---|---|---|
| short | front | 48 | 0 | 0 |
| short | middle | 48 | 0 | 0 |
| short | back | 48 | 0 | 0 |
| near_limit | front | 48 | 0 | 0 |
| near_limit | middle | 48 | 0 | 0 |
| near_limit | back | 48 | 0 | 0 |
| over_limit | front | 48 | 0 | 0 |
| over_limit | middle | 48 | 0 | 0 |
| over_limit | back | 48 | 0 | 0 |

### SGuard constrained_77

| level | position | full | partial | none |
|---|---|---|---|---|
| short | front | 48 | 0 | 0 |
| short | middle | 48 | 0 | 0 |
| short | back | 48 | 0 | 0 |
| near_limit | front | 48 | 0 | 0 |
| near_limit | middle | 43 | 5 | 0 |
| near_limit | back | 0 | 0 | 48 |
| over_limit | front | 48 | 0 | 0 |
| over_limit | middle | 0 | 0 | 48 |
| over_limit | back | 0 | 0 | 48 |

### AltDiffusion native

| level | position | full | partial | none |
|---|---|---|---|---|
| short | front | 48 | 0 | 0 |
| short | middle | 48 | 0 | 0 |
| short | back | 48 | 0 | 0 |
| near_limit | front | 48 | 0 | 0 |
| near_limit | middle | 48 | 0 | 0 |
| near_limit | back | 28 | 20 | 0 |
| over_limit | front | 48 | 0 | 0 |
| over_limit | middle | 0 | 0 | 48 |
| over_limit | back | 0 | 0 | 48 |

## 5. directional mismatch (H2a / H2b 표본)

| 비교 | A 둘다 봄 | B SGuard✓ AltDiff✗ (H2b) | C SGuard✗ AltDiff✓ (H2a) | D 둘다 못봄 |
|---|---|---|---|---|
| SGuard native vs AltDiff | 316 | **116** | **0** | 0 |
| SGuard constrained_77 vs AltDiff | 283 | **0** | **33** | 116 |

## 6. 판정

- SGuard constrained_77: near_limit 프롬프트 절단율 100.0%  (경계가 아니라 전부/전무)
- AltDiffusion native: near_limit 프롬프트 절단율 59.0%  (경계 조건으로 작동)
- SGuard native: visibility {'full': 432}
- SGuard constrained_77: visibility {'full': 283, 'none': 144, 'partial': 5}
- AltDiffusion native: visibility {'full': 316, 'none': 96, 'partial': 20}
- partial(경계에 정확히 걸린 key) 총 25행
- 절단이 key 내부 글자 중간을 지난 행: 5
- prompt 잘렸지만 key full: 252행 (prompt_truncated != key truncated 구분이 살아있음)
- error 행: 0