# RQ-T3 — 희귀 표현의 token budget 소비

filler 가 통제된 벤치마크에서만 유효한 분석이다. 같은 concept x length x
position 안에서 common 과 rare 의 차이가 key 표현뿐이므로, 총 토큰 수의
차이를 희귀도 효과로 읽을 수 있다.

## 1. paired 총 토큰 델타 (rare - common)

| tokenizer | 조건 | n | median | mean | range |
|---|---|---|---|---|---|
| SGuard | all | 216 | -3.50 | -5.24 | -16 … +6 |
| SGuard | short | 72 | -3.50 | -5.24 | -16 … +6 |
| SGuard | near_limit | 72 | -3.50 | -5.24 | -16 … +6 |
| SGuard | over_limit | 72 | -3.50 | -5.24 | -16 … +6 |
| AltDiffusion | all | 216 | -2.50 | -2.92 | -13 … +3 |
| AltDiffusion | short | 72 | -2.50 | -2.92 | -13 … +3 |
| AltDiffusion | near_limit | 72 | -2.50 | -2.92 | -13 … +3 |
| AltDiffusion | over_limit | 72 | -2.50 | -2.92 | -13 … +3 |

## 2. 총 델타가 key 토큰 델타로 설명되는가

- **SGuard** 총 델타 median -3.5, key 델타 median -3.5, 잔차 median +0.0 (잔차 0: 216/216)
- **AltDiffusion** 총 델타 median -2.5, key 델타 median -2.5, 잔차 median +0.0 (잔차 0: 216/216)

잔차는 key 경계의 토큰 병합에서 나온다. filler 가 통제되지 않았다면
훨씬 크게 벌어지므로, 0 근처라는 것이 통제 유효성의 근거가 된다.

## 3. key 표현이 content 예산에서 차지하는 비율

| tokenizer | 예산 | rarity | key 토큰 median | 예산 대비 |
|---|---|---|---|---|
| SGuard | 77 | common | 12.0 | 15.6% |
| SGuard | 77 | rare | 7.5 | 9.7% |
| AltDiffusion | 75 | common | 7.5 | 10.0% |
| AltDiffusion | 75 | rare | 5.0 | 6.7% |

## 4. 희귀도 전환이 예산 초과 여부를 뒤집는 경우

| tokenizer | 예산 | 뒤집힌 쌍 | 방향 |
|---|---|---|---|
| SGuard | 77 | 1/216 | {'rare 만 보임': 1} |
| AltDiffusion | 75 | 3/216 | {'rare 만 보임': 2, 'common 만 보임': 1} |
