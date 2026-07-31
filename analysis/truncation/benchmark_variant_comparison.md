# 벤치마크 변형 비교 — 정식 분석에 어떤 프롬프트 파일을 쓸 것인가

세 파일에 **동일한 파이프라인**(`analyze_tokens.py --full`)을 돌린 결과다.

> **이 문서의 수치는 논문에 인용하지 않는다.** 파일 선택 근거를 남기기 위한
> 내부 기록이다. 결과값이 좋은 쪽을 고르는 것을 막기 위해, 선택 기준은
> 아래 C1·C2 두 가지 **설계 조건**으로 사전에 고정했다.

## C1 — filler 통제 (선택 기준)

같은 concept x length x position 안에서 `raw_prompt` 에서 key 표현을 지운
나머지가 common 과 rare 사이에 동일해야 한다. 동일하지 않으면 총 토큰
델타에 filler 차이가 섞여 RQ-T3 를 희귀도 효과로 해석할 수 없다.

| 파일 | 통제된 쌍 | 판정 | 예시 |
|---|---|---|---|
| `prompts.csv` | 216/216 | **PASS** | — |
| `prompts_77.csv` | 76/216 | **FAIL** | SAFE_CULT_01/short/front, SAFE_CULT_01/short/middle |
| `prompts_127.csv` | 76/216 | **FAIL** | SAFE_CULT_01/short/front, SAFE_CULT_01/short/middle |

## C2 — key 표현 쌍 동일성 (선택 기준)

`prompts.csv` 를 기준으로 concept 별 (common, rare) 쌍을 비교한다.

| 파일 | 기준과 동일한 concept | 다른 concept |
|---|---|---|
| `prompts.csv` | 24/24 | — |
| `prompts_77.csv` | 18/24 | `SAFE_VOCAB_11`, `UNSAFE_CRIM_22`, `UNSAFE_NSFW_19`, `UNSAFE_NSFW_20`, `UNSAFE_VIOL_14`, `UNSAFE_VIOL_15` |
| `prompts_127.csv` | 18/24 | `SAFE_VOCAB_11`, `UNSAFE_CRIM_22`, `UNSAFE_NSFW_19`, `UNSAFE_NSFW_20`, `UNSAFE_VIOL_14`, `UNSAFE_VIOL_15` |

## 참고 1 — pre-truncation 토큰 수 (median)

| 파일 | SGuard | AltDiffusion | 비율 |
|---|---|---|---|
| `prompts.csv` | 122.0 | 75.0 | 1.63 |
| `prompts_77.csv` | 169.0 | 104.0 | 1.62 |
| `prompts_127.csv` | 450.5 | 273.5 | 1.65 |

## 참고 2 — key_visibility (full / partial / none, 432 중)

| 파일 | SGuard native | SGuard constrained_77 | AltDiffusion (75) |
|---|---|---|---|
| `prompts.csv` | 432 / 0 / 0 | 287 / 1 / 144 | 317 / 19 / 96 |
| `prompts_77.csv` | 432 / 0 / 0 | 240 / 34 / 158 | 288 / 0 / 144 |
| `prompts_127.csv` | 432 / 0 / 0 | 189 / 3 / 240 | 220 / 0 / 212 |

## 참고 3 — SGuard constrained_77 × AltDiffusion 교차표

**C** 가 H2a(안전 필터는 위험 표현을 못 보지만 생성 모델은 보는 구간)의
구조적 창이다. 실제 필터 판정이 아니라 입력 가시성만으로 정의된다.

| 파일 | A 둘 다 full | B SGuard만 | **C AltDiff만 (H2a)** | D 둘 다 아님 |
|---|---|---|---|---|
| `prompts.csv` | 287 | 0 | **30** | 115 |
| `prompts_77.csv` | 240 | 0 | **48** | 144 |
| `prompts_127.csv` | 189 | 0 | **31** | 212 |

## 참고 4 — RQ-T3 paired 총 토큰 델타

잔차 = (총 토큰 델타) − (key 토큰 델타). filler 가 통제되면 0 이어야 한다.
C1 FAIL 인 파일에서는 이 수치를 희귀도 효과로 해석할 수 없다.

| 파일 | C1 | SGuard median | 잔차 0 | AltDiff median | 잔차 0 |
|---|---|---|---|---|---|
| `prompts.csv` | PASS | -3.5 | 216/216 | -2.5 | 216/216 |
| `prompts_77.csv` | FAIL | -6.0 | 91/216 | -3.0 | 76/216 |
| `prompts_127.csv` | FAIL | -6.0 | 76/216 | -3.0 | 76/216 |

## 결론

C1 을 통과한 파일: `prompts.csv`

정식 분석(`tokenization_results.csv`, figure A–E, 논문 X장)은 `prompts.csv` 로만 수행했다.
나머지 파일의 수치는 위 표에만 남기고 인용하지 않는다.
