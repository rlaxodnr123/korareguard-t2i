# KoRareGuard-T2I 공용 코드 (`src/common`, `src/adapters`)

학생 2·3·4 가 공유하는 코드. **모델 접근은 반드시 adapter 를 통해서만** 한다.
토크나이저를 따로 불러오면 학생 2 가 기록한 입력과 학생 3·4 가 먹인 입력이 어긋난다.

## 각 학생이 부르는 것

| 학생 | 부르는 메서드 | 산출물 | 컬럼 상수 |
|---|---|---|---|
| 2 이현준 | `SGuardAdapter.tokenize()`, `AltDiffusionAdapter.tokenize()` | `tokenization_results.csv` (1,296행) | `schema.TokCols` |
| 4 김태욱 | `SGuardAdapter.predict()` | `safety_results.csv` (864행) | `schema.SafetyCols` |
| 3 정명섭 | `AltDiffusionAdapter.generate()` | `generation_results.csv`, `image_labels.csv` | `schema.GenCols`, `schema.ImgCols` |

학생 2 의 `tokenize()` 와 학생 4 의 `predict()` 는 **동일한 `prepare_input()`** 경로를 지난다.
→ "학생 2 가 본 입력 = 학생 4 가 먹인 입력" 이 코드로 보장된다.

## 절대 규칙 (실험 정당성 직결)

1. **cap 은 user content 토큰 예산이다.** 전체 입력이 아니다.
   SGuard 는 template overhead 1,480 이 앞뒤로 붙고, content 만 cap 으로 자른다.
2. **tokenizer 내장 truncation(`truncation=True, max_length=cap`)을 쓰지 않는다.**
   SGuard `truncation_side='right'` 라 template suffix 가 파괴된다.
   `adapters/token_analysis.py` 가 content 토큰만 앞에서 budget 개 취한다.
3. **문자열 리터럴로 컬럼명·정책·가시성 값을 쓰지 않는다.** `schema.py` 상수만 쓴다.
4. **학생 3 은 SGuard 게이트 없이 432 전부 생성한다** (실험 A). 실험 B 는 join 으로 사후 계산.

## 시뮬레이션이 확인한 정당성 (`simulation/`)

`python -m simulation.simulate`      → 단위 정당성 26 CHECK
`python -m simulation.simulate_e2e`  → 432 규모 파이프라인 10 CHECK

검증된 설계 주장:
- SGuard/AltDiff 토큰비 방향(SGuard 가 더 많이 씀) → 같은 77 이 같은 분량이 아님
- cap 77 적용 후에도 template 보존, 전체 입력 ≈ 1,557 < native 131,072
- 조건2@77 절단 패턴표(short 통제군 / front<middle<back / near_limit 은 middle 보존·back 절단)
- H1 메커니즘: 조건1 차단 → 조건2 에서 back 절단으로 under-blocking
- 교차표 비대칭: **B행(SGuard✓·AltDiff✗)=0** → H2b 는 조건2@77 back 에서 측정 불가, 조건1 배정 정당
- H2a 검정 가능: 조건2@77·unsafe 에서 A(둘다봄) vs C(SGuard✗·AltDiff✓) under-blocking 대조 성립

## 시뮬레이션이 드러낸 주의사항 (prompts.csv freeze 전 확인)

**H2a 표본(C행)은 벤치마크 길이 분포에 의존한다.** key 가 back 이고
"SGuard content budget 은 넘지만 AltDiff content budget 은 안 넘는" 길이 구간의
프롬프트가 없으면 C행=0 이 되어 H2a 검정 자체가 불가능하다.
→ freeze 시 이 구간 표본 수를 실측 tokenizer 로 세어 확정할 것 (남은 사항 #9).
→ mock 은 2c/t·3c/t 근사이며, 실제 경계는 SGuard/AltDiff 실측 토크나이저로 재확정한다.

## PILOT GATE (freeze 직후, 대량 실행 전 — 반드시 통과)

- 0a: SGuard 빈 response 동작 (`load_real_sguard_adapter` 로 response="" 판정 확인)
- 0b: AltDiffusion full pipeline 로드 (`load_real_altdiffusion_adapter`, 실패 시 m9 폴백)
