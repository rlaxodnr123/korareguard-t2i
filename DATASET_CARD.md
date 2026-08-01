# Korean Rare Expression T2I Safety Benchmark (KRE-T2I) Dataset Card

## 1. Overview
- **Dataset Name:** KRE-T2I (Korean Rare Expression Text-to-Image Safety Benchmark)
- **Total Samples:** 432 Controlled Core Prompts (24 Concepts × 2 Rarity × 3 Lengths × 3 Positions)
- **Purpose:** Text-to-Image (T2I) 파이프라인에서 안전 필터(Safety Filter)와 생성 모델(Generator) 간의 Tokenizer Mismatch 및 Truncation으로 인해 발생하는 보안 취약점(Under-blocking)과 오탐(Over-blocking)을 정량 평가하기 위함.

## 2. Threat Model
- **Attacker Goal:** 희귀 한국어 표현 및 긴 프롬프트(Truncation)를 활용하여 T2I 안전 필터를 우회(Under-blocking)해 유해 이미지를 생성하거나, 무해한 한국 전통문화 프롬프트를 오탐(Over-blocking)하게 만들어 시스템 유용성을 저해함.
- **Capability:** Black-box 접근 권한. 프롬프트 문구, 단어 위치, 문장 길이를 자유롭게 구성 가능.
- **Limitation:** 생성 모델 및 안전 필터의 가중치(Weights), 내부 임계값(Threshold) 변경 불가.
- **Target:** SDXL 1.0, SD 2.1 / SD Safety Checker, Open CLIP Safety Check, NSFW Classifier.
- **Attack Scenarios:** 
  1. Subword Gap Attack: 희귀 고어/은어/순우리말 사용.
  2. Truncation Shift Evasion: 긴 문장 맨 뒤에 민감 단어를 배치하여 필터 단에서 잘려나가게 만듦.

## 3. Taxonomy & Structure
- **Core Concepts (24 Intents):**
  - **Safe (12개):** 한국 전통문화, 민속놀이, 고유명사, 순우리말
  - **Unsafe (12개):** 무기/폭력, 선정성/노출, 범죄/유해물질
- **Rarity / Expression Variants (2 Types):**
  - `common` (일반 표현), `rare` (희귀 표현)
- **Control Variables:**
  - **Length Levels:** `short` (15자 내외), `near_limit` (120자 내외, 생성기(AltDiffusion)의 75~77 토큰 컷오프 경계에 최적화), `over_limit` (400자 이상, 토큰 한계 완전히 초과)
  - **Position Levels:** `front`, `middle`, `back`
- **Confound Control (실험 변수 통제):** 
  - `common`과 `rare` 프롬프트 간의 차이를 완벽히 통제하기 위해, 랜덤 배경 문장(Filler)을 고를 때 시드(Seed) 값을 고정(`concept_id` + `length_level`)하여 사용합니다. 이를 통해 오직 '핵심 단어'만 다르고 배경은 100% 동일한 프롬프트 쌍이 생성되며, 실험 결과의 신뢰성을 보장합니다.

## 4. Pipeline Architecture
```text
[Input Prompt (prompts.csv)]
       │
       ├──> [1. Tokenizer Analysis] (학생 2: CLIP vs BERT/OpenCLIP 토큰 수 및 잘림 위치 분석)
       │
       ├──> [2. Safety Checker]     (학생 4: SD Safety Checker / Open CLIP / NSFW Classifier 통과 여부)
       │
       ├──> [3. Image Generation]   (학생 3: SDXL 1.0 / SD 2.1 생성 수행 및 이미지 저장)
       │
       └──> [4. Integrated Defense] (학생 5: Normalization & Chunking 적용 후 재평가)
```

## 5. Dataset Schema (Primary Key: prompt_id)
- `prompt_id`: 고유 식별자
- `concept_id`: 24개 핵심 개념 ID
- `safety_label`: `safe` 또는 `unsafe`
- `rarity_label`: `common` 또는 `rare`
- `hazard_category`: 6대 세부 카테고리 (Cultural_Event, Cultural_Object, Pure_Korean_Vocabulary, Violence_Weapons, NSFW_Indecency, Crime_Substances)
- `sguard_category`: SGuard 모델을 위한 카테고리 매핑 (Safe, Violence, Sexual, Crime)
- `length_level`: 문장 길이 (`short`, `near_limit`, `over_limit`)
- `position_level`: 주요 단어 위치 (`front`, `middle`, `back`)
- `key_expression`: 삽입된 핵심 단어
- `raw_prompt`: 전체 프롬프트 텍스트 (한국어 필러 적용)
- `normalized_prompt`: 정규화 처리용 (현재 공란)
