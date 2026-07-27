# Korean Rare Expression T2I Safety Benchmark (KRE-T2I) Dataset Card

## 1. Overview
- **Dataset Name:** KRE-T2I (Korean Rare Expression Text-to-Image Safety Benchmark)
- **Total Samples:** 1,728 Controlled Prompts
- **Purpose:** Text-to-Image (T2I) 파이프라인에서 안전 필터(Safety Filter)와 생성 모델(Generator) 간의 Tokenizer Mismatch 및 Truncation으로 인해 발생하는 보안 취약점(Under-blocking)과 오탐(Over-blocking)을 정량 평가하기 위함.

## 2. Taxonomy & Structure
- **Core Concepts (32 Intents):**
  - Normal (Benign, B01~B16): 한국 전통문화, 민속놀이, 고유명사, 순우리말 (16개)
  - Sensitive (Red-Teaming, S01~S16): T2I 안전 가이드라인 5대 영역(폭력/무기, NSFW, 약물, 혐오, 자해) 우회 표현 (16개)
- **Expression Types (6 Types):**
  - `common` (일반 표현), `rare` (희귀 표현), `dialect` (방언/특화 표현), `romanized` (로마자 표기), `code_mixing` (한영 혼용), `spelling_variant` (철자/띄어쓰기 변형)
- **Control Variables:**
  - Length Levels: `short` (~20 토큰), `threshold` (~80 토큰), `over` (150+ 토큰)
  - Key Positions: `start` (문장 앞), `middle` (문장 중간), `end` (문장 뒤)
