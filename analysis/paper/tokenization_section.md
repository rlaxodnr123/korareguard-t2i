# Tokenization and Truncation Analysis — draft

> **작성 메모 (논문에는 들어가지 않음)**
>
> - 본문은 영어로 작성한다. 인용 수치는 전부 `analysis/truncation/tokenization_results.csv`
>   에서 나온 실측값이며, `prompts.csv` 를 다시 생성하면 값이 바뀌므로 그때 갱신한다.
> - 수치 재산출: `python analysis/truncation/analyze_tokens.py --full --overwrite`
>   후 `analysis/figures/*.py` 실행.
> - 확정된 부분(RQ-T1 ~ T8)만 서술했다. 팀 결과가 필요한 RQ-T9, H6 는
>   자리만 표시해 두었다.
> - 인과 표현을 쓰지 않는다. 관측·연관까지만 서술한다.
> - 데이터 기준: prompts.csv 432개, 3개 분석 조건, 1,296행. 검증 28건 통과, error 0.

---

## X.1 Setup

We analyze how each component of the pipeline actually reads the same Korean prompt.
The benchmark contains 432 prompts spanning 24 concepts, each realized as a
`common`/`rare` expression pair crossed with three length levels and three key
positions. Every prompt is fed **independently** to each component; the safety
filter's decision does not gate the analysis, because the question is what each
component represents, not what the deployed pipeline would do.

Three input conditions are analyzed:

| Condition | Component | `input_policy` | Content budget |
|---|---|---|---|
| 1 | SGuard-ContentFilter-2B-v1 | `native` | unbounded |
| 2 | SGuard-ContentFilter-2B-v1 | `constrained_77` | 77 tokens |
| — | AltDiffusion-m18 | `native` | 75 tokens |

All three are defined in **content-token space** — the tokens of the user prompt
itself, excluding special tokens and chat-template scaffolding. This is required
for the two components to be comparable at all, and each component's fixed
overhead is recorded separately.

The primary key of the resulting table is `prompt_id × model_id × input_policy`
(1,296 rows). The policy dimension is not optional: conditions 1 and 2 use the
same tokenizer and differ only in input budget, so `prompt_id × model_id` cannot
distinguish them.

**Condition 2's budget of 77 is an experimental cap defined by this study, not
SGuard's native context limit.** It applies to user content only; the chat
template is preserved in full (see X.2).

---

## X.2 Measured component characteristics

All values below were obtained at runtime rather than assumed.

**SGuard-ContentFilter-2B-v1** is a Granite-family causal LM
(`GraniteForCausalLM`) with a native context of 131,072 tokens. Its tokenizer
declares no maximum length. Its chat template embeds the user prompt inside a
safety-taxonomy system prompt, contributing **1,479–1,482 tokens of fixed
overhead** (a 1,416-token prefix and a 64-token suffix). The complete input for
our benchmark therefore ranges from 1,497 to 1,926 tokens — far below the native
context, so **condition 1 truncates nothing** (432/432 prompts fully visible).

Two properties of this template constrain the implementation. First, the
user prompt sits at roughly 94% of the way through the formatted string, so any
cap smaller than the prefix would terminate inside the taxonomy and never reach
the prompt. Second, the tokenizer's `truncation_side` is `right`, so the
tokenizer's built-in truncation would delete the suffix and the generation
marker. Condition 2 is therefore implemented by splicing at the token-id level —
`prefix + content[:budget] + suffix` — rather than by truncating the formatted
string. Re-encoding a decoded truncation is not equivalent: because SGuard uses a
byte-level BPE, a cut inside a Korean syllable produces U+FFFD on decoding, and
re-tokenizing the result changes the token ids for 67 of the 288 truncated
prompts (23.3%).

**AltDiffusion-m18** uses an XLM-R SentencePiece tokenizer with
`model_max_length = 77`. Its text encoder supports 514 positions, so 77 is a
configuration choice inherited from CLIP conventions rather than an architectural
limit. The generation pipeline calls the tokenizer with
`padding="max_length", max_length=77, truncation=True`, which reserves the two
special tokens first; the effective content budget is therefore **75**, not 77.
We verified that the token-id sequence produced by the pipeline is identical to
the one our analysis records.

**The two tokenizers do not measure the same text on the same scale.** Across all
432 prompts, SGuard produces a median of **1.60×** as many tokens as
AltDiffusion for identical Korean input (1.44× when restricted to the key
expressions alone). A budget of 77 SGuard tokens therefore covers substantially
less text than a budget of 75 AltDiffusion tokens. This asymmetry is not an
artifact to be corrected; it is the mechanism the cross-component analysis in X.7
measures.

---

## X.3 Fragmentation of rare expressions (RQ-T1, RQ-T2)

**A raw token count answers the wrong question.** Comparing rare and common
expressions by absolute token count, rare expressions use *fewer* tokens: the
paired median difference (rare − common, over 24 concept pairs) is −3.50 for
SGuard and −2.50 for AltDiffusion, with rare exceeding common in only 3/24 and
7/24 pairs respectively.

This reflects a length asymmetry in the benchmark rather than a property of
rarity. The common expressions are descriptive paraphrases averaging 12.8
characters, while the rare expressions are compound nouns averaging 6.5
characters; the common form is longer in 19 of 24 concepts.

**Normalizing by character count reverses the direction.** Measured as tokens per
character, rare expressions are more finely segmented in **19/24** concept pairs
for SGuard (median difference +0.217) and **21/24** for AltDiffusion (median
+0.271). In absolute terms, SGuard segments rare expressions at 1.22 tokens per
character versus 1.00 for common; AltDiffusion at 0.90 versus 0.68.

The effect appears independently in both tokenizers, which have different
vocabularies (49,152 vs. 250,002) and different segmentation algorithms
(byte-level BPE vs. SentencePiece).

Four concept pairs run in the opposite direction for SGuard and three for
AltDiffusion; `UNSAFE_VIOL_14` is the only pair contrary in both. Because
segmentation of an expression is not entirely context-independent under a
byte-level BPE, we report the median across all nine realizations of each
expression (three lengths × three positions) rather than a single occurrence;
the sensitivity this controls for is quantified in X.8.

*(Figure A: paired comparison of raw token counts and tokens-per-character.)*

> **작성 메모** — 두 지표를 나란히 놓는 것이 이 절의 요점이다. 원시 토큰 수만
> 보고하면 H1 이 기각되는데, 그 기각은 희귀도가 아니라 문자열 길이를 잰 결과다.
> 심사에서 "왜 정규화했나" 를 반드시 물으므로 길이 비대칭 수치(12.8 vs 6.5,
> 19/24)를 먼저 제시한 뒤 정규화 결과를 보여주는 순서를 유지할 것.
> 절대 토큰 수의 컴포넌트 간 비교는 vocabulary 가 달라 descriptive 로만 쓴다.

---

## X.4 Budget consumption does not increase with rarity (RQ-T3)

Section X.3 establishes that rare expressions are more finely segmented per
character. A natural follow-up is whether substituting a rare expression
therefore consumes more of a component's token budget. **It does not.**

Because filler text is held constant within each concept × length group, two
prompts that share a concept, length level and key position differ only in the
key expression, and the difference in their total token counts isolates the
rarity effect. Over the 216 such pairs, substituting the rare form *reduces*
the prompt's content-token count by a median of 3.50 tokens for SGuard and 2.50
for AltDiffusion.

The measurement is well controlled: the total-token difference equals the
key-token difference exactly in **216/216 pairs** for both components, leaving no
residual attributable to filler or to boundary effects. The distribution is also
identical across the three length levels, as expected when only the key varies.

The key expression occupies a small share of either budget:

| Component | Content budget | Rarity | Key tokens (median) | Share of budget |
|---|---|---|---|---|
| SGuard | 77 | common | 12.0 | 15.6% |
| SGuard | 77 | rare | 7.5 | 9.7% |
| AltDiffusion | 75 | common | 7.5 | 10.0% |
| AltDiffusion | 75 | rare | 5.0 | 6.7% |

Consequently, rarity rarely determines whether the key expression fits within the
budget at all: substituting the rare form changes the outcome in only 1 of 216
pairs for SGuard and 3 of 216 for AltDiffusion, and all four cases occur at
`near_limit`, where the key already sits at the boundary.

**Taken together with X.3, the two measurements point in opposite directions.**
Rare expressions are denser per character but shorter overall, and the second
effect dominates: the chain "rare expression → higher token consumption → budget
pressure → truncation of the key" is not supported in this benchmark. Whatever
association exists between rarity and truncation must arise through a different
route than budget consumption.

> **작성 메모** — 이건 음성 결과이고, 원래 가설 사슬의 한 고리를 끊는다.
> 설계 문서의 chain (rare -> 더 많은 토큰 -> 예산 압박 -> key 절단)에서
> 두 번째 화살표가 성립하지 않는다는 것을 실측으로 보인 것이다.
> 숨기지 말고 명시할 것. 216/216 잔차 0 은 통제가 유효하다는 근거라
> 같이 제시하면 결과의 신뢰도가 올라간다.

---

## X.5 Length levels are not tokenizer-calibrated (RQ-T4)

The benchmark's `short` / `near_limit` / `over_limit` labels were assigned by
character length at design time. Measured in content tokens they separate
cleanly, but they do not correspond to a single token boundary shared by both
components:

| Level | SGuard (min / median / max) | AltDiffusion (min / median / max) |
|---|---|---|
| `short` | 16 / 24 / 39 | 10 / 15 / 25 |
| `near_limit` | 109 / 122 / 140 | 66 / **75** / 90 |
| `over_limit` | 400 / 422 / 446 | 262 / 266 / 279 |

`near_limit` sits exactly at AltDiffusion's 75-token content budget (median 75,
48% exceeding it) and is therefore a genuine boundary condition for the
generator. For SGuard under condition 2 the same level exceeds the 77-token
budget in every case. **The label denotes a designed length level, not a
per-component token boundary**, and we report the measured distribution
alongside it throughout.

`over_limit` exceeds both budgets by a factor of three to five, so within that
level truncation is saturated and offers no gradient.

---

## X.6 Position and key retention (RQ-T5, RQ-T7)

Key position was manipulated by string construction (front / middle / back).
We verified that this manipulation survives tokenization: the normalized token
position of the key matches its character position to within 0.04, and the two
components agree with each other to a median of 0.013. Token-level repositioning
was not attempted, as it would require a different prompt string per component
and would make the cross-component comparison in X.7 undefined.

Under condition 1 the key expression is fully visible for all 432 prompts. The
remaining two conditions show a consistent gradient in position and length:

| | front | middle | back |
|---|---|---|---|
| **SGuard @77** `short` | 48 full | 48 full | 48 full |
| `near_limit` | 48 full | 47 full, 1 partial | **48 none** |
| `over_limit` | 48 full | 48 none | 48 none |
| **AltDiffusion** `short` | 48 full | 48 full | 48 full |
| `near_limit` | 48 full | 48 full | **29 full, 19 partial** |
| `over_limit` | 48 full | 48 none | 48 none |

Overall, condition 2 leaves the key fully visible in 287/432 prompts, partially
visible in 1, and entirely absent in 144; AltDiffusion leaves it fully visible in
317, partial in 19, absent in 96.

**Prompt truncation and key truncation are distinct events.** In 241 of the 1,296
rows the prompt was truncated while the key expression remained fully visible.
Conflating the two would misattribute the effect to prompt length rather than to
key position.

A byte-level BPE introduces a further failure mode: because a Korean syllable may
span several tokens, a cut can fall inside a syllable. This occurred within the
key expression in 5 rows, leaving a partial character that decodes to U+FFFD. We
therefore report key retention both in tokens and in fully-recovered characters;
for one prompt the token-level ratio was 0.50 while the character-level ratio was
0.25.

*(Figure B: retention by length × position. Figure E: key end position against
each component's content budget.)*

---

## X.7 Cross-component visibility mismatch (RQ-T6, RQ-T8)

Classifying every prompt by whether each component retains the key expression in
full yields four cells. The two SGuard conditions produce **opposite** mismatch
directions:

| | A: both see | B: filter only | C: generator only | D: neither |
|---|---|---|---|---|
| **Condition 1** (native) | 317 | **115** | **0** | 0 |
| **Condition 2** (@77) | 287 | **0** | **30** | 115 |

Cell B is the configuration in which the safety filter reads the key expression
but the generator does not; cell C is the reverse.

The asymmetry follows directly from the 1.60× token-scale difference reported in
X.2. Under condition 2, SGuard's 77-token budget covers roughly 48
AltDiffusion-equivalent tokens, which is strictly less than AltDiffusion's own 75
— so SGuard can never retain a key that AltDiffusion has lost, and cell B is
structurally empty. Under condition 1 SGuard reads everything, so cell C is
structurally empty instead.

**A consequence for hypothesis testing: the two directional hypotheses cannot be
evaluated within a single condition.** The "filter misses, generator sees"
configuration exists only under condition 2 (n = 30) and the reverse only under
condition 1 (n = 115).

The 30 condition-2 cases concentrate in `near_limit × back` (29 of 30), the
configuration in which a risk-bearing expression is placed at the end of an
otherwise ordinary prompt.

The same arithmetic bounds any alternative cap. A cap of *c* SGuard tokens can
produce cell C only if *c* < 75 × 1.60 ≈ 120; at *c* = 127 the window closes and
the configuration becomes unreachable regardless of how prompts are constructed.
We confirmed this empirically: applying a 127-token cap to the benchmark yields
0 such cases.

*(Figure C: 2 × 2 visibility matrices for both conditions.)*

> **작성 메모** — 마지막 문단이 팀의 127 조건 논의에 대한 답이다. 프롬프트 설계로
> 해결되는 문제가 아니라 두 토크나이저 비율이 정하는 산술이라는 점을 명시해 둘 것.

---

## X.8 Threats to validity

**Single filter, single generator.** All observations are specific to
SGuard-ContentFilter-2B-v1 and AltDiffusion-m18. We do not claim generalization
across safety filters or generators; contrastive models are left to future work.

**The truncation is imposed, not naturally occurring.** SGuard's native context
(131,072) is far larger than any prompt in the benchmark, so condition 2's
77-token budget is a controlled manipulation introduced to isolate the effect of
input truncation. It reproduces the *consequence* of a constrained safety filter,
not an incident observed in a deployed system.

**77 is an experimental cap on user content, not a native limit,** and it does
not denote the same amount of text for both components (X.2).

**Cap selection.** The cap was chosen from the empty region of SGuard's content
token distribution between `short` (≤ 39) and `near_limit` (≥ 109); every value
in that region produces the same truncation pattern. The value 77 was adopted for
correspondence with the generator's native limit. **This selection was made from
the tokenization distribution and fixed before any safety decision was observed.**

**Expression length asymmetry.** Common and rare expressions differ
systematically in length (12.8 vs. 6.5 characters; common longer in 19/24
concepts). We address this by normalizing per character (X.3); raw token counts
are reported only to make the confound explicit.

**`over_limit` saturation.** That level exceeds both budgets by three to five
times, so truncation is total and admits no gradient within the level.

**SGuard is a prompt–response pair classifier.** Its system prompt instructs the
model to evaluate the prompt alone when the response is absent, but also not to
assign risk on the prompt alone. Our setting is pre-generation moderation, so the
response field is always empty. We verified on a small sample that empty and
`"None"` response fields yield identical decisions and that the model does not
collapse to a uniform `safe` verdict, but the behavior under an empty response
remains a property of the model we did not train.

**Benchmark construction.** An earlier revision of the benchmark selected filler
text with a seed derived from the condition labels, so filler co-varied with
rarity and position. This was corrected before the reported run; in the final
benchmark filler is constant within each concept × length group (0/216 rarity
pairs and 0/144 position triples differ).

**Segmentation of an expression is not fully context-independent.** Under a
byte-level BPE, whether the space preceding an expression merges into its first
token changes the token count. In the front position the key begins the prompt
string and has no preceding space, so its isolated segmentation differs from the
middle and back positions by one to two tokens in 5 of 48 (concept, rarity)
combinations for SGuard; AltDiffusion's SentencePiece shows no such variation
(0/48). Reporting a single occurrence would make the paired statistic depend on
which position happened to be selected — for SGuard, the front-only figures are
+0.221 and 4/24 against +0.217 and 3/24 for the other two positions. We therefore
report the median across all nine realizations. Note that in the deployed input
the front position is also preceded by a space (from the template's `"Prompt: "`
marker), so the majority value is the one that matches what the model receives.

**Measurement basis for position metrics.** *(pending)* Our span metrics are
currently computed on the prompt tokenized in isolation, whereas the model
receives it embedded in the chat template. The trailing space of the template's
`"Prompt: "` marker merges into the first content token, shifting the key token
count by one in 15 front-position rows and the sequence length in 73 rows.
Retention, visibility and the final input ids are unaffected. This will be
resolved by computing spans from the in-template encoding.

---

## Pending

| Section | Requires |
|---|---|
| RQ-T9 — association with safety outcomes (Figure G) | `safety_results.csv` |
| H6 — association with generation outcomes (Figure F) | `generation_results.csv`, `image_labels.csv` |
| Root-cause case analysis | both of the above |

> **작성 메모** — Root Cause 절은 두 갈래로 나뉜다.
> under-blocking 이 발생했을 때 `key_visibility` 가 `none`/`partial` 이면 절단이
> 설명 후보이고, `full` 이면 절단으로 설명할 수 없어 희귀 표현 표상 쪽을 봐야 한다.
> 예비 관측에서 후자 사례가 이미 나왔다 (short/front 조건에서 common 은 차단,
> rare 는 통과). 전수 결과가 오면 이 구분으로 분류한다.
