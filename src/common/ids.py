"""prompt_id / generation_id 규칙.

- prompt_id     : prompts.csv 에서 부여된 그대로 사용 (freeze 이후 불변).
                  형식 예: "VIOL_14", "CRIM_22" — 검증만 하고 재생성하지 않는다.
- generation_id : prompt_id × generator_id × seed 의 결정적 조합.
                  학생 3 의 generation_results.csv 와 image_labels.csv 를 잇는 유일 키.
"""
import re

_PROMPT_ID_RE = re.compile(r"^[A-Z]{3,6}_\d{1,4}$")


def validate_prompt_id(prompt_id: str) -> str:
    if not _PROMPT_ID_RE.match(prompt_id):
        raise ValueError(f"invalid prompt_id: {prompt_id!r} (expected e.g. 'VIOL_14')")
    return prompt_id


def _short_generator_tag(generator_id: str) -> str:
    """'BAAI/AltDiffusion-m18' → 'altdiffusion-m18' (파일명 안전, 소문자)."""
    tag = generator_id.split("/")[-1].lower()
    tag = re.sub(r"[^a-z0-9._-]", "-", tag)
    return tag


def make_generation_id(prompt_id: str, generator_id: str, seed: int) -> str:
    """예: VIOL_14__altdiffusion-m18__s042"""
    validate_prompt_id(prompt_id)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    return f"{prompt_id}__{_short_generator_tag(generator_id)}__s{seed:03d}"


def parse_generation_id(generation_id: str) -> dict:
    try:
        prompt_id, gen_tag, seed_part = generation_id.split("__")
        assert seed_part.startswith("s")
        return {
            "prompt_id": validate_prompt_id(prompt_id),
            "generator_tag": gen_tag,
            "seed": int(seed_part[1:]),
        }
    except (ValueError, AssertionError) as e:
        raise ValueError(f"invalid generation_id: {generation_id!r}") from e
