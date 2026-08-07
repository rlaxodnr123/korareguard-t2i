# -*- coding: utf-8 -*-
"""국문판과 영문판이 같은 수치를 말하고 있는지 대조한다.

왜 필요한가
-----------
두 판본은 손으로 옮겨 적는다. 국문판만 고치고 영문판을 안 고치거나, 옮기면서 숫자를
잘못 적으면 같은 연구가 두 가지 수치를 말하게 된다. 실제로 영문판은 한동안 X.9 절이
통째로 없는 채로 국문판보다 뒤처져 있었다.

무엇을 보는가
-------------
1. 절 구조가 대응하는가 (## X.n 과 ### X.n.m)
2. 두 판본이 인용한 수치 집합이 같은가

수치 비교에서 제외하는 것
-------------------------
- `> 작성 메모` 블록 — 국문판에만 있는 내부 메모다. 논문 본문이 아니라 번역 대상이
  아니므로, 여기 든 숫자는 비교에서 뺀다.
- 절 번호(X.9.2)와 모델 revision 해시 — 수치 주장이 아니다.
- 한 자리 정수 — 영어는 "4건" 을 "four" 로 풀어 쓰는 일이 잦아 개수가 어긋난다.
  대신 두 자리 이상 수, 소수, 분수(18/36) 는 전부 일치해야 한다.

데이터와의 대조는 이 파일이 하지 않는다. verify_prose_claims.py 와
../truncation/verify_safety_section.py 가 담당한다. 여기서는 두 판본이 서로
어긋나지 않는지만 본다.

    python analysis/paper/verify_translation_parity.py
"""
import io
import re
import sys
from collections import Counter
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
KO = HERE / "tokenization_section_ko.md"
EN = HERE / "tokenization_section.md"

fails = []


def ck(claim, ok, got=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {claim}" + (f"   -> {got}" if got else ""))
    if not ok:
        fails.append(claim)


def body(text):
    """번역 대상이 아닌 부분을 걷어낸 본문."""
    out = []
    for line in text.split("\n"):
        s = line.lstrip()
        if s.startswith(">"):          # 작성 메모 / 인용 메모
            continue
        out.append(line)
    t = "\n".join(out)
    t = re.sub(r"X\.\d(?:\.\d)?", " ", t)                 # 절 번호
    t = re.sub(r"`?[0-9a-f]{7,40}`?", " ", t)             # revision 해시
    return t


def figures(text):
    """두 자리 이상 정수와 소수를 뽑는다. 분수는 분자·분모로 쪼갠다.

    국문은 "24개 중 19개", 영문은 "19 of 24" 나 "19/24" 로 쓰는 등 같은 사실을
    다른 모양으로 적는다. 분수를 쪼개 놓으면 그 표기 차이가 사라지고, 남는 차이는
    실제로 한쪽에만 있는 수치뿐이다.
    """
    c = Counter()
    for m in re.findall(r"\d+\.\d+|\d+/\d+|\d+", text):
        for part in m.split("/"):
            if "." in part or len(part) >= 2:
                c[part] += 1
    return c


def sections(text):
    return (re.findall(r"^## (X\.\d)", text, re.M),
            re.findall(r"^### (X\.\d\.\d)", text, re.M))


def main() -> int:
    for p in (KO, EN):
        if not p.exists():
            sys.exit(f"파일이 없습니다: {p.name}")

    ko_raw = KO.read_text(encoding="utf-8")
    en_raw = EN.read_text(encoding="utf-8")

    print("=" * 88)
    print("국문판 / 영문판 대조")
    print("=" * 88)

    print("\n[1] 절 구조")
    ko_top, ko_sub = sections(ko_raw)
    en_top, en_sub = sections(en_raw)
    ck("최상위 절 목록 일치", ko_top == en_top,
       f"국문 {ko_top}\n         영문 {en_top}" if ko_top != en_top else " / ".join(ko_top))
    ck("하위 절 목록 일치", ko_sub == en_sub,
       f"국문 {ko_sub}\n         영문 {en_sub}" if ko_sub != en_sub else " / ".join(ko_sub))

    print("\n[2] 인용 수치 (두 자리 이상 / 소수 / 분수)")
    # 등장 '횟수' 가 아니라 '집합' 을 비교한다. 같은 사실을 국문은 한 번, 영문은 두 번
    # 적는 일이 흔해서(예: "216쌍 중 1건, AltDiffusion 에서 3건" vs "1 of 216 pairs
    # ... and 3 of 216") 횟수를 맞추면 표현 차이가 전부 FAIL 로 잡힌다. 반면 옮겨
    # 적다 숫자가 틀리면(0.0057 -> 0.0059) 그 값이 한쪽에만 남으므로 집합 비교로 잡힌다.
    kf, ef = set(figures(body(ko_raw))), set(figures(body(en_raw)))
    only_ko, only_en = sorted(kf - ef), sorted(ef - kf)
    ck("한쪽에만 등장하는 수치 없음", not only_ko and not only_en,
       (f"\n         국문에만 {only_ko}" if only_ko else "") +
       (f"\n         영문에만 {only_en}" if only_en else "") or
       f"{len(kf)}종 일치")

    print(f"\n{'=' * 88}")
    print(f"  FAIL {len(fails)}건" + ("".join("\n    - " + f for f in fails) if fails else ""))
    print("=" * 88)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
