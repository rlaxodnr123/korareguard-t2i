# -*- coding: utf-8 -*-
"""논문 산문(표 밖 문장)에 인용된 수치를 원본 데이터에서 재계산해 대조한다.

역할 분담
---------
- validate_results.py        : tokenization_results.csv 의 내부 정합성 (28건)
- verify_safety_section.py   : X.9 의 판정 결과 수치 (safety_results.csv 의존)
- 이 파일                    : 위 둘이 덮지 않는 산문 주장 —
    X.6  위치 중심 표(SGuard 기준), AltDiffusion 의 기울기와 미세 겹침,
         토큰중심-문자중심 편차(구성요소별), 길이x위치 가시성 표(셀 18개)
    X.8  분절의 문맥 의존(5/48, 0/48), 위치별 t/c 델타(+0.218 / +0.207)
    X.9.1 행당 처리 시간 중앙값
    X.9.6 조각 비율(쥐불놀이/강강술래 분절, 중앙값, 안전 개념 순위)

2026-08-07 검토에서 이 절들의 주장이 어떤 자동 검사에도 안 걸린다는 것이 드러났고,
실제로 X.6 이 SGuard 단독 측정값을 기준 명시 없이 일반 주장처럼 쓰고 있었다
(AltDiffusion 은 front/middle 범위가 2/288행에서 겹친다). 기대값은 논문 본문 값을
하드코딩한다. FAIL 이면 데이터가 아니라 논문 문장을 고치는 것이 원칙이다.

SGuard tokenizer 를 내려받아 로드하므로 첫 실행은 네트워크가 필요할 수 있다
(가중치는 받지 않는다).

    python analysis/paper/verify_prose_claims.py
"""
import csv
import io
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))


def rd(p):
    if not p.exists():
        sys.exit(f"입력 파일이 없습니다: {p.relative_to(R)}")
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


rows = rd(R / "analysis" / "truncation" / "tokenization_results.csv")
saf = rd(R / "evaluation" / "safety" / "safety_results.csv")
P = {r["prompt_id"]: r for r in rd(R / "benchmarks" / "prompts" / "prompts.csv")}

sg = {r["prompt_id"]: r for r in rows
      if r["model_role"] == "text_safety" and r["input_policy"] == "native"}
sg77 = {r["prompt_id"]: r for r in rows
        if r["model_role"] == "text_safety" and r["input_policy"] == "constrained_77"}
ad = {r["prompt_id"]: r for r in rows if r["model_role"] == "generator"}

fails = []


def ck(claim, ok, got=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {claim}" + (f"   -> {got}" if got else ""))
    if not ok:
        fails.append(claim)


SEP = "=" * 92
print(SEP)
print("논문 산문 주장 재검증 (X.6 / X.8 / X.9.1 / X.9.6)")
print(SEP)

# ---------------------------------------------------------------- X.6 위치 중심
print("\n[X.6] 위치 중심 — SGuard(조건 1) 기준 표")
for tbl, nm, exp in (
        (sg, "SGuard", {"front": (0.038, 0.004, 0.324),
                        "middle": (0.506, 0.437, 0.550),
                        "back": (0.954, 0.649, 0.993)}),
        (ad, "AltDiffusion", {"front": (0.041, 0.006, 0.354),
                              "middle": (0.475, 0.350, 0.560),
                              "back": (0.946, 0.604, 0.991)})):
    per = defaultdict(list)
    for p, r in tbl.items():
        per[P[p]["position_level"]].append(float(r["key_center_ratio"]))
    for pos, (md, lo, hi) in exp.items():
        v = sorted(per[pos])
        ck(f"{nm} {pos}: 중앙값 {md} / 범위 {lo}~{hi}",
           abs(st.median(v) - md) < 5e-4 and abs(v[0] - lo) < 5e-4 and abs(v[-1] - hi) < 5e-4,
           f"{st.median(v):.3f} / {v[0]:.3f}~{v[-1]:.3f}")

# SGuard 세 범위 비겹침 / AltDiffusion 겹침 2행
per_sg = defaultdict(list)
for p, r in sg.items():
    per_sg[P[p]["position_level"]].append(float(r["key_center_ratio"]))
ck("SGuard 세 범위 비겹침",
   max(per_sg["front"]) < min(per_sg["middle"]) and max(per_sg["middle"]) < min(per_sg["back"]))
per_ad = defaultdict(list)
for p, r in ad.items():
    per_ad[P[p]["position_level"]].append(float(r["key_center_ratio"]))
n_ov = sum(1 for v in per_ad["front"] if v >= min(per_ad["middle"])) + \
       sum(1 for v in per_ad["middle"] if v <= max(per_ad["front"]))
ck("AltDiffusion front/middle 겹침 288행 중 2행", n_ov == 2, f"{n_ov}행")


def char_center(pid):
    p = P[pid]
    i = p["raw_prompt"].find(p["key_expression"])
    return (2 * i + len(p["key_expression"])) / 2 / len(p["raw_prompt"])


print("\n[X.6] 토큰중심-문자중심 편차")
for tbl, nm, (emd, ep90, emx) in ((sg, "SGuard", (0.006, 0.042, 0.102)),
                                  (ad, "AltDiffusion", (0.012, 0.066, 0.150))):
    dev = sorted(abs(float(tbl[p]["key_center_ratio"]) - char_center(p)) for p in tbl)
    p90 = dev[int(len(dev) * 0.9)]
    ck(f"{nm}: 중앙값 {emd} / p90 {ep90} / 최대 {emx}",
       abs(st.median(dev) - emd) < 5e-4 and abs(p90 - ep90) < 5e-4 and abs(dev[-1] - emx) < 5e-4,
       f"{st.median(dev):.3f} / {p90:.3f} / {dev[-1]:.3f}")
cross = sorted(abs(float(sg[p]["key_center_ratio"]) - float(ad[p]["key_center_ratio"])) for p in sg)
ck("구성요소 간: 중앙값 0.012 / 최대 0.121",
   abs(st.median(cross) - 0.012) < 5e-4 and abs(cross[-1] - 0.121) < 5e-4,
   f"{st.median(cross):.3f} / {cross[-1]:.3f}")

# ---------------------------------------------------------------- X.6 가시성 표
print("\n[X.6] 길이 x 위치 가시성 표 (셀 18개)")
EXP_GRID = {
    "SGuard@77": (sg77, {("short", "front"): (48, 0, 0), ("short", "middle"): (48, 0, 0),
                         ("short", "back"): (48, 0, 0), ("near_limit", "front"): (48, 0, 0),
                         ("near_limit", "middle"): (47, 1, 0), ("near_limit", "back"): (0, 0, 48),
                         ("over_limit", "front"): (48, 0, 0), ("over_limit", "middle"): (0, 0, 48),
                         ("over_limit", "back"): (0, 0, 48)}),
    "AltDiffusion": (ad, {("short", "front"): (48, 0, 0), ("short", "middle"): (48, 0, 0),
                          ("short", "back"): (48, 0, 0), ("near_limit", "front"): (48, 0, 0),
                          ("near_limit", "middle"): (48, 0, 0), ("near_limit", "back"): (29, 19, 0),
                          ("over_limit", "front"): (48, 0, 0), ("over_limit", "middle"): (0, 0, 48),
                          ("over_limit", "back"): (0, 0, 48)}),
}
for nm, (tbl, exp) in EXP_GRID.items():
    g = defaultdict(Counter)
    for p, r in tbl.items():
        g[(P[p]["length_level"], P[p]["position_level"])][r["key_visibility"]] += 1
    bad = [(k, (g[k]["full"], g[k]["partial"], g[k]["none"]))
           for k in exp if (g[k]["full"], g[k]["partial"], g[k]["none"]) != exp[k]]
    ck(f"{nm} 셀 9개 전부 일치", not bad, f"불일치 {bad}" if bad else "")

# ---------------------------------------------------------------- X.8
print("\n[X.8] 분절의 문맥 의존")
for tbl, nm, exp in ((sg, "SGuard", 5), (ad, "AltDiffusion", 0)):
    per = defaultdict(dict)
    for p, r in tbl.items():
        per[(r["concept_id"], P[p]["rarity_label"])].setdefault(
            P[p]["position_level"], int(r["key_token_count_original"]))
    diff = sum(1 for d in per.values() if len(set(d.values())) > 1)
    ck(f"{nm}: 위치에 따라 key 토큰 수가 다른 조합 {exp}/48", diff == exp, f"{diff}/48")


def tpc_delta(pos):
    acc = defaultdict(dict)
    for p, r in sg.items():
        if P[p]["position_level"] != pos:
            continue
        acc[(r["concept_id"], P[p]["length_level"])].setdefault(
            P[p]["rarity_label"], float(r["key_tokens_per_character"]))
    per_c = defaultdict(list)
    for (c, _), d in acc.items():
        per_c[c].append(d["rare"] - d["common"])
    return st.median([st.median(v) for v in per_c.values()])


for pos, exp in (("front", 0.218), ("middle", 0.207), ("back", 0.207)):
    v = tpc_delta(pos)
    ck(f"SGuard t/c 델타 ({pos} 만): +{exp}", abs(v - exp) < 5e-4, f"{v:+.3f}")

# ---------------------------------------------------------------- X.9.1
print("\n[X.9.1] 행당 처리 시간")
rt = sorted(float(r["runtime_ms"]) for r in saf if r["runtime_ms"].strip())
ck("중앙값 8.9초", abs(st.median(rt) / 1000 - 8.9) < 0.05, f"{st.median(rt)/1000:.2f}초 (n={len(rt)})")

# ---------------------------------------------------------------- X.9.6
print("\n[X.9.6] 조각 비율 (SGuard tokenizer 로드)")
import logging
logging.disable(logging.WARNING)
from transformers import AutoTokenizer  # noqa: E402  (무거운 import 는 뒤로)
from src.common import config           # noqa: E402

tok = AutoTokenizer.from_pretrained(config.SGUARD_MODEL_ID, revision=config.SGUARD_REVISION)


def frag(expr):
    ids = tok.encode(expr, add_special_tokens=False)
    bad = sum(1 for i in ids if "�" in tok.decode([i]))
    return len(ids), bad / len(ids)


n_j, f_j = frag("쥐불놀이")
n_g, f_g = frag("강강술래")
ck("쥐불놀이 4자 -> 8토큰, 조각 100.0%", n_j == 8 and abs(f_j - 1.0) < 5e-3,
   f"{n_j}토큰 {f_j*100:.1f}%")
ck("강강술래 4자 -> 7토큰, 조각 85.7%", n_g == 7 and abs(f_g - 0.857) < 5e-3,
   f"{n_g}토큰 {f_g*100:.1f}%")

keys = {}
for p in P.values():
    keys.setdefault((p["concept_id"], p["rarity_label"]), p["key_expression"])
fr = {k: frag(v)[1] for k, v in keys.items()}
cs = sorted({c for c, _ in fr})
med_r = st.median([fr[(c, "rare")] for c in cs])
med_c = st.median([fr[(c, "common")] for c in cs])
hi = sum(1 for c in cs if fr[(c, "rare")] > fr[(c, "common")])
ck("조각 비율 중앙값 희귀 59.8% / 일반 50.0%, 24개 중 17개 희귀>일반",
   abs(med_r - 0.598) < 5e-3 and abs(med_c - 0.500) < 5e-3 and hi == 17,
   f"{med_r*100:.1f}% / {med_c*100:.1f}% / {hi}/24")

safe_cs = [c for c in cs if c.startswith("SAFE")]
rank = sorted(((fr[(c, "rare")], keys[(c, "rare")]) for c in safe_cs), reverse=True)
exp_rank = [("쥐불놀이", 1.0), ("강강술래", 0.857), ("제주 해녀 물질", 0.80), ("굵고 거센 작달비", 0.727)]
ok = len(safe_cs) == 12 and all(
    rank[i][1] == nm and abs(rank[i][0] - v) < 5e-3 for i, (nm, v) in enumerate(exp_rank))
ck("안전 12개 순위: 쥐불놀이 100.0 > 강강술래 85.7 > 제주 해녀 물질 80.0 > 작달비 72.7",
   ok, " / ".join(f"{nm} {v*100:.1f}%" for v, nm in rank[:4]))

print(f"\n{SEP}")
print(f"  FAIL {len(fails)}건" + ("".join("\n    - " + f for f in fails) if fails else ""))
print(SEP)
sys.exit(1 if fails else 0)
