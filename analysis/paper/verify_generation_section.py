# -*- coding: utf-8 -*-
"""논문 X.10(H6 — 가시성과 생성 결과의 연관)에 인용된 수치를 원본에서 재계산해 대조한다.

왜 별도 파일인가
----------------
X.10 은 팀원 두 명의 산출물에 동시에 의존한다 — 학생 3 의 generation_results.csv 와
image_labels.csv. 둘 중 하나라도 다시 만들어지면 이 절의 모든 수치가 바뀔 수 있다.
X.9 가 safety_results.csv 재실행으로 두 번 어긋났던 것과 같은 구조다.

학생 5 의 defense/analyze_labels.py 와 겹치는 값(평가자 일치도, 필터×이미지 2x2,
안전 한정 P)은 여기서도 계산해 양쪽이 같은 숫자를 말하는지 확인한다. 두 사람이
각자 계산해 논문에 다른 값이 들어가는 것을 막는 것이 목적이다.

    python analysis/paper/verify_generation_section.py

기대값은 논문 본문 값을 하드코딩한다. FAIL 이면 데이터가 아니라 논문 문장을 고친다.
"""
import csv
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

R = Path(__file__).resolve().parents[2]


def rd(p):
    if not p.exists():
        sys.exit(f"입력 파일이 없습니다: {p.relative_to(R)}\n"
                 f"tokenization_results.csv 는 analyze_tokens.py --full --overwrite 로 만듭니다.")
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def n(v):
    return str(v).strip().lower()


P = {r["prompt_id"]: r for r in rd(R / "benchmarks/prompts/prompts.csv")}
saf = {(r["prompt_id"], r["input_policy"]): r
       for r in rd(R / "evaluation/safety/safety_results.csv")}
gen = rd(R / "evaluation/generation/generation_results.csv")
labrows = rd(R / "evaluation/generation/image_labels.csv")
tok = rd(R / "analysis/truncation/tokenization_results.csv")

g2p = {r["generation_id"]: r["prompt_id"] for r in gen}
L = {g2p[r["generation_id"]]: r for r in labrows}
AD = {r["prompt_id"]: r for r in tok if r["model_role"] == "generator"}

cp = lambda p: n(L[p]["concept_present_final"]) == "true"
iu = lambda p: n(L[p]["image_safety_final"]) == "unsafe"
blk = lambda p, pol="native": n(saf[(p, pol)]["decision"]) == "unsafe"

fails = []


def ck(claim, ok, got=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {claim}" + (f"   -> {got}" if got else ""))
    if not ok:
        fails.append(claim)


def frac(ids, f):
    return sum(1 for p in ids if f(p)), len(ids)


SEP = "=" * 92
print(SEP)
print("X.10 (H6) 인용 수치 전수 재검증")
print(SEP)

# ---------------------------------------------------------------- X.10.1
print("\n[X.10.1] 라벨 데이터")
ck("생성 432건, 오류 0건, seed 42 단일",
   len(gen) == 432 and not any(r["error_type"].strip() for r in gen)
   and {r["seed"] for r in gen} == {"42"},
   f"{len(gen)}건")
ck("라벨 432행이 generation_id 로 전부 조인", len(L) == 432, f"{len(L)}행")

for col, ea, ek in (("concept_present", 398, 0.755), ("image_safety", 417, 0.804)):
    a1 = [n(r[f"{col}_a1"]) for r in labrows]
    a2 = [n(r[f"{col}_a2"]) for r in labrows]
    ag = sum(1 for x, y in zip(a1, a2) if x == y)
    cats = sorted(set(a1) | set(a2))
    pe = sum((a1.count(c) / len(a1)) * (a2.count(c) / len(a2)) for c in cats)
    k = (ag / len(a1) - pe) / (1 - pe)
    ck(f"{col}: 일치 {ea}/432, kappa {ek}",
       ag == ea and abs(k - ek) < 5e-4, f"{ag}/432, kappa {k:.3f}")

res = Counter(n(r["disagreement_resolved_by"]) for r in labrows)
ck("불일치 해소 consensus 387 / lead_review 45",
   res["consensus"] == 387 and res["lead_review"] == 45, str(dict(res)))

# ---------------------------------------------------------------- X.10.2
print("\n[X.10.2] 가시성별 개념 등장률")
for v, en_, er in (("full", 317, 0.300), ("partial", 19, 0.053), ("none", 96, 0.000)):
    ids = [p for p in L if AD[p]["key_visibility"] == v]
    t, m = frac(ids, cp)
    ck(f"{v}: n={en_}, 등장률 {er:.3f}",
       m == en_ and abs(t / m - er) < 5e-4, f"n={m}, {t}/{m}={t/m:.3f}")
ck("none 에서는 한 건도 그려지지 않음",
   frac([p for p in L if AD[p]["key_visibility"] == "none"], cp)[0] == 0)

EXP_GRID = {("short", "front"): 18, ("short", "middle"): 16, ("short", "back"): 18,
            ("near_limit", "front"): 14, ("near_limit", "middle"): 12, ("near_limit", "back"): 6,
            ("over_limit", "front"): 12, ("over_limit", "middle"): 0, ("over_limit", "back"): 0}
bad = []
for (lv, ps), e in EXP_GRID.items():
    ids = [p for p in L if P[p]["length_level"] == lv and P[p]["position_level"] == ps]
    t, m = frac(ids, cp)
    if not (m == 48 and t == e):
        bad.append(f"{lv}x{ps} {t}/{m} != {e}/48")
ck("길이 x 위치 표 9칸 전부 일치", not bad, "; ".join(bad))

# ---------------------------------------------------------------- X.10.3
print("\n[X.10.3] 희귀도별 개념 등장률")
for r_, en_, er in (("common", 216, 0.380), ("rare", 216, 0.065)):
    ids = [p for p in L if P[p]["rarity_label"] == r_]
    t, m = frac(ids, cp)
    ck(f"{r_} 전체: n={en_}, {er:.3f}", m == en_ and abs(t / m - er) < 5e-4,
       f"n={m}, {t}/{m}={t/m:.3f}")
for r_, en_, er in (("common", 158, 0.513), ("rare", 159, 0.088)):
    ids = [p for p in L if P[p]["rarity_label"] == r_ and AD[p]["key_visibility"] == "full"]
    t, m = frac(ids, cp)
    ck(f"{r_} 가시성 full 한정: n={en_}, {er:.3f}",
       m == en_ and abs(t / m - er) < 5e-4, f"n={m}, {t}/{m}={t/m:.3f}")

pair = defaultdict(dict)
for p in L:
    pair[(P[p]["concept_id"], P[p]["length_level"], P[p]["position_level"])][P[p]["rarity_label"]] = p
d = [int(cp(v["common"])) - int(cp(v["rare"])) for v in pair.values() if len(v) == 2]
ck("216쌍: 일반만 68 / 희귀만 0 / 동일 148",
   len(d) == 216 and d.count(1) == 68 and d.count(-1) == 0 and d.count(0) == 148,
   f"{len(d)}쌍, {d.count(1)} / {d.count(-1)} / {d.count(0)}")

# ---------------------------------------------------------------- X.10.4
print("\n[X.10.4] H2a")
unsafe_ids = [p for p in L if n(P[p]["safety_label"]) == "unsafe"]
ck("유해 라벨 216건", len(unsafe_ids) == 216, str(len(unsafe_ids)))
for pol, eu, ev, ec, eh in (("native", 188, 135, 22, 17),
                            ("constrained_77", 189, 136, 22, 17)):
    u = [p for p in unsafe_ids if not blk(p, pol)]
    vf = [p for p in u if AD[p]["key_visibility"] == "full"]
    rz = [p for p in vf if cp(p)]
    hz = [p for p in rz if iu(p)]
    ck(f"{pol}: 미차단 {eu} / full {ev} / 등장 {ec} / 유해 {eh}",
       (len(u), len(vf), len(rz), len(hz)) == (eu, ev, ec, eh),
       f"{len(u)} / {len(vf)} / {len(rz)} / {len(hz)}")
ck("17건은 전체 432의 3.9%", abs(17 / 432 * 100 - 3.9) < 0.05, f"{17/432*100:.1f}%")

for nm, ids, er in (("필터 통과", [p for p in unsafe_ids if not blk(p)], 0.133),
                    ("필터 차단", [p for p in unsafe_ids if blk(p)], 0.357)):
    t, m = frac(ids, iu)
    ck(f"유해 라벨 중 {nm}: 이미지 유해 {er:.3f}", abs(t / m - er) < 5e-4,
       f"{t}/{m}={t/m:.3f}")

# ---------------------------------------------------------------- X.10.5
print("\n[X.10.5] 필터 판정 x 이미지 위해도")
c = Counter((blk(p), iu(p)) for p in L)
ck("2x2 = 370 / 31 / 21 / 10",
   (c[(False, False)], c[(False, True)], c[(True, False)], c[(True, True)]) == (370, 31, 21, 10),
   f"{c[(False,False)]} / {c[(False,True)]} / {c[(True,False)]} / {c[(True,True)]}")
mis = c[(False, True)] + c[(True, False)]
ck("불일치 52건 = 12.0%", mis == 52 and abs(mis / 432 * 100 - 12.0) < 0.05,
   f"{mis}/432 = {mis/432*100:.1f}%")

# ---------------------------------------------------------------- X.10.6
print("\n[X.10.6] 종합 표")
ids = [p for p in L if P[p]["rarity_label"] == "common" and AD[p]["key_visibility"] == "full"]
tc, mc = frac(ids, cp)
ids = [p for p in L if P[p]["rarity_label"] == "rare" and AD[p]["key_visibility"] == "full"]
tr, mr = frac(ids, cp)
ck("일반 51.3% / 희귀 8.8% / 5.8배",
   abs(tc / mc * 100 - 51.3) < 0.05 and abs(tr / mr * 100 - 8.8) < 0.05
   and abs((tc / mc) / (tr / mr) - 5.8) < 0.05,
   f"{tc/mc*100:.1f}% / {tr/mr*100:.1f}% / {(tc/mc)/(tr/mr):.1f}배")

# ---------------------------------------------------------------- 학생5 대조
print("\n[대조] 학생 5 label_report.json 과 공유하는 값")
rep = R / "defense" / "phase5" / "label_report.json"
if not rep.exists():
    print("  건너뜀 — defense/phase5/label_report.json 이 없다")
else:
    import json
    hw = json.loads(rep.read_text(encoding="utf-8"))
    if hw.get("validation", {}).get("complete") is not True:
        print("  건너뜀 — 학생 5 리포트가 라벨 도착 전 상태다(재실행 필요).")
    else:
        for col in ("concept_present", "image_safety"):
            a1 = [n(r[f"{col}_a1"]) for r in labrows]
            a2 = [n(r[f"{col}_a2"]) for r in labrows]
            ag = sum(1 for x, y in zip(a1, a2) if x == y)
            h = hw["agreement"][col]
            ck(f"{col} 일치 건수", h["n_agree"] == ag, f"학생5 {h['n_agree']} / 내 {ag}")
        conf = hw["harm"]["confusion"]
        ck("필터x이미지 2x2 일치",
           (conf["filter_safe__image_safe"], conf["filter_safe__image_unsafe"],
            conf["filter_unsafe__image_safe"], conf["filter_unsafe__image_unsafe"])
           == (c[(False, False)], c[(False, True)], c[(True, False)], c[(True, True)]))
        safe_ids = [p for p in L if n(P[p]["safety_label"]) == "safe"]
        t, m = frac(safe_ids, cp)
        ck("안전 한정 P 일치 (과탐과 같은 모집단)",
           abs(hw["utility"]["safe_only"]["p"] - t / m) < 5e-4,
           f"학생5 {hw['utility']['safe_only']['p']} / 내 {t/m:.4f}")

print(f"\n{SEP}")
print(f"  FAIL {len(fails)}건" + ("".join("\n    - " + f for f in fails) if fails else ""))
print(SEP)
sys.exit(1 if fails else 0)
