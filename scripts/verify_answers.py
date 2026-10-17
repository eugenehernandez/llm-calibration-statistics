"""Checks the answer key in data/questions.jsonl.

For every numeric question, the true value is computed here (exactly where
possible, by simulation otherwise). The script confirms that the keyed answer
matches it and that no distractor is equally close. Conceptual questions are
listed at the end for manual review.

Run with:  python scripts/verify_answers.py
"""
import json
import math
import pathlib
from fractions import Fraction as F
from itertools import permutations, product
from math import comb, exp

import numpy as np
from scipy import stats

rng = np.random.default_rng(2026)
N_SIM = 2_000_000


def sim_mean(fn):
    return float(np.mean(fn()))


# ---------------------------------------------------------------- true values
def m20():
    return 365 * 364 * 363 / 365 ** 3


def h01():
    # P(car behind initial pick) = 1/4. Otherwise the car is behind one of the
    # two doors left after the host opens a goat door, and we pick one at random.
    return F(3, 4) * F(1, 2)


def h03():
    kids = list(product(["B", "G"], range(7)))           # (sex, weekday)
    families = list(product(kids, kids))
    cond = [f for f in families if ("B", 2) in f]
    both = [f for f in cond if f[0][0] == "B" and f[1][0] == "B"]
    return F(len(both), len(cond))


def h04():
    # E = 2^(k+1) - 2 for k heads in a row
    return 2 ** 3 - 2


def h05():
    return 6 * (1 - (5 / 6) ** 6)


def h06():
    perms = list(permutations(range(5)))
    return F(sum(all(p[i] != i for i in range(5)) for p in perms), len(perms))


def h07():
    n, var, rho = 10, 4, 0.5
    return var / n + (n - 1) * rho * var / n


def h10():
    se = 1 / math.sqrt(25)
    crit = stats.norm.ppf(0.95) * se
    return 1 - stats.norm.cdf((crit - 0.5) / se)


def h11():
    prior_odds = 0.02 / 0.98
    lr = 0.95 / 0.10
    post_odds = prior_odds * lr ** 2
    return post_odds / (1 + post_odds)


def h12():
    return F(sum(max(a, b) for a in range(1, 7) for b in range(1, 7)), 36)


def h13():
    return sim_mean(lambda: rng.random((N_SIM, 3)).sum(axis=1) < 1)


def h14():
    return 6 * sum(1 / k for k in range(1, 7))


def h15():
    # Simulate with n = 5, sigma^2 = 1. The keyed answer gives (n-1)/n = 0.8.
    x = rng.normal(size=(400_000, 5))
    return float(np.mean(((x - x.mean(axis=1, keepdims=True)) ** 2).mean(axis=1)))


def h16():
    x, y = rng.random(N_SIM), rng.random(N_SIM)
    return float(x[x > y].mean())


def h17():
    u = np.sort(rng.random((N_SIM, 2)), axis=1)
    a, b, c = u[:, 0], u[:, 1] - u[:, 0], 1 - u[:, 1]
    return float(np.mean((a < 0.5) & (b < 0.5) & (c < 0.5)))


def h18():
    return F(3, 10)


def h19():
    x = rng.uniform(-1, 1, N_SIM)
    return float(np.cov(x, x ** 2)[0, 1])


def h20():
    # P(gold drawn) = 1/3*1 + 1/3*1/2 = 1/2. P(GG box and gold) = 1/3.
    return F(1, 3) / F(1, 2)


def h22():
    return sim_mean(lambda: np.abs(rng.random(N_SIM) - rng.random(N_SIM)))


def h23():
    return comb(4, 2) * comb(48, 3) / comb(52, 5)


TRUE = {
    "e01": 0.3 * 0.5, "e02": F(2, 6), "e03": np.mean([2, 4, 4, 5, 10]),
    "e04": float(np.median([3, 8, 1, 9, 4])), "e05": F(3, 4), "e06": 0.5,
    "e08": 3.5, "e10": 0.6, "e11": math.factorial(4), "e13": 0, "e14": 3,
    "e15": 10 * 0.3, "e18": 0.2 + 0.3, "e19": (85 - 70) / 10, "e20": F(13, 52),
    "m01": 100 * 0.01 * 0.9 / (0.01 * 0.9 + 0.99 * 0.09),
    "m02": F(6, 36), "m03": 1 - (5 / 6) ** 4, "m04": comb(5, 2) / 32,
    "m05": 4 + 9, "m06": 9 * 4, "m07": comb(8, 3), "m08": exp(-2),
    "m09": 1 - stats.norm.cdf(2), "m10": 6, "m11": 20 / math.sqrt(100),
    "m14": 0.2 / 0.4,
    "m15": math.factorial(10) // (math.factorial(3) ** 2 * math.factorial(2)),
    "m16": 0.8 ** 2, "m17": 0.6 * 10 / 5, "m19": 10 / 6 - 2, "m20": m20(),
    "m21": 1 / 4, "m22": 20 * 0.25 * 0.75,
    "h01": h01(), "h02": F(1, 3), "h03": h03(), "h04": h04(), "h05": h05(),
    "h06": h06(), "h07": h07(), "h08": 60 / 4, "h09": 1 - 0.95 ** 20,
    "h10": h10(), "h11": h11(), "h12": h12(), "h13": h13(), "h14": h14(),
    "h15": h15(), "h16": h16(), "h17": h17(), "h18": h18(), "h19": h19(),
    "h20": h20(), "h21": comb(4, 2) * 0.5 ** 5, "h22": h22(), "h23": h23(),
}

# Simulated values are checked with a looser tolerance
SIMULATED = {"h13", "h15", "h16", "h17", "h19", "h22"}
# e13 and h15 are formula answers, checked by plugging in values
FORMULA_AT = {"e13": {"0": 0.0, "c": 7.0, "c squared": 49.0, "1": 1.0},   # at c = 7
              "h15": {"(n - 1) sigma squared / n": 0.8, "sigma squared": 1.0,
                      "n sigma squared / (n - 1)": 1.25, "sigma squared / n": 0.2}}
# h19 is checked through the covariance in its keyed statement
STATEMENT_VALUE = {"h19": {0: 0.0, 1: 0.0, 2: 1 / 3, 3: float("nan")}}


def parse_choice(text):
    t = text.replace("$", "").replace("%", "").replace(",", "").strip()
    if t == "1/e":
        return 1 / math.e
    if "/" in t:
        a, b = t.split("/")
        return float(F(int(a), int(b)))
    return float(t)


def tolerance(text, simulated=False):
    """How far the keyed answer may sit from the true value.

    A rounded answer such as 0.518 or 9% may be off by half of its last digit.
    Fractions and whole-number answers must be exact. Simulated values get 0.01.
    """
    if simulated:
        return 0.01
    t = text.replace("$", "").replace("%", "").replace(",", "").strip()
    if "." in t:
        decimals = len(t.split(".")[1])
        return 0.5 * 10 ** (-decimals) + 1e-12
    if text.strip().endswith("%"):
        return 0.5 + 1e-12                   # whole percent such as "9%"
    return 1e-9


def main():
    path = pathlib.Path(__file__).resolve().parents[1] / "data" / "questions.jsonl"
    questions = [json.loads(line) for line in open(path)]
    failures, manual = [], []

    for item in questions:
        qid, choices = item["id"], item["choices"]
        if qid not in TRUE:
            manual.append(qid)
            continue
        true = float(TRUE[qid])
        if qid in FORMULA_AT:
            values = [FORMULA_AT[qid][c] for c in choices]
        elif qid in STATEMENT_VALUE:
            values = [STATEMENT_VALUE[qid][i] for i in range(4)]
        else:
            values = [parse_choice(c) for c in choices]

        dists = [abs(v - true) if not math.isnan(v) else math.inf for v in values]
        keyed = dists[0]
        tol = tolerance(choices[0], simulated=qid in SIMULATED)
        best_other = min(dists[1:])

        ok = keyed <= tol and keyed < best_other
        if qid in STATEMENT_VALUE:
            ok = keyed <= tol      # choices 0 and 1 share a value, the wording decides
        status = "ok" if ok else "FAIL"
        if not ok:
            failures.append(qid)
        print(f"{qid}  {status:4}  true = {true:.6g}   keyed = {choices[0]!r}")

    print(f"\n{len(questions) - len(manual)} numeric answers checked, {len(failures)} failures.")
    if manual:
        print("Conceptual questions to review by hand:", ", ".join(manual))
    if failures:
        raise SystemExit(f"Failed: {failures}")


if __name__ == "__main__":
    main()
