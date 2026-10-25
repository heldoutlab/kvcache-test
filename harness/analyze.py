#!/opt/homebrew/bin/python3.12
"""Score the KV-cache runs and apply the pre-registered rule mechanically. Written before any model was run.

  analyze.py                       # public set, model glimmer -> results/glimmer/public/report.json
  analyze.py --set private         # descriptive only
  analyze.py --selftest            # synthetic data, no model output needed

Unit of analysis: one question. Questions on the same document are correlated, so every interval is a
paired bootstrap that resamples whole documents within each context length (stratified cluster bootstrap).
"""
import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def load(model, set_name, run, arm):
    path = os.path.join(cfg.RESULTS_DIR, model, set_name, run, "%s.jsonl" % arm)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["qid"]: r for r in map(json.loads, filter(str.strip, f))}


def pct(xs):
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def acc(recs, qids):
    return pct([1.0 if recs[q]["correct"] else 0.0 for q in qids])


def clusters(recs, qids):
    """{ctx: {doc_id: [qids]}}"""
    out = {}
    for q in qids:
        r = recs[q]
        out.setdefault(r["ctx"], {}).setdefault(r["doc_id"], []).append(q)
    return out


def paired(x, y, qids, seed=0):
    """Accuracy of x and y (points), difference x - y, 95% stratified cluster-bootstrap interval, discordant counts."""
    if not qids:
        return None
    cl = clusters(x, qids)
    sx = {q: 1.0 if x[q]["correct"] else 0.0 for q in qids}
    sy = {q: 1.0 if y[q]["correct"] else 0.0 for q in qids}
    rng = random.Random(seed)
    diffs = []
    strata = [list(docs.values()) for docs in cl.values()]
    for _ in range(cfg.BOOT):
        sample = []
        for docs in strata:
            for _ in docs:
                sample += rng.choice(docs)
        diffs.append(pct([sx[q] for q in sample]) - pct([sy[q] for q in sample]))
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    ax, ay = pct(list(sx.values())), pct(list(sy.values()))
    return {"n": len(qids), "docs": sum(len(d) for d in cl.values()), "x": ax, "y": ay, "diff": ax - ay,
            "ci": [lo, hi], "x_only": sum(1 for q in qids if sx[q] > sy[q]),
            "y_only": sum(1 for q in qids if sy[q] > sx[q])}


def compare(x, y, qids):
    out = {"pooled": paired(x, y, qids)}
    for ctx in cfg.CONTEXTS:
        sub = [q for q in qids if x[q]["ctx"] == ctx]
        if sub:
            out[ctx] = paired(x, y, sub)
    return out


def describe(recs):
    qids = list(recs)
    by_ctx = {c: [q for q in qids if recs[q]["ctx"] == c] for c in cfg.CONTEXTS}
    fams = sorted({recs[q]["family"] for q in qids})
    out = {"n": len(qids), "accuracy": acc(recs, qids),
           "by_context": {c: acc(recs, v) for c, v in by_ctx.items() if v},
           "by_family": {f: acc(recs, [q for q in qids if recs[q]["family"] == f]) for f in fams},
           "by_context_family": {c: {f: acc(recs, [q for q in v if recs[q]["family"] == f]) for f in fams}
                                 for c, v in by_ctx.items() if v},
           "retrieval_partial": pct([recs[q]["grader"].get("partial", 0.0) for q in qids if recs[q]["family"] == "R"]),
           "truncated": sum(1 for r in recs.values() if r.get("truncated")),
           "server_errors": sum(1 for r in recs.values() if r.get("server_error")),
           "no_answer_line": sum(1 for r in recs.values() if not r.get("server_error")
                                 and r["grader"].get("extracted") is None)}
    # speed: full-document prefills are the requests that processed most of the prompt
    speed = {}
    for c, v in by_ctx.items():
        t = [recs[q]["turn"].get("timings", {}) for q in v if not recs[q].get("server_error")]
        full = [x for x in t if x.get("prompt_n", 0) >= 0.5 * cfg.NOMINAL_TOKENS[c]]  # whole document read
        reuse = [x for x in t if 0 < x.get("prompt_n", 0) <= 1000]
        speed[c] = {
            "prefill_tok_s_median": statistics.median([x["prompt_per_second"] for x in full]) if full else None,
            "prefill_s_median": statistics.median([x["prompt_ms"] / 1000 for x in full]) if full else None,
            "decode_tok_s_median": statistics.median([x["predicted_per_second"] for x in t if x.get("predicted_n")])
            if t else None,
            "gen_tokens_median": statistics.median([x.get("predicted_n", 0) for x in t]) if t else None,
            "full_prefills": len(full), "cache_reused": len(reuse), "requests": len(t)}
    out["speed"] = speed
    out["hours"] = sum(r["turn"].get("wall_s", 0) for r in recs.values()) / 3600.0
    rss = [r["rss_mib"] for r in recs.values() if r.get("rss_mib")]
    out["rss_mib_max"] = max(rss) if rss else None
    return out


# ---------- candidate rules (PREREG.md); the author picks one in config.RULE before freezing ----------

def rule_1(cmp_ab, cmp_cd, floor):
    """Threshold rule per context (mirrors the Bonsai 2 rule)."""
    per = [cmp_ab[c] for c in cfg.CONTEXTS if c in cmp_ab]
    if any(p["diff"] >= 10 and p["ci"][0] > 0 for p in per):
        part1 = "NOT FREE"
    elif all(p["diff"] <= 5 for p in per) and cmp_ab["pooled"]["diff"] <= 3:
        part1 = "NO LOSS FOUND"
    else:
        part1 = "NOT RESOLVED"
    if cmp_cd is None:
        return part1, PART2_NOT_RUN
    g = cmp_cd["pooled"]
    if g["diff"] >= 5 and g["ci"][0] > 0:
        part2 = "SUPPORTED (V-first beats K-first)"
    elif g["diff"] <= -5 and g["ci"][1] < 0:
        part2 = "CONTRADICTED (K-first beats V-first)"
    elif abs(g["diff"]) <= 3:
        part2 = "NO DIFFERENCE FOUND"
    else:
        part2 = "NOT RESOLVED"
    return part1, part2


def rule_2(cmp_ab, cmp_cd, floor):
    """Noise-floor rule: compare against arm A re-run with another seed (minimum floor 2 points)."""
    f = max(2.0, floor)
    per = [cmp_ab[c] for c in cfg.CONTEXTS if c in cmp_ab]
    if any(p["diff"] > 2 * f and p["ci"][0] > 0 for p in per):
        part1 = "NOT FREE"
    elif all(abs(p["diff"]) <= f for p in per):
        part1 = "WITHIN NOISE"
    else:
        part1 = "NOT RESOLVED"
    if cmp_cd is None:
        return part1, PART2_NOT_RUN
    g = cmp_cd["pooled"]
    if g["diff"] > f and g["ci"][0] > 0:
        part2 = "SUPPORTED (V-first beats K-first)"
    elif g["diff"] < -f and g["ci"][1] < 0:
        part2 = "CONTRADICTED (K-first beats V-first)"
    elif abs(g["diff"]) <= f:
        part2 = "NO DIFFERENCE FOUND"
    else:
        part2 = "NOT RESOLVED"
    return part1, part2


def rule_3(cmp_ab, cmp_cd, floor, cmp_ac=None):
    """Pooled rule with a 48k guard and an equivalence-style 'free'."""
    p, l48 = cmp_ab["pooled"], cmp_ab.get("48k")
    if (p["diff"] >= 5 and p["ci"][0] > 0) or (l48 and l48["diff"] >= 10 and l48["ci"][0] > 0):
        part1 = "NOT FREE"
    elif -5 <= p["ci"][0] and p["ci"][1] <= 5:
        part1 = "FREE WITHIN 5 POINTS"
    else:
        part1 = "NOT RESOLVED"
    if cmp_cd is None:
        return part1, PART2_NOT_RUN
    g = cmp_cd["pooled"]
    c_ok = cmp_ac is None or cmp_ac["pooled"]["diff"] <= 5
    if g["diff"] >= 5 and g["ci"][0] > 0 and c_ok:
        part2 = "SUPPORTED (V-first beats K-first, and costs at most 5 points vs f16)"
    elif g["diff"] <= -5 and g["ci"][1] < 0:
        part2 = "CONTRADICTED (K-first beats V-first)"
    else:
        part2 = "NOT RESOLVED"
    return part1, part2


RULES = {"1": rule_1, "2": rule_2, "3": rule_3}
PART2_NOT_RUN = "NOT TESTED (arms C and D not run on this model)"


def above_floor(cmp_ab):
    """Contexts where arm A scores at least cfg.FLOOR_ACC; below that a loss cannot show (PREREG.md)."""
    return [c for c in cfg.CONTEXTS if c in cmp_ab and cmp_ab[c]["x"] >= cfg.FLOOR_ACC]


def noise_floor(main_a, noise_a):
    qids = [q for q in noise_a if q in main_a]
    if not qids:
        return None
    p = paired(main_a, noise_a, qids)
    p["floor"] = abs(p["diff"])
    return p


def report(model, set_name):
    runs = {a: load(model, set_name, "main", a) for a in cfg.ARMS}
    runs = {a: r for a, r in runs.items() if r}
    out = {"model": model, "set": set_name, "arms": {a: describe(r) for a, r in runs.items()}}
    for a, r in runs.items():
        out["arms"][a]["label"] = cfg.ARMS[a]["label"]
        out["arms"][a]["kv_mib_analytic"] = cfg.kv_bytes(model, cfg.ARMS[a]["ctk"], cfg.ARMS[a]["ctv"]) / 2 ** 20
    A = runs.get("A")
    if A:
        for a in runs:
            if a != "A":
                common = [q for q in A if q in runs[a]]
                out["A_minus_%s" % a] = compare(A, runs[a], common)
                # slowdowns that suggest attention left the GPU
                for c in cfg.CONTEXTS:
                    sa = out["arms"]["A"]["speed"].get(c, {}).get("prefill_tok_s_median")
                    sx = out["arms"][a]["speed"].get(c, {}).get("prefill_tok_s_median")
                    if sa and sx:
                        out.setdefault("prefill_speed_vs_A", {}).setdefault(a, {})[c] = sx / sa
    if "C" in runs and "D" in runs:
        common = [q for q in runs["C"] if q in runs["D"]]
        out["C_minus_D"] = compare(runs["C"], runs["D"], common)
    noise = load(model, set_name, "noise", "A")
    nf = noise_floor(A, noise) if A and noise else None
    out["noise_floor"] = nf
    if set_name == "public" and "A_minus_B" in out:
        if cfg.RULE in RULES:
            floor = nf["floor"] if nf else 0.0
            # the verdict uses only contexts where arm A is above the floor (PREREG.md, pilot amendment)
            keep = above_floor(out["A_minus_B"])
            out["floor_contexts"] = [c for c in cfg.CONTEXTS if c in out["A_minus_B"] and c not in keep]
            if not keep:
                out["verdict"] = {"rule": cfg.RULE, "q8_0_is_free": "NOT TESTABLE (arm A below the floor at every context)",
                                  "quantize_V_not_K": "NOT TESTABLE"}
                return out
            def sub(x, y):
                return compare(x, y, [q for q in x if q in y and x[q]["ctx"] in keep])
            ab = sub(A, runs["B"])
            cd = sub(runs["C"], runs["D"]) if "C" in runs and "D" in runs else None
            args = (ab, cd, floor)
            if cfg.RULE == "3":
                part1, part2 = rule_3(*args, cmp_ac=sub(A, runs["C"]) if "C" in runs else None)
            else:
                part1, part2 = RULES[cfg.RULE](*args)
            out["verdict"] = {"rule": cfg.RULE, "contexts_used": keep, "q8_0_is_free": part1, "quantize_V_not_K": part2}
        else:
            out["verdict"] = "no rule chosen in config.RULE - descriptive results only"
    return out


def write(model, set_name, rep):
    path = os.path.join(cfg.RESULTS_DIR, model, set_name, "report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    return path


def selftest():
    """Synthetic records: arm B identical to A, arm C loses retrieval at 48k, arm D loses more."""
    rng = random.Random(1)
    base = {}
    for ctx in cfg.CONTEXTS:
        for d in range(6):
            for k, fam in enumerate(["R", "R", "M3", "M2", "AS", "AC", "L", "L"]):
                qid = "pub-%s-%02d-%s%d" % (ctx, d, fam, k)
                base[qid] = {"qid": qid, "ctx": ctx, "doc_id": "pub-%s-%02d" % (ctx, d), "family": fam,
                             "correct": rng.random() < 0.8, "grader": {"extracted": "x", "partial": 1.0},
                             "turn": {"timings": {"prompt_n": 100, "prompt_ms": 1000, "prompt_per_second": 50,
                                                  "predicted_n": 300, "predicted_per_second": 6}, "wall_s": 60}}
    def arm(p_flip_48k):
        out = {}
        for q, r in base.items():
            r2 = json.loads(json.dumps(r))
            if r["ctx"] == "48k" and r2["correct"] and rng.random() < p_flip_48k:
                r2["correct"] = False
            out[q] = r2
        return out
    A, B, C, D = base, arm(0.0), arm(0.15), arm(0.6)
    ab, cd = compare(A, B, list(A)), compare(C, D, list(C))
    assert ab["pooled"]["diff"] == 0 and ab["pooled"]["ci"] == [0.0, 0.0]
    assert cd["48k"]["diff"] > 10 and cd["48k"]["ci"][0] > 0, cd["48k"]
    assert rule_1(ab, cd, 0.0) == ("NO LOSS FOUND", "SUPPORTED (V-first beats K-first)")
    assert rule_2(ab, cd, 0.0)[0] == "WITHIN NOISE"
    assert rule_3(ab, cd, 0.0, compare(A, C, list(A)))[0] == "FREE WITHIN 5 POINTS"
    assert rule_1(ab, None, 0.0) == ("NO LOSS FOUND", PART2_NOT_RUN)
    assert above_floor(ab) == list(cfg.CONTEXTS)
    low = {q: dict(r, correct=False) if r["ctx"] == "48k" else r for q, r in A.items()}
    assert above_floor(compare(low, low, list(low))) == [c for c in cfg.CONTEXTS if c != "48k"]
    d = describe(A)
    assert d["n"] == 144 and set(d["by_context"]) == set(cfg.CONTEXTS)
    return "analyze selftest OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glimmer", choices=sorted(cfg.MODELS))
    ap.add_argument("--set", default="public", choices=["public", "private", "calibration"])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(selftest())
        return
    rep = report(args.model, args.set)
    path = write(args.model, args.set, rep)
    print(json.dumps(rep, indent=2))
    print("written:", path)


if __name__ == "__main__":
    main()
