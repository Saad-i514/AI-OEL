"""Part E: 30 generated scenarios (seeded from the CMS IDs), mean +/- 95% CI, ablation, failure analysis.
Scenario = same city (grid, one-way streets, charging stations, fuel) but a random start cell and 4 random delivery stops.
Run: python -m src.experiments
"""
import os, copy, hashlib, math, time, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from members import STAMP, CMS_IDS
from src.twists import build_world
from src.problem import DeliveryProblem
from src.search import astar, ucs, greedy
from src.heuristics import h_farthest, h_own
from src.game import GameWorld

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
N_SCEN, N_STOPS = 30, 4


def mean_ci(x):
    x = np.array(x, dtype=float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def make_scenarios(w, n=N_SCEN, k=N_STOPS):
    key = "|".join(sorted(x.strip().upper() for x in CMS_IDS)) + "|scenarios"
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    gw = GameWorld(w)
    safe = [gw.rc[i] for i in range(gw.N) if gw.safe[i]]            # cells inside the start's strongly connected zone
    out = []
    while len(out) < n:
        pick = rng.choice(len(safe), k + 1, replace=False)
        cells = [tuple(safe[i]) for i in pick]
        sw = copy.copy(w); sw.start = cells[0]; sw.stops = cells[1:]
        out.append(sw)
    return out


def run_algo(name, P):
    t0 = time.perf_counter()
    if name == "UCS":                      r = ucs(P)
    elif name == "A* Manhattan farthest":  r = astar(P, h_farthest(P))
    elif name == "A* OWN":                 r = astar(P, h_own(P))
    elif name == "Weighted A* (W=2)":      r = astar(P, h_own(P), weight=2.0)
    elif name == "OWN - MST term":         r = astar(P, h_own(P, False, True))
    elif name == "OWN - fuel term":        r = astar(P, h_own(P, True, False))
    elif name == "OWN - both":             r = astar(P, h_own(P, False, False))
    r["ms"] = (time.perf_counter() - t0) * 1e3
    return r


def run_experiments(n_scen=N_SCEN):
    print(STAMP)
    w = build_world(); scen = make_scenarios(w, n=n_scen)
    algos = ["UCS", "A* Manhattan farthest", "A* OWN", "Weighted A* (W=2)"]
    ablation = ["A* OWN", "OWN - MST term", "OWN - fuel term", "OWN - both"]
    names = list(dict.fromkeys(algos + ablation)); R = {n: [] for n in names}
    unsolved = 0
    for sw in scen:
        P = DeliveryProblem(sw)
        res = {n: run_algo(n, P) for n in names}
        if not res["UCS"]["solved"]:
            unsolved += 1; continue
        for n in names:
            R[n].append(res[n])
    n_ok = len(R["UCS"])
    print(f"\n{n_scen} scenarios (random start + {N_STOPS} random stops, same city). Solvable: {n_ok}.  mean +/- 95% CI\n{STAMP}")
    print(f"{'algorithm':<24}{'cost':>14}{'nodes expanded':>20}{'runtime ms':>16}{'peak stored':>16}")
    rows = []
    for n in algos:
        c, nodes, ms, pk = (mean_ci([r[k] for r in R[n]]) for k in ("cost", "expanded", "ms", "peak_stored"))
        print(f"{n:<24}{c[0]:>8.1f}+-{c[1]:<5.1f}{nodes[0]:>12.0f}+-{nodes[1]:<6.0f}{ms[0]:>9.1f}+-{ms[1]:<5.1f}{pk[0]:>9.0f}+-{pk[1]:<5.0f}")
        rows.append((n, *c, *nodes, *ms, *pk))
    print(f"\nABLATION of the heuristic (same scenarios)\n{STAMP}")
    print(f"{'variant':<24}{'nodes expanded':>20}{'runtime ms':>16}{'vs full':>10}")
    base = np.mean([r["expanded"] for r in R["A* OWN"]]); arows = []
    for n in ablation:
        nodes = mean_ci([r["expanded"] for r in R[n]]); ms = mean_ci([r["ms"] for r in R[n]])
        print(f"{n:<24}{nodes[0]:>12.0f}+-{nodes[1]:<6.0f}{ms[0]:>9.1f}+-{ms[1]:<5.1f}{nodes[0]/base:>9.2f}x")
        arows.append((n, *nodes, *ms))
    # ---- failure analysis
    opt = [r["cost"] for r in R["UCS"]]
    wa = [r["cost"] for r in R["Weighted A* (W=2)"]]
    sub = [(a - b) / b * 100 for a, b in zip(wa, opt) if a > b]
    own_e = [r["expanded"] for r in R["A* OWN"]]; man_e = [r["expanded"] for r in R["A* Manhattan farthest"]]
    worse = [i for i in range(n_ok) if own_e[i] > man_e[i]]
    bad_opt = [i for i in range(n_ok) if R["A* OWN"][i]["cost"] != opt[i]]
    print(f"\nFAILURE ANALYSIS\n{STAMP}")
    print(f"  unsolvable scenarios discarded: {unsolved} of {n_scen}")
    print(f"  A* OWN returned a non-optimal cost in {len(bad_opt)} of {n_ok} scenarios (must be 0: heuristic is admissible)")
    print(f"  weighted A* (W=2) suboptimal in {len(sub)} of {n_ok} scenarios; mean gap when suboptimal "
          f"{(np.mean(sub) if sub else 0):.1f}%, worst {(max(sub) if sub else 0):.1f}%")
    print(f"  scenarios where OWN expanded MORE nodes than Manhattan-farthest: {len(worse)} of {n_ok}")
    if worse:
        i = max(worse, key=lambda j: own_e[j] / max(man_e[j], 1))
        print(f"     worst case: scenario {i}: OWN {own_e[i]} vs Manhattan {man_e[i]} (extra table lookups cost, bound no tighter)")
    sp = [R["UCS"][i]["expanded"] / max(own_e[i], 1) for i in range(n_ok)]
    print(f"  node reduction of OWN vs UCS: mean {np.mean(sp):.1f}x, min {min(sp):.1f}x, max {max(sp):.1f}x")
    # ---- files
    with open(os.path.join(RESULTS, "partE_scenarios.csv"), "w", newline="") as f:
        f.write(f"# {STAMP}\n# {n_ok} scenarios, mean and 95% CI half-width\n")
        wr = csv.writer(f); wr.writerow(["algorithm", "cost", "cost_ci", "nodes", "nodes_ci", "ms", "ms_ci", "peak", "peak_ci"])
        wr.writerows([[r[0]] + [f"{v:.2f}" for v in r[1:]] for r in rows])
        wr.writerow([]); wr.writerow(["ablation", "nodes", "nodes_ci", "ms", "ms_ci"])
        wr.writerows([[r[0]] + [f"{v:.2f}" for v in r[1:]] for r in arows])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in ((axs[0], rows, "Algorithms"), (axs[1], arows, "Ablation of h_own")):
        ax.bar([r[0] for r in data], [r[3] if title == "Algorithms" else r[1] for r in data],
               yerr=[r[4] if title == "Algorithms" else r[2] for r in data], capsize=4, color="#5c8dd6")
        ax.set_ylabel("nodes expanded (mean, 95% CI)"); ax.set_title(title); ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    fig.suptitle(f"{n_ok} scenarios\n{STAMP}", fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "partE_scenarios.png"), dpi=150); plt.close(fig)
    print(f"\nSaved results/partE_scenarios.csv and partE_scenarios.png\n{STAMP}")


if __name__ == "__main__":
    run_experiments()
