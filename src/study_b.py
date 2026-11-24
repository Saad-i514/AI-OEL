"""Part B experiments: heuristic comparison, empirical consistency/admissibility, ablation, inadmissible variant.
Run: python -m src.study_b"""
import os
from members import STAMP
from src.twists import build_world
from src.problem import DeliveryProblem, path_cost
from src.search import ucs, astar, best_first, measure, verify, print_table, save_csv, RESULTS
from src.heuristics import h_zero, h_nearest, h_farthest, h_own


def consistency_report(P, hs):
    """Expand every state UCS expands; test h(s) <= c + h(s') on ALL their outgoing edges, and
    h(s) <= remaining cost along the optimal plan (admissibility check)."""
    trace = []
    r = best_first(P, "ucs", name="UCS", trace=trace)
    Cstar = r["cost"]
    out = {}
    for name, h in hs.items():
        edges = viol = 0; worst = 0
        for s, g in trace:
            hs_ = h(s)
            for a, ns, c in P.successors(s):
                edges += 1
                gap = hs_ - (c + h(ns))
                if gap > 1e-9:
                    viol += 1; worst = max(worst, gap)
        # admissibility along the optimal path
        s = P.initial_state(); g = 0; adm_viol = 0
        for a in r["actions"]:
            if h(s) > Cstar - g + 1e-9:
                adm_viol += 1
            for act, ns, c in P.successors(s):
                if act == a:
                    s = ns; g += c; break
        out[name] = (edges, viol, worst, adm_viol)
    return out, len(trace), Cstar


def run_heuristic_study():
    print(STAMP)
    w = build_world(); P = DeliveryProblem(w)
    hs = {"h=0 (UCS)": h_zero(P), "Manhattan*cmin nearest": h_nearest(P),
          "Manhattan*cmin farthest": h_farthest(P), "OWN (full)": h_own(P)}
    # 1) comparison
    rows = [measure(lambda: ucs(P, name="UCS (h=0)")),
            measure(lambda: astar(P, h_nearest(P), name="A* Manhattan nearest")),
            measure(lambda: astar(P, h_farthest(P), name="A* Manhattan farthest")),
            measure(lambda: astar(P, h_own(P), name="A* OWN"))]
    for r in rows: verify(P, r)
    print_table(rows, "HEURISTIC COMPARISON (full problem)")
    save_csv(rows, os.path.join(RESULTS, "partB_heuristic_comparison.csv"), "heuristic comparison")
    # 2) consistency / admissibility
    rep, nstates, Cstar = consistency_report(P, hs)
    print(f"\nCONSISTENCY / ADMISSIBILITY TEST over {nstates} expanded states (optimal cost {Cstar})\n{STAMP}")
    print(f"{'heuristic':<26}{'edges':>9}{'violations':>12}{'worst gap':>11}{'adm. viol. on opt path':>24}")
    with open(os.path.join(RESULTS, "partB_consistency.csv"), "w") as f:
        f.write(f"# {STAMP}\nheuristic,edges,consistency_violations,worst_gap,admissibility_violations_on_optimal_path\n")
        for n, (e, v, wg, av) in rep.items():
            print(f"{n:<26}{e:>9}{v:>12}{wg:>11.1f}{av:>24}")
            f.write(f"{n},{e},{v},{wg},{av}\n")
    # 3) ablation
    ab = [measure(lambda: astar(P, h_own(P, True, True), name="OWN full")),
          measure(lambda: astar(P, h_own(P, False, True), name="OWN - MST term")),
          measure(lambda: astar(P, h_own(P, True, False), name="OWN - fuel term")),
          measure(lambda: astar(P, h_own(P, False, False), name="OWN - both"))]
    for r in ab: verify(P, r)
    print_table(ab, "ABLATION of h_own")
    save_csv(ab, os.path.join(RESULTS, "partB_ablation.csv"), "ablation of own heuristic")
    # 4) inadmissible faster variant: weighted A*, f = g + W*h
    wa = [measure(lambda: astar(P, h_own(P), name="A* OWN (W=1, optimal)"))]
    for W in (1.5, 2, 3, 5):
        wa.append(measure(lambda W=W: astar(P, h_own(P), weight=W, name=f"Weighted A* W={W}")))
    for r in wa: verify(P, r)
    print_table(wa, "INADMISSIBLE VARIANT: weighted A* (h inflated by W)")
    opt = wa[0]["cost"]
    for r in wa[1:]:
        print(f"  {r['name']}: cost {r['cost']} (gap {r['cost']-opt:+d}, {100*(r['cost']-opt)/opt:+.1f}%), "
              f"expanded {r['expanded']} vs {wa[0]['expanded']}")
    save_csv(wa, os.path.join(RESULTS, "partB_inadmissible.csv"), "weighted A* (inadmissible)")
    return rows


if __name__ == "__main__":
    run_heuristic_study()