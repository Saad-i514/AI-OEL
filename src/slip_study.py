"""Part C, slip analysis WITHOUT the rival: what does the deterministic-move assumption cost?
The planner (A* + h_own) assumes every move succeeds. We run the real 6-parcel job from the depot with slips,
replan after every slip, and compare the cost it PREDICTED with what it actually paid.   Run: python -m src.slip_study
"""
import os, math
import numpy as np
from members import STAMP
from src.twists import build_world
from src.problem import DeliveryProblem
from src.heuristics import h_own
from src.game import GameWorld, plan_from
from src.tune_c import seed_base

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def mean_ci(x):
    x = np.array(x, dtype=float)
    return (float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 1 else (float(x.mean()), 0.0)


def slip_game(gw, P, h, slip_p, seed, t_max=3000):
    rng = np.random.default_rng(seed)
    a = gw.cid[gw.w.start]; mask = 0; fuel = gw.F
    plan, nominal = plan_from(P, h, (*gw.rc[a], mask, fuel))
    cost = slips = replans = turns = 0; done = False
    while plan and turns < t_max:
        turns += 1
        legal = [(nid, c, ch) for nid, c, ch in gw.adj[a] if c <= fuel]
        intended = next(x for x in legal if x[2] == plan[0]); actual = intended
        if rng.random() < slip_p:
            others = [x for x in legal if x[2] != plan[0] and
                      gw.recoverable(x[0], gw.F if gw.station[x[0]] else fuel - x[1], mask | gw.stopbit[x[0]])]
            if others:
                actual = others[int(rng.integers(len(others)))]
            slips += 1
        nid, c, _ = actual
        fuel = gw.F if gw.station[nid] else fuel - c
        a = nid; mask |= gw.stopbit[nid]; cost += c
        if mask == gw.full:
            done = True; break
        if nid != intended[0]:                                   # we are not where the plan thinks we are -> replan
            replans += 1; plan, _ = plan_from(P, h, (*gw.rc[a], mask, fuel))
        else:
            plan = plan[1:]
    return dict(done=done, nominal=nominal, cost=cost, slips=slips, replans=replans, turns=turns)


def run_slip_study(n_games=30):
    print(STAMP)
    w = build_world(); gw = GameWorld(w); P = DeliveryProblem(w); h = h_own(P); sb = seed_base()
    print(f"\nSLIP COST WITHOUT A RIVAL: full 6-parcel job, {n_games} games per slip level, replan after every slip\n{STAMP}")
    print(f"{'slip p':>7}{'finished':>10}{'planned':>9}{'actual cost':>16}{'extra cost':>16}{'extra %':>9}{'slips':>8}{'replans':>9}{'extra/replan':>14}")
    rows = []
    for p, n_p, t_max in ((0.0, n_games, 3000), (0.10, n_games, 3000), (0.26, n_games, 3000),
                          (0.50, min(10, n_games), 1500)):
        R = [slip_game(gw, P, h, p, sb + 3000 + i, t_max) for i in range(n_p)]
        ok = [r for r in R if r["done"]]
        if not ok:
            print(f"{p:>7.2f}{'0/%d' % n_p:>10}{347:>9}   no game finished within {t_max} turns (replanning does not converge)")
            rows.append((p, 0, n_p, 347, (0, 0), (0, 0), 0, 0, 0)); continue
        nom = ok[0]["nominal"]
        act = mean_ci([r["cost"] for r in ok]); ex = mean_ci([r["cost"] - r["nominal"] for r in ok])
        sl = np.mean([r["slips"] for r in ok]); rp = np.mean([r["replans"] for r in ok])
        per = ex[0] / rp if rp else 0.0
        print(f"{p:>7.2f}{f'{len(ok)}/{n_p}':>10}{nom:>9}{act[0]:>10.0f}+-{act[1]:<4.0f}{ex[0]:>10.0f}+-{ex[1]:<4.0f}{100*ex[0]/nom:>8.0f}%{sl:>8.1f}{rp:>9.1f}{per:>14.1f}")
        rows.append((p, len(ok), n_p, nom, act, ex, sl, rp, per))
    with open(os.path.join(RESULTS, "partC_slip_cost.csv"), "w") as f:
        f.write(f"# {STAMP}\nslip_p,finished,games,planned,actual_mean,actual_ci,extra_mean,extra_ci,slips,replans,extra_per_replan\n")
        for p, k, n, nom, act, ex, sl, rp, per in rows:
            f.write(f"{p},{k},{n},{nom},{act[0]:.1f},{act[1]:.1f},{ex[0]:.1f},{ex[1]:.1f},{sl:.1f},{rp:.1f},{per:.2f}\n")
    print(f"\n(slip 0.50 uses {min(10, n_games)} games and a 1500-turn cap; cost columns average only the games that finished)")
    print("\nReading: 'planned' is what the deterministic planner promised; 'extra' is what slips really cost on top.")
    print(f"Saved results/partC_slip_cost.csv\n{STAMP}")


if __name__ == "__main__":
    run_slip_study()
