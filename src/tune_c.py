"""Part C experiments: (1) grid search over evaluation weights, (2) 30 games vs the rival with 95% CIs,
(3) slip / replanning analysis, (4) slip-probability sweep.   Run: python -m src.tune_c [--fast]
Games run in parallel (one process per CPU core). Every game is seeded from the CMS IDs, so results are reproducible.
"""
import os, sys, math, time, hashlib, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from members import STAMP, CMS_IDS

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
_C = {}


def seed_base():
    key = "|".join(sorted(x.strip().upper() for x in CMS_IDS)) + "|games"
    return int(hashlib.sha256(key.encode()).hexdigest()[:6], 16)


def _worker(args):
    weights, slip_p, seed = args
    if not _C:
        from src.twists import build_world
        from src.problem import DeliveryProblem
        from src.heuristics import h_own
        from src.game import GameWorld
        w = build_world(); P = DeliveryProblem(w)
        _C.update(w=w, P=P, h=h_own(P), gw=GameWorld(w))
    from src.game import play_game
    return play_game(_C["gw"], _C["P"], _C["h"], weights, slip_p, _C["w"].inst["adversary_search_depth"], seed)


def run_games(pool, weights, slip_p, seeds):
    return list(pool.map(_worker, [(weights, slip_p, s) for s in seeds]))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def mean_ci(x):
    x = np.array(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    return x.mean(), (1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0)


def wdl(res):
    n = len(res)
    return [sum(r["result"] == k for r in res) for k in ("win", "draw", "loss")], n


def fmt_wdl(res):
    (w, d, l), n = wdl(res)
    out = []
    for k in (w, d, l):
        p, lo, hi = wilson(k, n); out.append(f"{100*p:4.0f}% [{100*lo:3.0f},{100*hi:3.0f}]")
    return "  ".join(out)


def main(fast=False):
    print(STAMP)
    sb = seed_base()
    n_tune, n_final, n_sweep = (1, 2, 1) if fast else (8, 30, 12)
    from src.game import DEFAULT_W
    slip = 0.26
    t0 = time.perf_counter()
    # On Windows each worker starts a fresh interpreter and imports NumPy,
    # matplotlib, and this module. Starting one worker per logical CPU made
    # startup disproportionately expensive for these small game batches.
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        # ---------- 1) weight grid search (separate seeds from the evaluation games) ----------
        if fast:
            # A small representative set keeps the full pipeline quick while
            # still exercising tuning, evaluation, and result generation.
            grid = [(1.0, 40.0, 2.0, 0.0, 10.0), (1.0, 40.0, 8.0, 5.0, 10.0),
                    (2.0, 40.0, 2.0, 5.0, 10.0), (2.0, 40.0, 8.0, 0.0, 10.0)]
        else:
            grid = [(wp, 40.0, wd, wr, 10.0) for wp, wd, wr in itertools.product((1.0, 2.0), (2.0, 4.0, 8.0), (0.0, 5.0))]
        tune_seeds = [sb + 100 + i for i in range(n_tune)]
        print(f"\nWEIGHT GRID SEARCH: {len(grid)} configs x {n_tune} games (slip {slip})\n{STAMP}")
        print(f"{'(progress, deliv, rival-dist, race, fuel)':<44}{'W':>3}{'D':>3}{'L':>3}{'score':>7}")
        scored = []
        for W in grid:
            res = run_games(pool, W, slip, tune_seeds)
            (w, d, l), n = wdl(res); score = w + 0.5 * d
            scored.append((score, -l, W, (w, d, l)))
            print(f"{str(W):<44}{w:>3}{d:>3}{l:>3}{score:>7.1f}")
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = scored[0][2]
        print(f"best weights: {best}   (default was {DEFAULT_W})")
        with open(os.path.join(RESULTS, "partC_tuning.csv"), "w") as f:
            f.write(f"# {STAMP}\nprogress,delivered,rival_dist,race,fuel,wins,draws,losses,score\n")
            for sc, _, W, (w, d, l) in scored:
                f.write(",".join(map(str, W)) + f",{w},{d},{l},{sc}\n")
        # ---------- 2) final: 30 games ----------
        fseeds = [sb + 1000 + i for i in range(n_final)]
        tuned = run_games(pool, best, slip, fseeds)
        default = run_games(pool, DEFAULT_W, slip, fseeds)
        nos = run_games(pool, best, 0.0, fseeds)                  # same seeds, no slips (paired)
        print(f"\nFINAL: {n_final} games vs interceptor, slip {slip}.  win / draw / loss with 95% Wilson CI\n{STAMP}")
        print(f"{'agent':<34}{'win':>16}{'draw':>16}{'loss':>16}")
        for name, res in (("tuned weights", tuned), ("default weights", default), ("tuned weights, slip = 0", nos)):
            print(f"{name:<34}{fmt_wdl(res)}")
        # ---------- 3) slip / replanning analysis ----------
        T = sum(r["turns"] for r in tuned)
        print(f"\nSLIP & REPLANNING ANALYSIS (tuned weights, {n_final} games)\n{STAMP}")
        for key, label in (("slips", "slips per game"), ("replan_slip", "replans triggered by slips per game"),
                           ("replan_evade", "replans after tactical evasions per game"),
                           ("replan_rival", "rival-aware strategic replans per game"), ("turns", "turns per game")):
            m, ci = mean_ci([r[key] for r in tuned]); print(f"  {label:<44}{m:8.1f} +/- {ci:.1f}")
        print(f"  {'share of turns that ended in a slip replan':<44}{100*sum(r['replan_slip'] for r in tuned)/T:8.1f} %")
        pairs = [(a["cost"] - b["cost"]) for a, b in zip(tuned, nos) if a["result"] == "win" and b["result"] == "win"]
        m, ci = mean_ci(pairs)
        print(f"  extra cost caused by slips (paired wins, n={len(pairs)}): {m:.0f} +/- {ci:.0f}  (cost units)")
        caught = [r for r in tuned if r.get("why") == "caught"]
        cas = sum(r["caught_after_slip"] for r in caught)
        print(f"  captures within 3 turns after a slip: {cas} of {len(caught)}  <- the deterministic-move assumption failing")
        strand = sum(r.get("why") == "stranded" for r in tuned)
        print(f"  stranded (no fuel/path): {strand}")
        # ---------- 4) slip sweep ----------
        sw_seeds = [sb + 2000 + i for i in range(n_sweep)]
        print(f"\nSLIP SWEEP ({n_sweep} games each, tuned weights)\n{STAMP}")
        print(f"{'slip p':>7}{'win':>16}{'draw':>16}{'loss':>16}{'replans/game':>14}")
        sweep = []
        for p in (0.0, 0.10, 0.26, 0.50):
            res = run_games(pool, best, p, sw_seeds); sweep.append((p, res))
            print(f"{p:>7.2f}{fmt_wdl(res)}{np.mean([r['replan_slip'] for r in res]):>14.1f}")
    # ---------- files ----------
    with open(os.path.join(RESULTS, "partC_final.csv"), "w") as f:
        f.write(f"# {STAMP}\n# weights {best}\nagent,win,draw,loss,n\n")
        for name, res in (("tuned", tuned), ("default", default), ("tuned_slip0", nos)):
            (w, d, l), n = wdl(res); f.write(f"{name},{w},{d},{l},{n}\n")
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["tuned", "default", "tuned, slip=0"]; x = np.arange(3); wd = 0.26
    for j, (k, col) in enumerate((("win", "#4caf50"), ("draw", "#9e9e9e"), ("loss", "#e53935"))):
        vals, errs = [], [[], []]
        for res in (tuned, default, nos):
            kk = sum(r["result"] == k for r in res); p, lo, hi = wilson(kk, len(res))
            vals.append(100 * p); errs[0].append(100 * (p - lo)); errs[1].append(100 * (hi - p))
        ax.bar(x + (j - 1) * wd, vals, wd, yerr=errs, capsize=3, color=col, label=k)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("% of games (95% Wilson CI)"); ax.legend()
    ax.set_title(f"{n_final} games vs interceptor (slip {slip})\n{STAMP}", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "partC_results.png"), dpi=150); plt.close(fig)
    print(f"\nSaved results/partC_tuning.csv, partC_final.csv, partC_results.png   ({time.perf_counter()-t0:.0f}s)\n{STAMP}")


if __name__ == "__main__":
    main(fast="--fast" in sys.argv)
