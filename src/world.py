"""Part A (Lab 1): load instance, build cost map (vectorised vs loop), benchmark, plot.
Run:  python -m src.world        (from the project root)
Convention: cost of a move = terrain cost of the cell you ENTER. Blocked cells = np.inf.
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from members import CMS_IDS, STAMP
from oel_instance import make_instance, TERRAINS

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def load():
    """Returns (instance dict, grid as int ndarray; -1 = blocked, 0..3 = TERRAINS index)."""
    inst, grid = make_instance(CMS_IDS)
    return inst, grid


def cost_table(inst):
    return np.array([inst["terrain_cost"][t] for t in TERRAINS], dtype=float)


def cost_map_vectorised(grid, inst):
    """No Python loops: fancy-indexing lookup + np.where for blocked cells."""
    table = cost_table(inst)
    return np.where(grid >= 0, table[np.clip(grid, 0, None)], np.inf)


def cost_map_loop(grid, inst):
    """Pure-Python baseline."""
    n = len(grid)
    names = TERRAINS
    out = [[0.0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            v = int(grid[r][c])
            out[r][c] = float("inf") if v < 0 else float(inst["terrain_cost"][names[v]])
    return np.array(out)


def _time(fn, reps):
    best = []
    for _ in range(5):                      # 5 trials, keep the best (least noise)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best.append((time.perf_counter() - t0) / reps)
    return min(best)


def benchmark(inst, grid):
    rows = []
    # 1) your real grid, 2) bigger grids so the speed-up is clearly visible
    cases = [("your instance", grid)]
    rng = np.random.default_rng(0)
    for n in (100, 500):
        g = rng.integers(0, 4, size=(n, n)); g[rng.random((n, n)) < 0.2] = -1
        cases.append((f"synthetic {n}x{n}", g))
    for name, g in cases:
        reps = 200 if g.shape[0] <= 50 else (10 if g.shape[0] <= 100 else 1)
        gl = g.tolist()
        t_loop = _time(lambda: cost_map_loop(gl, inst), reps)
        t_vec = _time(lambda: cost_map_vectorised(g, inst), reps)
        assert np.array_equal(cost_map_loop(gl, inst), cost_map_vectorised(g, inst))
        rows.append((name, g.shape[0], t_loop * 1e3, t_vec * 1e3, t_loop / t_vec))
    return rows


def plot_world(inst, grid, path):
    cmap = ListedColormap(["#222222", "#9e9e9e", "#7cb342", "#6d4c41", "#4fc3f7"])  # blocked, road, grass, mud, ford
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(grid + 1, cmap=cmap, vmin=0, vmax=4)
    sr, sc = inst["start"]
    ax.scatter([sc], [sr], c="red", s=140, marker="s", label="depot/start", zorder=3)
    for i, (r, c) in enumerate(inst["delivery_stops"], 1):
        ax.scatter([c], [r], c="yellow", edgecolors="k", s=120, marker="*", zorder=3)
        ax.text(c, r - 0.6, str(i), ha="center", color="k", fontsize=8, weight="bold")
    from matplotlib.patches import Patch
    leg = [Patch(color=col, label=f"{t} (cost {inst['terrain_cost'][t]})")
           for col, t in zip(["#9e9e9e", "#7cb342", "#6d4c41", "#4fc3f7"], TERRAINS)]
    leg.append(Patch(color="#222222", label="blocked"))
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_title(f"Instance {inst['instance_id']} ({inst['grid_size']}x{inst['grid_size']})\n{STAMP}", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_part_a():
    print(STAMP)
    inst, grid = load()
    print(f"Instance {inst['instance_id']}  grid={grid.shape}  twists={inst['twists']}")
    cm = cost_map_vectorised(grid, inst)
    print("Blocked cells:", int((grid < 0).sum()), "| free cells:", int((grid >= 0).sum()))
    print("Cost map sample (top-left 6x6):\n", cm[:6, :6])
    rows = benchmark(inst, grid)
    print(f"\n{'case':<20}{'n':>5}{'loop (ms)':>12}{'numpy (ms)':>12}{'speed-up':>10}")
    for name, n, tl, tv, sp in rows:
        print(f"{name:<20}{n:>5}{tl:>12.4f}{tv:>12.4f}{sp:>9.1f}x")
    print(STAMP)
    with open(os.path.join(RESULTS, "partA_benchmark.csv"), "w") as f:
        f.write(f"# {STAMP}\ncase,n,loop_ms,numpy_ms,speedup\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.5f},{r[3]:.5f},{r[4]:.2f}\n")
    plot_world(inst, grid, os.path.join(RESULTS, "partA_world.png"))
    print("Saved results/partA_benchmark.csv and results/partA_world.png")


if __name__ == "__main__":
    run_part_a()