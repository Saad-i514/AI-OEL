"""Twist layer: builds the FULL world = instructor instance + our two twists.
  Twist 1  fuel limit + charging stations  (adds `fuel` to the state, stations refuel to full)
  Twist 2  one-way streets                 (directed graph: out_ok[r, c, d])
  Twist 3  narrow-alley surcharge          (OUR OWN: a free cell with >= 2 blocked orthogonal neighbours is a narrow
                                            alley; entering it costs ALLEY_SURCHARGE extra, which also burns fuel)
Both are generated deterministically from the CMS IDs so instructors can regenerate them.
Run:  python -m src.twists
"""
import os, hashlib, heapq, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from members import CMS_IDS, STAMP
from oel_instance import make_instance, TERRAINS

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))          # N, S, W, E
DIR_NAMES = "NSWE"
ALLEY_SURCHARGE = 2
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


class World:
    """Everything the planners / simulator need."""
    def __init__(self, inst, grid, cost, out_ok, stations, fuel_cap):
        self.inst, self.grid, self.cost, self.out_ok = inst, grid, cost, out_ok
        self.n = grid.shape[0]
        self.start = tuple(inst["start"])
        self.stops = [tuple(s) for s in inst["delivery_stops"]]
        self.stations = [tuple(s) for s in stations]
        self.station_set = set(self.stations)
        self.fuel_cap = int(fuel_cap)
        self.slip_p = inst["slip_probability"]
        # precomputed neighbour lists: nbrs[(r,c)] -> [(nr, nc, step_cost, dir_idx), ...]
        self.nbrs = {}
        for r in range(self.n):
            for c in range(self.n):
                if grid[r, c] < 0:
                    continue
                lst = []
                for d, (dr, dc) in enumerate(DIRS):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.n and 0 <= nc < self.n and grid[nr, nc] >= 0 and out_ok[r, c, d]:
                        lst.append((nr, nc, int(cost[nr, nc]), d))
                self.nbrs[(r, c)] = lst

    def n_one_way(self):
        """Number of undirected adjacent pairs that are one-way."""
        cnt = 0
        for (r, c), lst in self.nbrs.items():
            for nr, nc, _, _ in lst:
                if (r, c) not in {(a, b) for a, b, _, _ in self.nbrs[(nr, nc)]}:
                    cnt += 1
        return cnt


# ---------- helpers -------------------------------------------------------------------
def _rng(salt):
    key = "|".join(sorted(x.strip().upper() for x in CMS_IDS)) + "|" + salt
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def _reach(nbrs, src):
    seen = {src}; q = deque([src])
    while q:
        u = q.popleft()
        for nr, nc, _, _ in nbrs.get(u, ()):
            if (nr, nc) not in seen:
                seen.add((nr, nc)); q.append((nr, nc))
    return seen


def _dijkstra(nbrs, src):
    dist = {src: 0}; pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for nr, nc, w, _ in nbrs.get(u, ()):
            v = (nr, nc)
            if d + w < dist.get(v, math.inf):
                dist[v] = d + w; heapq.heappush(pq, (d + w, v))
    return dist


def _make_out_ok(grid, cost_free, rng, frac):
    n = grid.shape[0]
    out_ok = np.zeros((n, n, 4), dtype=bool)
    for r in range(n):
        for c in range(n):
            if grid[r, c] < 0:
                continue
            for d, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] >= 0:
                    out_ok[r, c, d] = True
    # make a fraction of the undirected edges one-way (random orientation); S and E only to visit each edge once
    for r in range(n):
        for c in range(n):
            if grid[r, c] < 0:
                continue
            for d in (1, 3):
                dr, dc = DIRS[d]; nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] >= 0 and rng.random() < frac:
                    if rng.random() < 0.5:
                        out_ok[nr, nc, d ^ 1] = False        # forbid going back
                    else:
                        out_ok[r, c, d] = False              # forbid going forward
    return out_ok


def _nbrs_from(grid, cost, out_ok):
    n = grid.shape[0]; nb = {}
    for r in range(n):
        for c in range(n):
            if grid[r, c] < 0:
                continue
            lst = []
            for d, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] >= 0 and out_ok[r, c, d]:
                    lst.append((nr, nc, int(cost[nr, nc]), d))
            nb[(r, c)] = lst
    return nb


# ---------- world builder -------------------------------------------------------------
def build_world(one_way_frac=0.15, n_stations=4, fuel_slack=1.2):
    inst, grid = make_instance(CMS_IDS)
    table = np.array([inst["terrain_cost"][t] for t in TERRAINS], dtype=float)
    blocked = grid < 0
    pad = np.pad(blocked, 1, constant_values=False)                       # vectorised neighbour count (border = open)
    n_blocked = (pad[:-2, 1:-1].astype(int) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])
    alley = (grid >= 0) & (n_blocked >= 2)                                # twist 3: narrow alleys
    cost = np.where(grid >= 0, table[np.clip(grid, 0, None)] + ALLEY_SURCHARGE * alley, np.inf)
    start = tuple(inst["start"]); stops = [tuple(s) for s in inst["delivery_stops"]]
    key = [start] + stops
    rng = _rng("twists")

    # --- one-way streets: retry until start+stops are mutually reachable (solvable) ---
    frac = one_way_frac
    for attempt in range(200):
        out_ok = _make_out_ok(grid, cost, rng, frac)
        nb = _nbrs_from(grid, cost, out_ok)
        if all(all(k in _reach(nb, a) for k in key) for a in key):
            break
        if attempt % 10 == 9:
            frac *= 0.8
    else:
        raise RuntimeError("could not generate a strongly-connected one-way layout")

    # --- charging stations: depot + farthest-point sampling inside start's strongly connected zone ---
    fwd = _reach(nb, start)
    zone = [c for c in fwd if c not in set(key) and start in _reach(nb, c)]
    stations = [start]
    for _ in range(n_stations):
        cand = sorted(zone, key=lambda p: -min(abs(p[0]-s[0]) + abs(p[1]-s[1]) for s in stations))[:12]
        stations.append(tuple(cand[int(rng.integers(len(cand)))]))
        zone = [z for z in zone if z != stations[-1]]

    # --- fuel capacity: every stop must be servable station -> stop -> station, stations must chain ---
    d_from = {k: _dijkstra(nb, k) for k in set(stations) | set(stops)}
    d_to = {s: {} for s in stops}
    for s in stops:
        d_to[s] = {a: d_from[a].get(s, math.inf) for a in stations}
    need = max(min(d_to[s][a] + d_from[s].get(b, math.inf) for a in stations for b in stations) for s in stops)
    F = math.ceil(fuel_slack * need)
    while True:                                           # stations must be chainable with legs <= F
        ok = {stations[0]}; changed = True
        while changed:
            changed = False
            for a in list(ok):
                for b in stations:
                    if b not in ok and d_from[a].get(b, math.inf) <= F:
                        ok.add(b); changed = True
        if len(ok) == len(stations):
            break
        F += 2
    world = World(inst, grid, cost, out_ok, stations, F)
    world.alley = alley
    return world


# ---------- figure --------------------------------------------------------------------
def plot_world_twists(w, path):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap(["#222222", "#d0d0d0", "#aed581", "#a1887f", "#81d4fa"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(w.grid + 1, cmap=cmap, vmin=0, vmax=4)
    for (r, c), lst in w.nbrs.items():
        for nr, nc, _, d in lst:
            if (r, c) not in {(a, b) for a, b, _, _ in w.nbrs[(nr, nc)]}:      # one-way edge
                ax.annotate("", xy=(c + 0.3*(nc-c), r + 0.3*(nr-r)), xytext=(c, r),
                            arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))
    ar, ac = np.where(w.alley)
    ax.scatter(ac, ar, marker=".", c="darkorange", s=14, zorder=2)          # narrow alleys (+surcharge)
    for s in w.stations:
        ax.scatter([s[1]], [s[0]], marker="^", c="limegreen", edgecolors="k", s=110, zorder=3)
    ax.scatter([w.start[1]], [w.start[0]], marker="s", c="red", s=130, zorder=4)
    for i, (r, c) in enumerate(w.stops, 1):
        ax.scatter([c], [r], marker="*", c="yellow", edgecolors="k", s=160, zorder=4)
        ax.text(c, r - 0.6, str(i), ha="center", fontsize=8, weight="bold")
    leg = [Patch(color=col, label=f"{t} ({w.inst['terrain_cost'][t]})")
           for col, t in zip(["#d0d0d0", "#aed581", "#a1887f", "#81d4fa"], TERRAINS)]
    ax.legend(handles=leg + [Patch(color="#222222", label="blocked"), Patch(color="darkorange", label=f"narrow alley (+{ALLEY_SURCHARGE})")], loc="upper left",
              bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_title(f"Instance {w.inst['instance_id']}: green ^ = charging stations, red arrows = one-way, "
                 f"fuel cap {w.fuel_cap}\n{STAMP}", fontsize=7.5)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_twists():
    print(STAMP)
    w = build_world()
    print(f"Charging stations (incl. depot): {w.stations}")
    print(f"Fuel capacity: {w.fuel_cap}   one-way adjacencies: {w.n_one_way()}   narrow-alley cells: {int(w.alley.sum())}")
    plot_world_twists(w, os.path.join(RESULTS, "partB_twists_world.png"))
    print("Saved results/partB_twists_world.png")
    print(STAMP)
    return w


if __name__ == "__main__":
    run_twists()