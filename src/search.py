"""Part B: BFS, UCS, DLS, IDS, Greedy Best-First, A* -- all written from scratch.
State space = (r, c, mask, fuel); see problem.py.  Run: python -m src.search

Pruning used by the graph searches (BFS/UCS/Greedy/A*): FUEL DOMINANCE. Among states with the same
(r, c, mask), one that was already expanded with >= fuel dominates the new one -> skip.
DLS/IDS keep NO global closed list (that is the point of IDS: O(depth) memory); they only avoid
revisiting states ON THE CURRENT PATH (full-state check, held in a set that is updated on push/pop).
"""
import os, time, heapq, tracemalloc
from collections import deque
from members import STAMP
from src.twists import build_world
from src.problem import DeliveryProblem, path_cost
from src.heuristics import h_zero, h_nearest, h_farthest

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _rebuild(nodes, nid):
    acts = []
    while nid > 0:
        pid, a = nodes[nid]; acts.append(a); nid = pid
    return "".join(reversed(acts))


def _res(name, ok, acts, cost, exp, gen, peak, note=""):
    return dict(name=name, solved=ok, actions=acts, cost=cost, expanded=exp, generated=gen, peak_stored=peak, note=note)


# ------------------------------------------------------------------ BFS
def bfs(problem, max_stored=None):
    s0 = problem.initial_state()
    nodes = [(-1, "")]; q = deque([(s0, 0)]); seen = {problem.dom_key(s0): s0[3]}
    exp = gen = 0; peak = 1
    while q:
        s, nid = q.popleft(); exp += 1
        for a, ns, c in problem.successors(s):
            k = problem.dom_key(ns)
            if seen.get(k, -1) >= ns[3]:
                continue
            seen[k] = ns[3]; gen += 1
            nodes.append((nid, a)); nn = len(nodes) - 1
            if problem.is_goal(ns):
                acts = _rebuild(nodes, nn)
                return _res("BFS", True, acts, path_cost(problem, acts)[0], exp, gen, peak)
            q.append((ns, nn))
        peak = max(peak, len(q) + len(seen))
        if max_stored and peak > max_stored:
            return _res("BFS", False, "", None, exp, gen, peak, "memory limit")
    return _res("BFS", False, "", None, exp, gen, peak, "no solution")


# ------------------------------------------------------- best-first family
def best_first(problem, mode, h=None, weight=1.0, max_stored=None, max_expanded=None, name=None, trace=None):
    """mode: 'ucs' (f=g) | 'astar' (f=g+weight*h) | 'greedy' (f=h). weight>1 gives an inadmissible A* variant."""
    name = name or mode.upper()
    h = h or h_zero(problem)
    s0 = problem.initial_state()
    nodes = [(-1, "")]; cnt = 0
    def prio(g, s):
        return g if mode == "ucs" else (g + weight * h(s) if mode == "astar" else h(s))
    pq = [(prio(0, s0), 0, 0, s0, 0)]            # (f, tiebreak, g, state, node_id)
    best_fuel = {}; exp = gen = 0; peak = 1
    while pq:
        f, _, g, s, nid = heapq.heappop(pq)
        k = problem.dom_key(s)
        if best_fuel.get(k, -1) >= s[3]:
            continue
        best_fuel[k] = s[3]; exp += 1
        if trace is not None:
            trace.append((s, g))
        if problem.is_goal(s):
            acts = _rebuild(nodes, nid)
            return _res(name, True, acts, g, exp, gen, peak)
        for a, ns, c in problem.successors(s):
            if best_fuel.get(problem.dom_key(ns), -1) >= ns[3]:
                continue
            cnt += 1; gen += 1
            nodes.append((nid, a))
            heapq.heappush(pq, (prio(g + c, ns), cnt, g + c, ns, len(nodes) - 1))
        peak = max(peak, len(pq) + len(best_fuel))
        if (max_stored and peak > max_stored) or (max_expanded and exp > max_expanded):
            return _res(name, False, "", None, exp, gen, peak, "limit hit")
    return _res(name, False, "", None, exp, gen, peak, "no solution")


def ucs(problem, name="UCS", **kw):
    return best_first(problem, "ucs", name=name, **kw)

def greedy(problem, h, name="Greedy", **kw):
    return best_first(problem, "greedy", h=h, name=name, **kw)

def astar(problem, h, weight=1.0, name="A*", **kw):
    return best_first(problem, "astar", h=h, weight=weight, name=name, **kw)


# ------------------------------------------------------------------ DLS / IDS
def dls(problem, limit, budget=None, stats=None):
    """Depth-limited DFS, iterative (explicit stack). Returns ('solution', actions) | ('cutoff', None) |
    ('failure', None) | ('budget', None). Only the CURRENT PATH is stored (path_set) -> O(limit) memory."""
    stats = stats if stats is not None else {"expanded": 0, "generated": 0, "peak": 0}
    s0 = problem.initial_state()
    if problem.is_goal(s0):
        return "solution", ""
    path_set = {s0}; acts = []
    stack = [iter(problem.successors(s0))]; states = [s0]; cutoff = False
    stats["expanded"] += 1
    while stack:
        if budget and stats["expanded"] > budget:
            return "budget", None
        if len(states) - 1 >= limit:                # at depth limit: do not expand
            cutoff = True
            stack.pop(); path_set.discard(states.pop())
            if acts: acts.pop()
            continue
        try:
            a, ns, c = next(stack[-1])
        except StopIteration:
            stack.pop(); path_set.discard(states.pop())
            if acts: acts.pop()
            continue
        if ns in path_set:                          # cycle on current path -> skip
            continue
        stats["generated"] += 1
        acts.append(a)
        if problem.is_goal(ns):
            return "solution", "".join(acts)
        path_set.add(ns); states.append(ns); stack.append(iter(problem.successors(ns)))
        stats["expanded"] += 1
        stats["peak"] = max(stats["peak"], len(states))
    return ("cutoff" if cutoff else "failure"), None


def run_dls(problem, limit, budget=None):
    st = {"expanded": 0, "generated": 0, "peak": 0}
    status, acts = dls(problem, limit, budget, st)
    ok = status == "solution"
    return _res(f"DLS(l={limit})", ok, acts or "", path_cost(problem, acts)[0] if ok else None,
                st["expanded"], st["generated"], st["peak"], status)


def ids(problem, max_depth=200, budget=None):
    st = {"expanded": 0, "generated": 0, "peak": 0}
    for L in range(max_depth + 1):
        status, acts = dls(problem, L, budget, st)
        if status == "solution":
            return _res("IDS", True, acts, path_cost(problem, acts)[0], st["expanded"], st["generated"], st["peak"], f"depth {L}")
        if status in ("failure", "budget"):
            return _res("IDS", False, "", None, st["expanded"], st["generated"], st["peak"], status)
    return _res("IDS", False, "", None, st["expanded"], st["generated"], st["peak"], "max depth")


# ------------------------------------------------------------------ measurement
def measure(fn):
    """Runs fn twice: once for clean wall-clock time, once under tracemalloc for peak memory (KB)."""
    t0 = time.perf_counter(); r = fn(); r["time_ms"] = (time.perf_counter() - t0) * 1e3
    tracemalloc.start(); fn(); r["mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024; tracemalloc.stop()
    return r


def verify(problem, r):
    """Every reported plan is re-simulated from scratch: same cost, and it really reaches the goal."""
    if r["solved"]:
        assert path_cost(problem, r["actions"]) == (r["cost"], True), f"{r['name']} returned an invalid plan"


def print_table(rows, title):
    print(f"\n{title}\n{STAMP}")
    print(f"{'algorithm':<22}{'solved':>7}{'cost':>7}{'expanded':>10}{'generated':>10}{'peak':>9}{'ms':>10}{'KB':>10}  note")
    for r in rows:
        print(f"{r['name']:<22}{str(r['solved']):>7}{str(r['cost']):>7}{r['expanded']:>10}{r['generated']:>10}"
              f"{r['peak_stored']:>9}{r['time_ms']:>10.1f}{r['mem_kb']:>10.0f}  {r['note']}")


def save_csv(rows, path, title):
    import csv
    with open(path, "w", newline="") as f:
        f.write(f"# {STAMP}\n# {title}\n")
        wr = csv.writer(f); wr.writerow(["algorithm", "solved", "cost", "expanded", "generated", "peak_stored", "ms", "KB", "note"])
        for r in rows:
            wr.writerow([r["name"], r["solved"], r["cost"], r["expanded"], r["generated"], r["peak_stored"],
                         f"{r['time_ms']:.2f}", f"{r['mem_kb']:.0f}", r["note"]])


def make_mini_problem(world, target_steps=12):
    """Reduced problem for DLS/IDS: ONE stop, start placed ~target_steps moves away (BFS on the directed graph)."""
    stop = world.stops[0]
    best = None
    for (r, c) in sorted(world.nbrs):
        if (r, c) == stop or abs(r - stop[0]) + abs(c - stop[1]) > target_steps:
            continue
        dist = {(r, c): 0}; q = deque([(r, c)])
        while q:
            u = q.popleft()
            for nr, nc, _, _ in world.nbrs[u]:
                if (nr, nc) not in dist:
                    dist[(nr, nc)] = dist[u] + 1; q.append((nr, nc))
        d = dist.get(stop)
        if d is not None and (best is None or abs(d - target_steps) < abs(best[1] - target_steps)):
            best = ((r, c), d)
            if d == target_steps:
                break
    return DeliveryProblem(world, stop_ids=[0], start=best[0]), best[1]


def run_search_demo():
    print(STAMP)
    w = build_world(); P = DeliveryProblem(w)
    rows = [measure(lambda: bfs(P)),
            measure(lambda: ucs(P)),
            measure(lambda: greedy(P, h_farthest(P), name="Greedy")),
            measure(lambda: astar(P, h_nearest(P), name="A*(h=nearest)")),
            measure(lambda: astar(P, h_farthest(P), name="A*(h=farthest)"))]
    for r in rows:
        verify(P, r)
    print_table(rows, "FULL PROBLEM (6 stops, fuel, one-way)")
    save_csv(rows, os.path.join(RESULTS, "partB_full_problem.csv"), "full problem")
    return rows


def run_ids_demo():
    w = build_world(); mini, steps = make_mini_problem(w, 12)
    print(f"\nMini problem: 1 stop, start {mini.start}, shortest path = {steps} moves")
    rows = [measure(lambda: astar(mini, h_farthest(mini), name="A*(h=farthest)")),
            measure(lambda: astar(mini, h_farthest(mini), name="A*, memory cap 50", max_stored=50)),
            measure(lambda: ucs(mini)),
            measure(lambda: bfs(mini)),
            measure(lambda: run_dls(mini, steps - 1)),
            measure(lambda: run_dls(mini, steps)),
            measure(lambda: ids(mini))]
    for r in rows:
        verify(mini, r)
    print_table(rows, "MINI PROBLEM: DLS / IDS vs the rest")
    save_csv(rows, os.path.join(RESULTS, "partB_mini_ids.csv"), "mini problem, memory cap demo")
    return rows


if __name__ == "__main__":
    run_search_demo()
    run_ids_demo()