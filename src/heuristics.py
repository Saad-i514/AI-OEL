"""Heuristics for the delivery problem. Signature: factory(problem) -> h(state) -> float.
Baselines here; OUR OWN heuristic is added in Step 6.
All baselines are admissible AND consistent: every move costs >= c_min and changes Manhattan distance by <= 1.
"""
from members import STAMP


def _cmin(problem):
    w = problem.w
    return min(w.inst["terrain_cost"].values())


def h_zero(problem):
    return lambda s: 0


def _undelivered(problem, s):
    m = s[2]
    return [st for st in problem.stops if not (m & problem.stop_bit[st])]


def h_nearest(problem):
    """Baseline 1: c_min * Manhattan distance to the NEAREST undelivered stop."""
    cm = _cmin(problem)
    def h(s):
        und = _undelivered(problem, s)
        return cm * min(abs(s[0]-r) + abs(s[1]-c) for r, c in und) if und else 0
    return h


def h_farthest(problem):
    """Baseline 2: c_min * Manhattan distance to the FARTHEST undelivered stop."""
    cm = _cmin(problem)
    def h(s):
        und = _undelivered(problem, s)
        return cm * max(abs(s[0]-r) + abs(s[1]-c) for r, c in und) if und else 0
    return h


# =====================================================================================
#  OUR OWN HEURISTIC  h_own  (Part B)
# =====================================================================================
#  Built on TRUE directed shortest-path distances (so it understands one-way streets), computed once
#  per problem by reverse Dijkstra, ignoring fuel (a relaxation).  Let U = undelivered stops, f = fuel.
#
#  (1) FUEL-AWARE FARTHEST-STOP TERM       T_i(s) = d(pos->i)                      if d(pos->i) <= f
#                                                 = min_a d(pos->a) + d(a->i)       otherwise  (a = charging station)
#      If the direct trip costs more than the fuel on board, EVERY feasible plan must pass a station first.
#      h1 = max_{i in U} T_i(s)
#  (2) TOUR TERM    h2 = min_{i in U} d(pos->i) + MST(U),  MST over stops with symmetric weights
#      w(i,j) = min(d(i->j), d(j->i)).  Any route that visits all of U is a spanning tree of U -> cost >= MST(U).
#  h_own = max(h1, h2).  Admissible (max of admissible bounds); consistent (see report: each term obeys
#  h(s) <= c(s,s') + h(s'); also checked empirically in study_b.py).
import heapq, math, itertools


def _reverse_graph(world):
    rev = {u: [] for u in world.nbrs}
    for u, lst in world.nbrs.items():
        for nr, nc, cost, _ in lst:
            rev[(nr, nc)].append((u, cost))           # edge u -> v costs cost[v]
    return rev


def _rev_dijkstra(rev, sources):
    """sources: dict cell -> initial distance. Returns dist[cell] = min_src (d(cell->src) + init[src])."""
    dist = dict(sources); pq = [(d, u) for u, d in sources.items()]; heapq.heapify(pq)
    while pq:
        d, v = heapq.heappop(pq)
        if d > dist.get(v, math.inf):
            continue
        for u, w in rev[v]:
            nd = d + w
            if nd < dist.get(u, math.inf):
                dist[u] = nd; heapq.heappush(pq, (nd, u))
    return dist


def h_own(problem, use_mst=True, use_fuel=True):
    w = problem.w; stops = problem.stops; k = len(stops); INF = math.inf
    rev = _reverse_graph(w)
    d_to = [_rev_dijkstra(rev, {st: 0}) for st in stops]                       # d_to[i][cell] = d(cell -> stop i)
    via = []                                                                   # via[i][cell] = min_a d(cell->a)+d(a->i)
    for i in range(k):
        init = {a: d_to[i][a] for a in w.stations if a in d_to[i]}
        via.append(_rev_dijkstra(rev, init))
    # symmetric stop-to-stop weights and MST for every subset of stops
    W = [[0 if i == j else min(d_to[j].get(stops[i], INF), d_to[i].get(stops[j], INF)) for j in range(k)] for i in range(k)]
    mst = {}
    for mask in range(1 << k):
        nodes = [i for i in range(k) if mask >> i & 1]
        if len(nodes) <= 1:
            mst[mask] = 0; continue
        seen = {nodes[0]}; total = 0
        while len(seen) < len(nodes):
            c, j = min((W[a][b], b) for a in seen for b in nodes if b not in seen)
            total += c; seen.add(j)
        mst[mask] = total
    full = (1 << k) - 1; bit = [1 << i for i in range(k)]

    def h(s):
        r, c, mask, fuel = s
        if mask == full:
            return 0
        pos = (r, c); h1 = 0; near = INF
        for i in range(k):
            if mask & bit[i]:
                continue
            d = d_to[i].get(pos, INF)
            near = min(near, d)
            t = d
            if use_fuel and d > fuel:
                t = via[i].get(pos, INF)
            if t > h1:
                h1 = t
        und = full & ~mask
        h2 = near + mst[und] if use_mst else 0
        return h1 if h1 >= h2 else h2
    return h