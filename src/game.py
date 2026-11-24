"""Part C: the rival (INTERCEPTOR), reduced game, Minimax, Alpha-Beta, slip simulator.
Run: python -m src.game

REDUCED GAME. Full-grid game search is too big, so every search is (a) depth-limited, (b) confined to a LOCAL
WINDOW (Manhattan radius W around the agent) and (c) triggered only when the rival is within the danger radius R.
Outside the window the rest of the grid is compressed into precomputed distance tables used by the evaluation.
Game: turn-based, agent then rival (rival acts every 2nd turn, may not enter a stop cell), each moves ONE cell along the directed graph (blocked cells / one-way streets
obey the same rules).  Agent also burns fuel (twist 1).  Caught = same cell.  Win = all parcels delivered.
"""
import os, math, time
from collections import deque
import numpy as np
from members import STAMP, CMS_IDS
from src.twists import build_world, DIR_NAMES, DIRS
from src.problem import DeliveryProblem
from src.search import best_first
from src.heuristics import h_own, _reverse_graph, _rev_dijkstra

BIG = 100000
R_RIVAL = 8           # the rival searches (Alpha-Beta) when Manhattan(agent, rival) <= R_RIVAL, otherwise it approaches
R_DANGER = 4          # tactical Minimax is triggered when Manhattan(agent, rival) <= R_DANGER
WINDOW = 10           # local-window radius (Manhattan) of the reduced game
T_MAX = 800           # turn limit -> draw
DANGER_PEN = (40, 20, 8)   # planning penalty at <=2, <=4, <=6 hops from the rival
RIVAL_EVERY = 2       # rival acts every 2nd turn (speed 1/2): terrain slows vehicles, the rival uses fast roads

# default weights: (progress, delivered, rival distance, interception race, fuel)
WAIT_PEN = 15.0       # search penalty per WAIT move, so the agent cannot stall forever
DEFAULT_W = (1.0, 40.0, 4.0, 5.0, 10.0)


class GameWorld:
    def __init__(self, world):
        self.w = world
        cells = sorted(world.nbrs)
        self.cells = cells; self.N = len(cells); self.cid = {c: i for i, c in enumerate(cells)}
        self.adj = [[] for _ in range(self.N)]
        for u, lst in world.nbrs.items():
            for nr, nc, cost, d in lst:
                self.adj[self.cid[u]].append((self.cid[(nr, nc)], cost, DIR_NAMES[d]))
        self.rc = cells
        self.F = world.fuel_cap
        self.k = len(world.stops); self.full = (1 << self.k) - 1
        self.stop_ids = [self.cid[s] for s in world.stops]
        self.stopbit = [0] * self.N
        for i, sid in enumerate(self.stop_ids):
            self.stopbit[sid] = 1 << i
        self.station = [False] * self.N
        for s in world.stations:
            self.station[self.cid[s]] = True
        # hop distance matrix (rival moves one cell per turn) -- BFS from every cell on the directed graph
        INF = 10 ** 6
        self.hop = []
        for s in range(self.N):
            d = [INF] * self.N; d[s] = 0; q = deque([s])
            while q:
                u = q.popleft()
                for v, _, _ in self.adj[u]:
                    if d[v] == INF:
                        d[v] = d[u] + 1; q.append(v)
            self.hop.append(d)
        # SAFE ZONE: cells inside the start's strongly connected component. One-way streets can create dead-end
        # pockets; we assume the agent knows them (never enters them) and a slip never throws it into one.
        s0 = self.cid[world.start]
        self.safe = [self.hop[s0][i] < INF and self.hop[i][s0] < INF for i in range(self.N)]
        # ESCAPE COST: cheapest way from each cell to ANY charging station (reverse multi-source Dijkstra)
        rev = _reverse_graph(world)
        e = _rev_dijkstra(rev, {st: 0 for st in world.stations})
        self.esc = [e.get(c, INF) for c in cells]
        # cost distance to each stop (true directed terrain cost, ignores fuel)
        self.dcost = []
        for st in world.stops:
            dd = _rev_dijkstra(rev, {st: 0})
            self.dcost.append([dd.get(c, INF) for c in cells])

    def recoverable(self, nid, fuel_after, mask_after):
        """A move is acceptable if the agent stays in the safe zone AND can still reach a charger
        (or this move finishes the job). Used for slips and for evasive minimax moves."""
        return self.safe[nid] and (self.station[nid] or fuel_after >= self.esc[nid] or mask_after == self.full)

    def target(self, a, mask):
        """next undelivered stop (cell id) the agent is most likely heading to = cheapest to reach."""
        best, bt = 10 ** 9, -1
        for i in range(self.k):
            if not mask >> i & 1:
                c = self.dcost[i][a]
                if c < best:
                    best, bt = c, self.stop_ids[i]
        return bt


# ---------------------------------------------------------------- evaluation functions (agent's point of view)
def make_agent_eval(gw, h, weights):
    wp, wd, wr, wi, wf = weights
    rc = gw.rc; hop = gw.hop; F = gw.F
    def ev(st):
        a, r, mask, fuel = st
        ar, ac = rc[a]
        prog = -h((ar, ac, mask, fuel))                         # f1: estimated remaining delivery cost (own h)
        deliv = bin(mask).count("1")                            # f2: parcels delivered
        dist = min(hop[r][a], 10)                               # f3: rival's distance to us (capped)
        t = gw.target(a, mask)
        race = max(-8, min(8, hop[r][t] - hop[a][t])) if t >= 0 else 8   # f4: do we reach our next stop first?
        return wp * prog + wd * deliv + wr * dist + wi * race + wf * fuel / F   # f5: fuel margin
    return ev


def make_rival_eval(gw):
    """INTERCEPTOR: cares about the agent's next stop (the interception point), not just the agent itself."""
    hop = gw.hop
    def ev(st):
        a, r, mask, fuel = st
        t = gw.target(a, mask)
        race = max(-8, min(8, hop[r][t] - hop[a][t])) if t >= 0 else 8
        return 25 * bin(mask).count("1") + 2 * min(hop[r][a], 10) + 9 * race
    return ev


# ---------------------------------------------------------------- Minimax / Alpha-Beta
class Searcher:
    """mode: 'minimax' | 'alphabeta';  ordering: 'none' | 'own' | 'reverse' (worst case, for the study)."""
    def __init__(self, gw, ev, mode="alphabeta", ordering="own", period=1):
        """period: the rival acts only on game turns divisible by `period` (the agent knows the rival's speed)."""
        self.gw, self.ev, self.mode, self.ordering, self.period = gw, ev, mode, ordering, period
        self.nodes = 0; self.evals = 0

    def _kids(self, st, player):
        gw = self.gw; a, r, mask, fuel = st; out = []
        if player == "A":
            out.append((".", (a, r, mask, fuel)))                # WAIT: stay put (no fuel, cannot slip)
            for nid, cost, ch in gw.adj[a]:
                if cost > fuel or not self._inw(nid):
                    continue
                nf = gw.F if gw.station[nid] else fuel - cost
                nm = mask | gw.stopbit[nid]
                if not gw.recoverable(nid, nf, nm):
                    continue
                out.append((ch, (nid, r, nm, nf)))
        else:
            for nid, _, ch in gw.adj[r]:
                if self._inw(nid) and not gw.stopbit[nid]:      # customers' driveways are off-limits to the rival
                    out.append((ch, (a, nid, mask, fuel)))
        return out

    def _inw(self, nid):
        rr, cc = self.gw.rc[nid]
        return abs(rr - self.cx) + abs(cc - self.cy) <= self.W

    def _ordered(self, kids, player):
        if self.ordering == "none":
            return kids
        keyed = []
        for ch, s in kids:
            self.evals += 1
            keyed.append((self.ev(s), ch, s))
        rev = (player == "A") if self.ordering == "own" else (player != "A")
        keyed.sort(key=lambda x: x[0], reverse=rev)             # own: best-first for the side to move
        return [(ch, s) for _, ch, s in keyed]

    def _val(self, st, depth, alpha, beta, player, turn):
        self.nodes += 1
        a, r, mask, fuel = st
        if a == r:
            return -BIG - depth
        if mask == self.gw.full:
            return BIG + depth
        if depth == 0:
            self.evals += 1
            return self.ev(st)
        if player == "R" and turn % self.period != 0:               # rival is too slow to act this turn: it passes
            return self._val(st, depth - 1, alpha, beta, "A", turn + 1)
        kids = self._kids(st, player)
        if not kids:
            if player == "A":
                return -BIG // 2 - depth                         # agent stuck
            self.evals += 1
            return self.ev(st)                                   # rival stuck: just evaluate
        kids = self._ordered(kids, player)
        ab = self.mode == "alphabeta"
        if player == "A":
            v = -math.inf
            for ch, s in kids:
                v = max(v, self._val(s, depth - 1, alpha, beta, "R", turn) - (WAIT_PEN if ch == "." else 0))
                if ab:
                    alpha = max(alpha, v)
                    if alpha >= beta:
                        break
            return v
        v = math.inf
        for _, s in kids:
            v = min(v, self._val(s, depth - 1, alpha, beta, "A", turn + 1))
            if ab:
                beta = min(beta, v)
                if alpha >= beta:
                    break
        return v

    def best_move(self, st, depth, player="A", turn=1):
        """Returns (value, move_char or None). Window is centred on the AGENT's current cell."""
        self.cx, self.cy = self.gw.rc[st[0]]; self.W = WINDOW
        self.nodes = 0; self.evals = 0
        kids = self._ordered(self._kids(st, player), player)
        best_v = -math.inf if player == "A" else math.inf; best_m = None
        alpha, beta = -math.inf, math.inf
        for ch, s in kids:
            v = self._val(s, depth - 1, alpha, beta, "R" if player == "A" else "A", turn if player == "A" else turn + 1)
            if player == "A" and ch == ".":
                v -= WAIT_PEN
            if (player == "A" and v > best_v) or (player == "R" and v < best_v):
                best_v, best_m = v, ch
            if self.mode == "alphabeta":
                if player == "A": alpha = max(alpha, best_v)
                else: beta = min(beta, best_v)
        self.nodes += 1                                          # root
        return best_v, best_m


# ---------------------------------------------------------------- simulator with slips + replanning
def danger_penalty(gw, r):
    """Extra planning cost for cells the rival can reach soon (hops from the rival). Costs only grow, so h_own stays
    admissible and consistent; fuel burn is unchanged."""
    pen = {}
    row = gw.hop[r]
    for i in range(gw.N):
        d = row[i]
        if d <= 2: pen[gw.rc[i]] = DANGER_PEN[0]
        elif d <= 4: pen[gw.rc[i]] = DANGER_PEN[1]
        elif d <= 6: pen[gw.rc[i]] = DANGER_PEN[2]
    return pen


def plan_from(P, h, state, pen=None):
    P._init = state; P.pen = pen or {}
    r = best_first(P, "astar", h=h, name="A*")
    P._init = None; P.pen = {}
    return (r["actions"], r["cost"]) if r["solved"] else (None, None)


def play_game(gw, P, h, weights, slip_p, rival_depth, seed, d_agent=4, rival_every=RIVAL_EVERY, trace=None):
    rng = np.random.default_rng(seed)
    ev_a = make_agent_eval(gw, h, weights); ev_r = make_rival_eval(gw)
    S_a = Searcher(gw, ev_a, "alphabeta", "own", rival_every); S_r = Searcher(gw, ev_r, "alphabeta", "own", rival_every)
    a = gw.cid[gw.w.start]; mask = 0; fuel = gw.F
    far = [i for i in range(gw.N) if gw.hop[a][i] >= 12 and gw.safe[i]]
    r = far[int(rng.integers(len(far)))]
    plan, nominal = plan_from(P, h, (*gw.rc[a], mask, fuel))
    last_plan_r = None
    st = dict(result="draw", cost=0, turns=0, slips=0, replan_slip=0, replan_evade=0, replan_rival=0, waits=0, searches=0,
              nodes=0, nominal=nominal, caught_after_slip=0, last_slip_turn=-99)
    for turn in range(1, T_MAX + 1):
        st["turns"] = turn
        if trace is not None:
            trace.append((a, r))                                   # (agent cell id, rival cell id) at the start of each turn
        near = abs(gw.rc[a][0] - gw.rc[r][0]) + abs(gw.rc[a][1] - gw.rc[r][1]) <= R_DANGER
        used_ab = False
        if abs(gw.rc[a][0] - gw.rc[r][0]) + abs(gw.rc[a][1] - gw.rc[r][1]) <= 8 and r != last_plan_r:
            plan, _ = plan_from(P, h, (*gw.rc[a], mask, fuel), danger_penalty(gw, r))   # strategic layer: rival-aware A*
            last_plan_r = r; st["replan_rival"] += 1
        if near:
            _, move = S_a.best_move((a, r, mask, fuel), d_agent, "A", turn)
            st["searches"] += 1; st["nodes"] += S_a.nodes; used_ab = move is not None
        else:
            move = plan[0] if plan else None
        if move is None:
            st["result"] = "loss"; st["why"] = "stranded"; break
        if move == ".":                                          # agent waits: nothing moves, plan stays valid
            st["waits"] += 1
            if abs(gw.rc[a][0] - gw.rc[r][0]) + abs(gw.rc[a][1] - gw.rc[r][1]) <= R_RIVAL and turn % rival_every == 0:
                _, rm = S_r.best_move((a, r, mask, fuel), rival_depth, "R", turn)
                if rm is not None:
                    r = next(v for v, _, ch in gw.adj[r] if ch == rm and not gw.stopbit[v])
                if a == r:
                    st["result"] = "loss"; st["why"] = "caught"; break
            elif turn % rival_every == 0:
                tt = gw.target(a, mask)
                nb = sorted((x for x in gw.adj[r] if not gw.stopbit[x[0]]), key=lambda x: gw.hop[x[0]][tt])
                if nb: r = nb[0][0]
            continue
        # legal moves from here, then slip
        legal = [(nid, cost, ch) for nid, cost, ch in gw.adj[a] if cost <= fuel]
        intended = next(x for x in legal if x[2] == move)
        actual = intended
        if rng.random() < slip_p:
            others = [x for x in legal if x[2] != move and
                      gw.recoverable(x[0], gw.F if gw.station[x[0]] else fuel - x[1], mask | gw.stopbit[x[0]])]
            if others:
                actual = others[int(rng.integers(len(others)))]
            st["slips"] += 1; st["last_slip_turn"] = turn
        slipped = actual is not intended and actual[0] != intended[0]
        nid, cost, _ = actual
        fuel = gw.F if gw.station[nid] else fuel - cost
        a = nid; mask |= gw.stopbit[nid]; st["cost"] += cost
        if a == r:
            st["result"] = "loss"; st["why"] = "caught"
            st["caught_after_slip"] = int(turn - st["last_slip_turn"] <= 3); break
        if mask == gw.full:
            st["result"] = "win"; break
        # bookkeeping of the plan: replan after a slip (or after an evasive deviation)
        if slipped:
            st["replan_slip"] += 1
            plan, _ = plan_from(P, h, (*gw.rc[a], mask, fuel), danger_penalty(gw, r))
        elif used_ab and (not plan or move != plan[0]):
            st["replan_evade"] += 1
            plan, _ = plan_from(P, h, (*gw.rc[a], mask, fuel), danger_penalty(gw, r))
        else:
            plan = plan[1:] if plan else plan
            if not plan:
                plan, _ = plan_from(P, h, (*gw.rc[a], mask, fuel), danger_penalty(gw, r))
        if plan is None:
            st["result"] = "loss"; st["why"] = "stranded"; break
        # rival's turn (it is slower than the agent: acts every `rival_every`-th turn)
        if turn % rival_every != 0:
            continue
        if abs(gw.rc[a][0] - gw.rc[r][0]) + abs(gw.rc[a][1] - gw.rc[r][1]) <= R_RIVAL:
            _, rm = S_r.best_move((a, r, mask, fuel), rival_depth, "R", turn)
            if rm is not None:
                r = next(v for v, _, ch in gw.adj[r] if ch == rm and not gw.stopbit[v])
        else:                                                    # approach: step toward the agent's next stop
            t = gw.target(a, mask)
            nb = sorted((x for x in gw.adj[r] if not gw.stopbit[x[0]]), key=lambda x: gw.hop[x[0]][t])
            if nb: r = nb[0][0]
        if a == r:
            st["result"] = "loss"; st["why"] = "caught"
            st["caught_after_slip"] = int(turn - st["last_slip_turn"] <= 3); break
    if trace is not None:
        trace.append((a, r))
    return st


# ---------------------------------------------------------------- node-count study (Minimax vs Alpha-Beta)
def sample_roots(gw, n, rng):
    roots = []
    while len(roots) < n:
        a = int(rng.integers(gw.N)); r = int(rng.integers(gw.N))
        d = abs(gw.rc[a][0] - gw.rc[r][0]) + abs(gw.rc[a][1] - gw.rc[r][1])
        if 3 <= d <= R_RIVAL and gw.hop[r][a] < 10 ** 5 and a != r:
            roots.append((a, r, int(rng.integers(0, gw.full)), int(rng.integers(gw.F // 2, gw.F + 1))))
    return roots


def run_node_study(n_roots=30, max_depth=6):
    print(STAMP)
    w = build_world(); gw = GameWorld(w); P = DeliveryProblem(w); h = h_own(P)
    ev = make_agent_eval(gw, h, DEFAULT_W)
    rng = np.random.default_rng(int.from_bytes(("|".join(sorted(CMS_IDS))).encode()[:8], "little"))
    roots = sample_roots(gw, n_roots, rng)
    print(f"\nNODE COUNTS, mean over {n_roots} sampled game positions (window radius {WINDOW})\n{STAMP}")
    print(f"{'depth':>5}{'minimax':>12}{'AB no order':>14}{'AB own order':>14}{'AB worst order':>16}{'saved (own)':>13}{'saved (none)':>14}")
    rows = []
    for d in range(2, max_depth + 1):
        tot = {"mm": 0, "none": 0, "own": 0, "rev": 0}
        for st in roots:
            vals = {}
            for key, mode, order in (("mm", "minimax", "none"), ("none", "alphabeta", "none"),
                                     ("own", "alphabeta", "own"), ("rev", "alphabeta", "reverse")):
                S = Searcher(gw, ev, mode, order); v, m = S.best_move(st, d, "A")
                tot[key] += S.nodes; vals[key] = v
            assert len({round(x, 6) for x in vals.values()}) == 1, f"alpha-beta changed the root value! {vals}"
        mm, no, ow, rv = (tot[k] / n_roots for k in ("mm", "none", "own", "rev"))
        print(f"{d:>5}{mm:>12.0f}{no:>14.0f}{ow:>14.0f}{rv:>16.0f}{100*(1-ow/mm):>12.1f}%{100*(1-no/mm):>13.1f}%")
        rows.append((d, mm, no, ow, rv))
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "partC_nodes.csv"), "w") as f:
        f.write(f"# {STAMP}\ndepth,minimax,ab_no_order,ab_own_order,ab_worst_order,saved_own_pct,saved_none_pct\n")
        for d, mm, no, ow, rv in rows:
            f.write(f"{d},{mm:.1f},{no:.1f},{ow:.1f},{rv:.1f},{100*(1-ow/mm):.1f},{100*(1-no/mm):.1f}\n")
    print("(root values identical for all four searches: pruning never changes the decision)\n" + STAMP)


def run_demo_games(n=6, weights=DEFAULT_W):
    print(STAMP)
    w = build_world(); gw = GameWorld(w); P = DeliveryProblem(w); h = h_own(P)
    print(f"safe cells: {sum(gw.safe)} of {gw.N} free cells")
    print(f"\nDEMO GAMES vs {w.inst['adversary']} (rival depth {w.inst['adversary_search_depth']}, slip {w.slip_p})\n{STAMP}")
    for g in range(n):
        t0 = time.perf_counter()
        s = play_game(gw, P, h, weights, w.slip_p, w.inst["adversary_search_depth"], seed=1000 + g)
        print(f"game {g}: {s['result']:<5} {s.get('why',''):<8} turns {s['turns']:>3} cost {s['cost']:>4} slips {s['slips']:>2} "
              f"replan(slip) {s['replan_slip']:>2} (evade) {s['replan_evade']:>2} (rival) {s['replan_rival']:>2} searches {s['searches']:>3}  {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run_node_study()
    run_demo_games()