"""Part B formulation: the whole delivery task as ONE search problem.

State  = (r, c, mask, fuel)
   r, c  : position
   mask  : bit i set  <=> parcel i delivered   (delivered on ENTERING stop i's cell)
   fuel  : integer fuel left (twist 1). Entering a station refuels to fuel_cap.
Action = move N/S/W/E  (only if cell free, edge allowed [twist 2: one-way], fuel >= terrain cost)
Cost   = terrain cost of the entered cell  (fuel burned = same number)
Goal   = mask == all ones
Run:  python -m src.problem   (sanity check + reference solver)
"""
import time, heapq, math
from members import STAMP
from src.twists import build_world, DIR_NAMES


class DeliveryProblem:
    def __init__(self, world, stop_ids=None, start=None, init_state=None):
        """stop_ids: optional subset of stop indices (0-based); start: optional start cell.
        Both are used to build REDUCED problems for DLS/IDS tests."""
        self.w = world
        self.start = tuple(start) if start is not None else world.start
        self.blocked = set()                          # cells the planner must not enter (e.g. 'avoid mud' requests)
        self.first_bit = 0                            # if set: this stop must be delivered before any other
        self.pen = {}                                 # optional extra step cost per cell (rival danger zones); fuel is NOT affected
        self._init = init_state                       # lets the simulator REPLAN from any (r, c, mask, fuel)
        ids = list(range(len(world.stops))) if stop_ids is None else list(stop_ids)
        self.stops = [world.stops[i] for i in ids]
        self.stop_bit = {s: 1 << k for k, s in enumerate(self.stops)}
        self.full = (1 << len(self.stops)) - 1
        self.F = world.fuel_cap

    def initial_state(self):
        if self._init is not None:
            return self._init
        r, c = self.start
        return (r, c, 0, self.F)

    def is_goal(self, s):
        return s[2] == self.full

    def successors(self, s):
        """yield (action_char, next_state, step_cost)"""
        r, c, mask, fuel = s
        for nr, nc, cost, d in self.w.nbrs[(r, c)]:
            if cost > fuel or (nr, nc) in self.blocked:
                continue
            nf = fuel - cost
            if (nr, nc) in self.w.station_set:
                nf = self.F
            b = self.stop_bit.get((nr, nc), 0)
            if self.first_bit and not (mask & self.first_bit) and b != self.first_bit:
                b = 0                                  # precedence constraint: other stops do not count yet
            nm = mask | b
            yield DIR_NAMES[d], (nr, nc, nm, nf), cost + self.pen.get((nr, nc), 0)

    @staticmethod
    def dom_key(s):
        """States with equal dom_key are comparable: more fuel at <= cost dominates (used for pruning)."""
        return (s[0], s[1], s[2])


def path_cost(problem, actions):
    """Re-simulate an action string; returns (cost, ok). Used to VERIFY every planner's answer."""
    s = problem.initial_state(); total = 0
    for a in actions:
        for act, ns, c in problem.successors(s):
            if act == a:
                s = ns; total += c; break
        else:
            return total, False
    return total, problem.is_goal(s)


def reference_solver(problem):
    """Quick Dijkstra with fuel-dominance pruning. ONLY for the solvability check; graded UCS is in search.py."""
    s0 = problem.initial_state()
    pq = [(0, 0, s0, "")]; best_fuel = {}; cnt = 0; expanded = 0
    while pq:
        g, _, s, acts = heapq.heappop(pq)
        k = problem.dom_key(s)
        if best_fuel.get(k, -1) >= s[3]:
            continue
        best_fuel[k] = s[3]; expanded += 1
        if problem.is_goal(s):
            return g, acts, expanded
        for a, ns, c in problem.successors(s):
            if best_fuel.get(problem.dom_key(ns), -1) < ns[3]:
                cnt += 1; heapq.heappush(pq, (g + c, cnt, ns, acts + a))
    return None, None, expanded


if __name__ == "__main__":
    print(STAMP)
    w = build_world(); p = DeliveryProblem(w)
    t0 = time.perf_counter(); g, acts, ex = reference_solver(p); dt = time.perf_counter() - t0
    print(f"Solvable: {g is not None} | optimal cost {g} | steps {len(acts)} | expanded {ex} | {dt:.2f}s")
    print("verify:", path_cost(p, acts))
    print(STAMP)
