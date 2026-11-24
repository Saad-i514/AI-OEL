"""Part D: the tools the LLM may call, the validation layer, the independent route verifier, the 20 requests.
Tools:  plan_route (calls our Part B planners)  |  simulate (steps the world)  |  fuel_report (custom tool, fuel twist)
"""
import re, json
from members import STAMP
from oel_instance import TERRAINS
from src.twists import build_world
from src.problem import DeliveryProblem
from src.search import astar, ucs, bfs, greedy
from src.heuristics import h_own

ALGOS = ("astar", "ucs", "bfs", "greedy")
TOOLS = {
    "plan_route": {"stops": "list[int] 1..N (default all)", "first": "int, a stop to deliver first (optional)",
                   "avoid_terrain": "list of road|grass|mud|ford (optional)", "max_cost": "number (optional)",
                   "algorithm": "astar|ucs|bfs|greedy (default astar)"},
    "simulate": {"actions": "string of moves N,S,W,E from the depot"},
    "fuel_report": {"actions": "string of moves N,S,W,E from the depot"},
}


class Env:
    def __init__(self):
        self.w = build_world()
        self.N = len(self.w.stops)
        self.last_plan = None          # last FEASIBLE plan_route result (what the harness evaluates)
        self.full = DeliveryProblem(self.w)

    # ---------- prompt text ----------
    def describe(self):
        w = self.w
        stops = "; ".join(f"stop {i+1} at {s}" for i, s in enumerate(w.stops))
        costs = ", ".join(f"{k} cost {v}" for k, v in w.inst["terrain_cost"].items())
        return (f"City grid {w.n}x{w.n}. Depot at {w.start}. Parcels: {stops}. Terrain: {costs}. "
                f"Fuel capacity {w.fuel_cap}; fuel burned = terrain cost of each entered cell; charging stations at "
                f"{w.stations} refuel to full. Some streets are one-way. Narrow alleys (cells with 2+ blocked neighbours) cost 2 extra. Moves: N,S,W,E.")

    # ---------- problem construction ----------
    def make_problem(self, stops, first=None, avoid=()):
        P = DeliveryProblem(self.w, stop_ids=[s - 1 for s in stops])
        P.blocked = {(r, c) for (r, c) in self.w.nbrs if TERRAINS[int(self.w.grid[r, c])] in set(avoid)}
        if first:
            P.first_bit = P.stop_bit[self.w.stops[first - 1]]
        return P

    def replay(self, actions, P=None):
        P = P or self.full
        s = P.initial_state(); total = 0; order = []; fuels = [s[3]]; refuels = 0; used = []
        for i, a in enumerate(actions):
            for act, ns, c in P.successors(s):
                if act == a:
                    newly = ns[2] & ~s[2]
                    for k, st in enumerate(P.stops):
                        if newly >> k & 1:
                            order.append(self.w.stops.index(st) + 1)
                    if ns[3] == P.F and (ns[0], ns[1]) in self.w.station_set and ns[3] > s[3] - c:
                        refuels += 1; used.append((ns[0], ns[1]))
                    s = ns; total += c; fuels.append(ns[3]); break
            else:
                return dict(valid=False, error_step=i + 1, cost=total, delivered=order, goal=False)
        return dict(valid=True, cost=total, delivered=order, goal=P.is_goal(s), final_position=(s[0], s[1]),
                    fuel_left=s[3], min_fuel=min(fuels), refuels=refuels, stations_used=used)

    # ---------- validation layer ----------
    def validate(self, name, args):
        """Checks tool name and arguments against the instance. Returns (ok, message, clean_args)."""
        if name not in TOOLS:
            return False, f"REJECTED: unknown tool '{name}'. Valid tools: {list(TOOLS)}", None
        if not isinstance(args, dict):
            return False, "REJECTED: args must be a JSON object", None
        bad = [k for k in args if k not in TOOLS[name]]
        if bad:
            return False, f"REJECTED: unknown argument(s) {bad}. Valid arguments for {name}: {list(TOOLS[name])}", None
        clean = {}
        if name == "plan_route":
            if "stops" in args:
                st = args["stops"]
                if not (isinstance(st, list) and st and all(isinstance(x, int) and not isinstance(x, bool) for x in st)
                        and all(1 <= x <= self.N for x in st) and len(set(st)) == len(st)):
                    return False, f"REJECTED: stops must be distinct integers in 1..{self.N}, got {st}", None
                clean["stops"] = st
            stops = clean.get("stops", list(range(1, self.N + 1)))
            if args.get("first") is not None:
                f = args["first"]
                if f not in stops or isinstance(f, bool):
                    return False, f"REJECTED: first={f} must be one of the requested stops {stops}", None
                clean["first"] = f
            if args.get("avoid_terrain") is not None:
                av = args["avoid_terrain"]
                if not (isinstance(av, list) and all(t in TERRAINS for t in av)):
                    return False, f"REJECTED: avoid_terrain {av} invalid; valid terrains are {list(TERRAINS)}", None
                clean["avoid_terrain"] = av
            if args.get("max_cost") is not None:
                mc = args["max_cost"]
                if isinstance(mc, bool) or not isinstance(mc, (int, float)) or mc <= 0:
                    return False, f"REJECTED: max_cost must be a positive number, got {mc}", None
                clean["max_cost"] = mc
            if args.get("algorithm") is not None:
                if args["algorithm"] not in ALGOS:
                    return False, f"REJECTED: algorithm must be one of {list(ALGOS)}, got '{args['algorithm']}'", None
                clean["algorithm"] = args["algorithm"]
        else:
            a = args.get("actions")
            if not (isinstance(a, str) and a and re.fullmatch(r"[NSWE]+", a)):
                return False, "REJECTED: actions must be a non-empty string over the letters N,S,W,E", None
            clean["actions"] = a
        return True, "ok", clean

    def coerce_raw(self, name, args):
        """NO validation: typical permissive behaviour -- anything unrecognised is silently ignored."""
        if name == "plan_route":
            c = {}
            st = args.get("stops")
            if isinstance(st, list):
                st = [x for x in st if isinstance(x, int) and 1 <= x <= self.N]
                if st: c["stops"] = list(dict.fromkeys(st))
            f = args.get("first")
            if isinstance(f, int) and f in c.get("stops", range(1, self.N + 1)): c["first"] = f
            av = args.get("avoid_terrain")
            if isinstance(av, list): c["avoid_terrain"] = [t for t in av if t in TERRAINS]
            mc = args.get("max_cost")
            if isinstance(mc, (int, float)) and not isinstance(mc, bool) and mc > 0: c["max_cost"] = mc
            al = args.get("algorithm")
            c["algorithm"] = al if al in ALGOS else "astar"
            return c
        a = args.get("actions", "")
        return {"actions": "".join(ch for ch in str(a) if ch in "NSWE")}

    # ---------- the tools ----------
    def plan_route(self, stops=None, first=None, avoid_terrain=None, max_cost=None, algorithm="astar"):
        stops = stops or list(range(1, self.N + 1)); avoid = avoid_terrain or []
        P = self.make_problem(stops, first, avoid); h = h_own(P)
        r = {"astar": lambda: astar(P, h), "ucs": lambda: ucs(P), "bfs": lambda: bfs(P),
             "greedy": lambda: greedy(P, h)}[algorithm]()
        if not r["solved"]:
            return {"feasible": False, "reason": "no route exists under these constraints (blocked terrain, fuel or one-way streets)"}
        if max_cost is not None and r["cost"] > max_cost:
            return {"feasible": False, "reason": f"best route found costs {r['cost']} which exceeds max_cost {max_cost}"}
        rep = self.replay(r["actions"], P)
        out = {"feasible": True, "actions": r["actions"], "cost": r["cost"], "steps": len(r["actions"]),
               "delivery_order": rep["delivered"], "algorithm": algorithm}
        self.last_plan = out
        return out

    def simulate(self, actions):
        r = self.replay(actions)
        if not r["valid"]:
            return {"valid": False, "error": f"move {r['error_step']} is illegal (blocked, one-way or out of fuel)",
                    "delivered_so_far": r["delivered"]}
        return {"valid": True, "delivered": r["delivered"], "cost": r["cost"], "final_position": r["final_position"],
                "fuel_left": r["fuel_left"], "all_requested_stops_delivered": r["goal"]}

    def fuel_report(self, actions):
        r = self.replay(actions)
        if not r["valid"]:
            return {"feasible": False, "error": f"move {r['error_step']} is illegal (blocked, one-way or out of fuel)"}
        return {"feasible": True, "min_fuel_on_route": r["min_fuel"], "refuels": r["refuels"],
                "stations_used": r["stations_used"], "fuel_capacity": self.w.fuel_cap}

    def call(self, name, args, validate):
        """Executes one tool call. Returns dict(obs, rejected, wrong_tool, bad_args)."""
        ok, msg, clean = self.validate(name, args if isinstance(args, dict) else {})
        wrong_tool = name not in TOOLS
        bad_args = (not ok) and not wrong_tool
        if validate:
            if not ok:
                return dict(obs={"error": msg}, rejected=True, wrong_tool=wrong_tool, bad_args=bad_args)
            use = clean
        else:                                                  # raw mode: no checking
            if wrong_tool:
                return dict(obs={"error": f"tool '{name}' not found"}, rejected=False, wrong_tool=True, bad_args=False)
            use = self.coerce_raw(name, args if isinstance(args, dict) else {})
        try:
            obs = getattr(self, name)(**use)
        except Exception as e:                                 # raw mode can crash: report it like a real tool would
            obs = {"error": f"{type(e).__name__}: {e}"}
        return dict(obs=obs, rejected=False, wrong_tool=wrong_tool, bad_args=bad_args)

    def verify(self, spec, actions):
        """Independent judge: replay the route under the REQUEST's constraints."""
        P = self.make_problem(spec.get("stops") or list(range(1, self.N + 1)), spec.get("first"), spec.get("avoid_terrain") or [])
        r = self.replay(actions, P)
        return bool(r["valid"] and r["goal"] and (spec.get("max_cost") is None or r["cost"] <= spec["max_cost"]))


# ---------------------------------------------------------------------------------------- 20 requests
# level 1: plain | 2: one constraint | 3: two constraints | 4: tight / infeasible / unusual
REQUESTS = [
    (1, 1, "Deliver all the parcels with the cheapest route.", {}),
    (2, 1, "Just deliver stop 3, nothing else.", {"stops": [3]}),
    (3, 1, "Deliver stops 1 and 2 only.", {"stops": [1, 2]}),
    (4, 1, "Plan a route for stops 4, 5 and 6.", {"stops": [4, 5, 6]}),
    (5, 2, "Deliver all parcels, but stop 3 must be the first one delivered.", {"first": 3}),
    (6, 2, "Deliver stops 2, 5 and 6, with stop 6 first.", {"stops": [2, 5, 6], "first": 6}),
    (7, 2, "Deliver stops 4 and 5 and stay out of the fords.", {"stops": [4, 5], "avoid_terrain": ["ford"]}),
    (8, 2, "Deliver stop 5 without crossing any ford.", {"stops": [5], "avoid_terrain": ["ford"]}),
    (9, 2, "Deliver all parcels with a total cost of at most 350.", {"max_cost": 350}),
    (10, 2, "Deliver stops 3, 4 and 5 for a total cost under 120.", {"stops": [3, 4, 5], "max_cost": 119}),
    (11, 3, "Deliver stops 4 and 5, stop 5 first, and avoid fords.", {"stops": [4, 5], "first": 5, "avoid_terrain": ["ford"]}),
    (12, 3, "Deliver stops 4 and 5, no fords, total cost at most 120.", {"stops": [4, 5], "avoid_terrain": ["ford"], "max_cost": 120}),
    (13, 3, "Deliver stops 2, 3 and 4 with stop 4 first, and keep the cost at or below 320.", {"stops": [2, 3, 4], "first": 4, "max_cost": 320}),
    (14, 3, "Deliver all parcels, avoiding both mud and ford.", {"avoid_terrain": ["mud", "ford"]}),
    (15, 3, "Deliver stops 5 and 6, stop 6 first, no fords, cost at most 160.", {"stops": [5, 6], "first": 6, "avoid_terrain": ["ford"], "max_cost": 160}),
    (16, 4, "Deliver every parcel for less than 100 in total cost.", {"max_cost": 99}),
    (17, 4, "Deliver all parcels but never drive on a road.", {"avoid_terrain": ["road"]}),
    (18, 4, "Deliver stops 1, 2 and 3, starting with stop 2.", {"stops": [1, 2, 3], "first": 2}),
    (19, 4, "Deliver every parcel except stop 6.", {"stops": [1, 2, 3, 4, 5]}),
    (20, 4, "Deliver stops 4 and 5 using the fast greedy algorithm.", {"stops": [4, 5], "algorithm": "greedy"}),
]


def ground_truth(env, spec):
    """The 'direct planner' baseline: call plan_route with the exact arguments of the request."""
    a = dict(spec)
    r = env.plan_route(**a)
    return r