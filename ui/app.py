"""Streamlit UI (Part E).  Run from the project folder:   streamlit run ui/app.py"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import streamlit as st
from members import MEMBERS, STAMP
from oel_instance import TERRAINS
from src.twists import build_world, DIRS, DIR_NAMES
from src.problem import DeliveryProblem
from src.search import bfs, ucs, greedy, astar
from src.heuristics import h_own, h_nearest, h_farthest
from src.game import GameWorld, play_game, DEFAULT_W

st.set_page_config(page_title="Autonomous Delivery Agent", layout="wide")


@st.cache_resource
def get_world():
    return build_world()


@st.cache_resource
def get_game():
    w = build_world()                          # separate world object: the planner tab edits fuel capacity on its own copy
    P = DeliveryProblem(w)
    return w, GameWorld(w), P, h_own(P)


def draw(w, route=None, title="", agent=None, rival=None, trail=None):
    cmap = ListedColormap(["#222222", "#d0d0d0", "#aed581", "#a1887f", "#81d4fa"])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(w.grid + 1, cmap=cmap, vmin=0, vmax=4)
    for (r, c), lst in w.nbrs.items():                              # one-way streets
        for nr, nc, _, _ in lst:
            if (r, c) not in {(a, b) for a, b, _, _ in w.nbrs[(nr, nc)]}:
                ax.annotate("", xy=(c + 0.3 * (nc - c), r + 0.3 * (nr - r)), xytext=(c, r),
                            arrowprops=dict(arrowstyle="->", color="crimson", lw=0.7))
    if route:
        ys, xs = zip(*route); ax.plot(xs, ys, color="royalblue", lw=2.2, zorder=3)
    if trail and len(trail) > 1:
        ys, xs = zip(*trail); ax.plot(xs, ys, color="royalblue", lw=1.2, alpha=0.6, zorder=3)
    for s in w.stations:
        ax.scatter([s[1]], [s[0]], marker="^", c="limegreen", edgecolors="k", s=110, zorder=4)
    ax.scatter([w.start[1]], [w.start[0]], marker="s", c="red", s=130, zorder=5)
    for i, (r, c) in enumerate(w.stops, 1):
        ax.scatter([c], [r], marker="*", c="yellow", edgecolors="k", s=170, zorder=5)
        ax.text(c, r - 0.65, str(i), ha="center", fontsize=8, weight="bold")
    if agent:
        ax.scatter([agent[1]], [agent[0]], marker="o", c="dodgerblue", edgecolors="k", s=190, zorder=7, label="agent")
    if rival:
        ax.scatter([rival[1]], [rival[0]], marker="X", c="magenta", edgecolors="k", s=210, zorder=7, label="rival (interceptor)")
    leg = [Patch(color=col, label=f"{t} ({w.inst['terrain_cost'][t]})")
           for col, t in zip(["#d0d0d0", "#aed581", "#a1887f", "#81d4fa"], TERRAINS)]
    leg += [Patch(color="#222222", label="blocked")]
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=leg + h, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_title(title + "\n" + STAMP, fontsize=7.5)
    fig.tight_layout()
    return fig


def route_cells(w, actions):
    r, c = w.start; cells = [(r, c)]
    for a in actions:
        dr, dc = DIRS[DIR_NAMES.index(a)]; r, c = r + dr, c + dc; cells.append((r, c))
    return cells


w = get_world()
st.title("Autonomous Delivery Agent under Uncertainty and Competition")
st.markdown("**" + "  |  ".join(f"{n} ({c})" for n, c in MEMBERS) + "**")        # names + CMS IDs on screen
st.caption(f"CS-370 OEL | instance {w.inst['instance_id']} | {w.n}x{w.n} grid | rival: {w.inst['adversary']} "
           f"(depth {w.inst['adversary_search_depth']}) | slip {w.slip_p} | twists: {', '.join(w.inst['twists'])}")

tab_plan, tab_game = st.tabs(["Route planner", "Play against the rival"])

# ------------------------------------------------------------------------------------------- planner tab
with tab_plan:
    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Algorithm", ["A* (own heuristic)", "A* (Manhattan farthest)", "A* (Manhattan nearest)",
                                      "Weighted A* (inadmissible)", "UCS", "BFS", "Greedy best-first"])
    stops = c2.multiselect("Parcels to deliver", list(range(1, len(w.stops) + 1)), default=list(range(1, len(w.stops) + 1)))
    cap = c3.slider("Fuel capacity", 20, 200, int(w.fuel_cap), 5)
    weight = st.slider("Heuristic weight W (weighted A* only)", 1.0, 5.0, 2.0, 0.5)
    go = st.button("Plan route", type="primary")
    if go or "last" not in st.session_state:
        if not stops:
            st.warning("Select at least one parcel.")
        else:
            w.fuel_cap = cap
            P = DeliveryProblem(w, stop_ids=[s - 1 for s in stops])
            t0 = time.perf_counter()
            if algo.startswith("A* (own"):             r = astar(P, h_own(P), name="A*")
            elif algo.startswith("A* (Manhattan far"): r = astar(P, h_farthest(P), name="A*")
            elif algo.startswith("A* (Manhattan near"): r = astar(P, h_nearest(P), name="A*")
            elif algo.startswith("Weighted"):          r = astar(P, h_own(P), weight=weight, name="WA*")
            elif algo == "UCS":                        r = ucs(P)
            elif algo == "BFS":                        r = bfs(P)
            else:                                      r = greedy(P, h_own(P))
            r["ms"] = (time.perf_counter() - t0) * 1e3
            st.session_state["last"] = (algo, stops, r)
    algo_u, stops_u, r = st.session_state["last"]
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(draw(w, route_cells(w, r["actions"]) if r["solved"] else None, f"{algo_u}: parcels {stops_u}"))
    with right:
        st.subheader("Result")
        if r["solved"]:
            st.metric("Total cost", r["cost"])
            a, b = st.columns(2)
            a.metric("Moves", len(r["actions"])); b.metric("Nodes expanded", r["expanded"])
            a.metric("Runtime (ms)", f"{r['ms']:.1f}"); b.metric("Peak stored nodes", r["peak_stored"])
            st.text_area("Action string", r["actions"], height=90)
        else:
            st.error(f"No route found ({r['note']}). Try a larger fuel capacity.")
        st.caption(STAMP)

# ------------------------------------------------------------------------------------------- rival tab
with tab_game:
    gw_world, gw, GP, gh = get_game()
    g1, g2, g3 = st.columns(3)
    slip = g1.slider("Slip probability", 0.0, 0.5, float(gw_world.slip_p), 0.02)
    seed = g2.number_input("Game seed (sets the rival's start cell)", 0, 10**9, 14781596)
    depth = g3.slider("Rival search depth", 2, 6, int(gw_world.inst["adversary_search_depth"]))
    play = st.button("Play one game (takes a few seconds)", type="primary")
    if play:
        trace = []
        with st.spinner("Playing..."):
            stats = play_game(gw, GP, gh, DEFAULT_W, slip, depth, int(seed), trace=trace)
        st.session_state["game"] = (trace, stats)
    if "game" in st.session_state:
        trace, stats = st.session_state["game"]
        turn = st.slider("Turn", 0, len(trace) - 1, 0)
        a_id, r_id = trace[turn]
        trail = [gw.rc[a] for a, _ in trace[:turn + 1]]
        left, right = st.columns([3, 2])
        with left:
            st.pyplot(draw(gw_world, None, f"Turn {turn} of {len(trace) - 1}", agent=gw.rc[a_id], rival=gw.rc[r_id], trail=trail))
        with right:
            st.subheader("Game result")
            why = stats.get("why", "")
            (st.success if stats["result"] == "win" else st.error if stats["result"] == "loss" else st.info)(
                f"{stats['result'].upper()} {('(' + why + ')') if why else ''}")
            a, b = st.columns(2)
            a.metric("Turns", stats["turns"]); b.metric("Travel cost", stats["cost"])
            a.metric("Slips", stats["slips"]); b.metric("Replans after slips", stats["replan_slip"])
            a.metric("Evasion replans", stats["replan_evade"]); b.metric("Waits", stats["waits"])
            st.caption(STAMP)