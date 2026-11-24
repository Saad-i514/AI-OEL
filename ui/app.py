"""Streamlit UI (Part E).  Run from the project folder:   streamlit run ui/app.py"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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


def draw_3d(w, route=None, title="", agent=None, rival=None, trail=None):
    """Render terrain as raised tiles and barriers, with route markers above them."""
    colors = ["#d0d0d0", "#aed581", "#a1887f", "#81d4fa", "#222222"]
    n = w.grid.shape[0]
    heights = np.zeros((n, n), dtype=float)
    facecolors = []
    faces = []

    def shade(color, factor):
        from matplotlib.colors import to_rgb, to_hex
        return to_hex(np.asarray(to_rgb(color)) * factor)

    for row in range(n):
        for col in range(n):
            terrain = int(w.grid[row, col])
            heights[row, col] = 11.0 if terrain < 0 else float(w.inst["terrain_cost"][TERRAINS[terrain]])

    for row in range(n):
        for col in range(n):
            terrain = int(w.grid[row, col])
            blocked = terrain < 0
            height = heights[row, col]
            x0, x1, y0, y1 = col - 0.5, col + 0.5, row - 0.5, row + 0.5
            color = colors[4 if blocked else terrain]
            faces.append([(x0, y0, height), (x1, y0, height), (x1, y1, height), (x0, y1, height)])
            facecolors.append(color)
            for dr, dc, side in ((-1, 0, (x0, y0, x1, y0)), (1, 0, (x1, y1, x0, y1)),
                                 (0, -1, (x0, y1, x0, y0)), (0, 1, (x1, y0, x1, y1))):
                nr, nc = row + dr, col + dc
                base = heights[nr, nc] if 0 <= nr < n and 0 <= nc < n else 0.0
                if base >= height:
                    continue
                sx0, sy0, sx1, sy1 = side
                faces.append([(sx0, sy0, base), (sx1, sy1, base), (sx1, sy1, height), (sx0, sy0, height)])
                facecolors.append(shade(color, 0.68))

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(faces, facecolors=facecolors, edgecolors="white", linewidths=0.12))

    # One-way streets are shown as small red arrows just above the tile tops.
    for (row, col), neighbors in w.nbrs.items():
        for nr, nc, _, _ in neighbors:
            if (row, col) not in {(rr, cc) for rr, cc, _, _ in w.nbrs[(nr, nc)]}:
                z = heights[row, col] + 0.18
                ax.quiver(col, row, z, nc - col, nr - row, 0, color="crimson", linewidth=0.8,
                          arrow_length_ratio=0.35, length=0.42, normalize=False)

    def point_z(cell, offset=0.5):
        return heights[cell[0], cell[1]] + offset

    if route:
        ys, xs = zip(*route)
        zs = [point_z(cell, 0.25) for cell in route]
        ax.plot(xs, ys, zs, color="royalblue", linewidth=3, zorder=10, label="planned route")
    if trail and len(trail) > 1:
        ys, xs = zip(*trail)
        zs = [point_z(cell, 0.55) for cell in trail]
        ax.plot(xs, ys, zs, color="#1746c7", linewidth=2.6, alpha=0.9, zorder=12, label="agent trail")

    stations = np.asarray(w.stations)
    if len(stations):
        ax.scatter(stations[:, 1], stations[:, 0], [point_z(s, 0.5) for s in stations],
                   marker="^", c="limegreen", edgecolors="k", s=75, depthshade=False, label="charging station")
    ax.scatter([w.start[1]], [w.start[0]], [point_z(w.start, 0.65)], marker="s", c="red", s=85,
               depthshade=False, label="depot")
    stops = np.asarray(w.stops)
    ax.scatter(stops[:, 1], stops[:, 0], [point_z(s, 0.65) for s in w.stops], marker="*", c="yellow",
               edgecolors="k", s=150, depthshade=False, label="delivery stop")
    if agent is not None:
        ax.scatter([agent[1]], [agent[0]], [point_z(agent, 0.9)], marker="o", c="dodgerblue", edgecolors="k",
                   s=120, depthshade=False, label="agent")
    if rival is not None:
        ax.scatter([rival[1]], [rival[0]], [point_z(rival, 0.9)], marker="X", c="magenta", edgecolors="k",
                   s=140, depthshade=False, label="rival")

    handles = [Patch(color=colors[i], label=f"{terrain} ({w.inst['terrain_cost'][terrain]})")
               for i, terrain in enumerate(TERRAINS)]
    handles.append(Patch(color=colors[4], label="blocked barrier"))
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5); ax.set_zlim(0, 13)
    ax.set_xlabel("Column"); ax.set_ylabel("Row"); ax.set_zlabel("Terrain cost / height")
    ax.set_title(title + "\n" + STAMP, fontsize=8)
    ax.view_init(elev=38, azim=-58)
    ax.set_box_aspect((1, 1, 0.42))
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
        st.pyplot(draw_3d(w, route_cells(w, r["actions"]) if r["solved"] else None,
                          f"{algo_u}: parcels {stops_u}"), clear_figure=True)
        st.caption("3D height represents terrain cost; black columns are blocked cells.")
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
    play = st.button("Simulate one game", type="primary")
    if play:
        trace = []
        with st.spinner("Playing..."):
            stats = play_game(gw, GP, gh, DEFAULT_W, slip, depth, int(seed), trace=trace)
        st.session_state["game"] = (trace, stats)
    if "game" in st.session_state:
        trace, stats = st.session_state["game"]
        controls = st.columns([2, 2, 1])
        turn = controls[0].slider("Turn", 0, len(trace) - 1, 0)
        delay = controls[1].slider("Animation speed (seconds per turn)", 0.02, 0.25, 0.06, 0.01)
        animate = controls[2].button("▶ Animate route", type="primary")
        left, right = st.columns([3, 2])
        with left:
            chart = st.empty()
            progress = st.progress(0) if animate else None
            frame_step = max(1, (len(trace) - 1) // 119) if animate else 1
            frames = list(range(0, len(trace), frame_step)) if animate else [turn]
            if animate and frames[-1] != len(trace) - 1:
                frames.append(len(trace) - 1)
            for frame_index in frames:
                a_id, r_id = trace[frame_index]
                trail = [gw.rc[a] for a, _ in trace[:frame_index + 1]]
                fig = draw_3d(gw_world, None, f"Turn {frame_index} of {len(trace) - 1}",
                              agent=gw.rc[a_id], rival=gw.rc[r_id], trail=trail)
                chart.pyplot(fig, clear_figure=True)
                plt.close(fig)
                if animate:
                    progress.progress((frames.index(frame_index) + 1) / len(frames))
                    time.sleep(delay)
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
