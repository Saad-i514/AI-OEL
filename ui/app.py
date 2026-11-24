"""Streamlit UI (Part E).  Run from the project folder:   streamlit run ui/app.py"""
import sys, os, time, importlib, html, secrets
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import streamlit as st
import members as member_data
importlib.reload(member_data)
MEMBERS, STAMP = member_data.MEMBERS, member_data.STAMP
from oel_instance import TERRAINS
from src.twists import build_world, DIRS, DIR_NAMES
from src.problem import DeliveryProblem
from src.search import bfs, ucs, greedy, astar
from src.heuristics import h_own, h_nearest, h_farthest
from src.game import GameWorld, play_game, DEFAULT_W
import src.llm_agent as llm_agent
importlib.reload(llm_agent)
DispatchEnv, make_backend, run_agent = llm_agent.Env, llm_agent.make_backend, llm_agent.run_agent
from ui.three_scene import show_delivery_scene
from ui.route_graph import show_route_graph
from ui.action_graph import show_action_graph

st.set_page_config(page_title="Autonomous Delivery Agent", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#172b29; --muted:#596d69; --line:#dce6e2; --panel:#fff; --canvas:#f3f6f4;
        --sidebar:#e8efeb; --accent:#14786f; --accent-dark:#0e625a; }
html,body,[class*="css"] { font-family:'DM Sans',sans-serif;color:var(--ink); }
.stApp { background:var(--canvas);color:var(--ink); }
[data-testid="stHeader"] { background:rgba(243,246,244,.96); }
[data-testid="stToolbar"] { right:1rem; }
[data-testid="stAppViewContainer"] > .main .block-container { max-width:1500px;padding:2.1rem 3rem 3.5rem; }
.stApp [data-testid="stSidebar"],.stApp [data-testid="stSidebarCollapsedControl"],
.stApp button[data-testid="collapsedControl"] { display:none!important; }
.brand-lockup { display:flex;align-items:center;gap:12px;padding:8px 2px 22px; }
.brand-mark { width:42px;height:42px;border-radius:12px;display:grid;place-items:center;color:#fff;
  background:#14786f;font:800 17px Manrope,sans-serif; }
.brand-name { color:#18332e;font:750 15px Manrope,sans-serif;letter-spacing:-.2px; }
.brand-sub { color:#60766f;font-size:11px;margin-top:4px;letter-spacing:.09em;text-transform:uppercase; }
.eyebrow { color:#14786f;font-size:11px;font-weight:800;letter-spacing:.13em;text-transform:uppercase; }
.hero-title { color:#172b29;font:800 clamp(28px,3vw,38px) Manrope,sans-serif;letter-spacing:-1.15px;margin:5px 0 5px; }
.hero-subtitle { color:#596d69;font-size:15px;line-height:1.5; }
.status-pill { display:inline-flex;align-items:center;gap:8px;border:1px solid #b6d8cc;background:#e5f2ec;
  color:#176a59;border-radius:8px;padding:9px 12px;font-size:11px;font-weight:800;letter-spacing:.04em;white-space:nowrap; }
.status-dot { width:8px;height:8px;border-radius:50%;background:#1e9a70; }
[data-testid="stMetric"] { background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px;box-shadow:0 2px 7px #1c39300a; }
[data-testid="stMetricLabel"] { color:#526962;font-size:12px;font-weight:700; }
[data-testid="stMetricValue"] { color:#18332e;font:750 27px Manrope,sans-serif; }
[data-testid="stMetricDelta"] { font-size:11px; }
[data-testid="stTabs"] [role="tablist"] { gap:8px;border:0!important;padding:0 0 10px;align-items:center; }
[data-testid="stTabs"] [data-testid="stTab"] { color:#304a44!important;background:#e4ebe7!important;border:1px solid #cedbd5!important;
  border-radius:9px!important;font-size:15px!important;font-weight:700!important;padding:12px 18px!important;min-height:46px;opacity:1!important; }
[data-testid="stTabs"] [data-testid="stTab"] * { color:inherit!important;opacity:1!important; }
[data-testid="stTabs"] [data-testid="stTab"]:hover { color:#0e625a!important;background:#d6e5de!important;border-color:#b6cec3!important; }
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] { color:#fff!important;border-color:#14786f!important;background:#14786f!important; }
[data-testid="stTabs"] [data-testid="stTab"] .react-aria-SelectionIndicator { display:none!important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line)!important;border-radius:12px!important;background:#fff; }
[data-testid="stButton"] button { border-radius:8px;font-weight:700;min-height:43px;transition:background .16s ease,border-color .16s ease; }
[data-testid="stButton"] button:hover { border-color:#80aaa0; }
[data-testid="stButton"] button[kind="primary"] { background:var(--accent);border-color:var(--accent);box-shadow:none; }
[data-testid="stButton"] button[kind="primary"]:hover { background:var(--accent-dark);border-color:var(--accent-dark); }
div[data-testid="stRadio"] [role="radiogroup"] { display:flex;flex-direction:row!important;gap:8px; }
div[data-testid="stRadio"] [role="radiogroup"] label { min-height:46px;padding:10px 17px!important;border:1px solid #d3dfd9;border-radius:9px;background:#fff;color:#304c46!important;
  font-size:15px!important;font-weight:700!important;opacity:1!important; }
div[data-testid="stRadio"] [role="radiogroup"] label * { color:inherit!important;opacity:1!important; }
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) { background:#14786f!important;border-color:#14786f!important;color:#fff!important; }
div[data-testid="stSelectbox"] label,div[data-testid="stSlider"] label,div[data-testid="stMultiSelect"] label,
div[data-testid="stNumberInput"] label,div[data-testid="stTextArea"] label { color:#405a54;font-size:13px;font-weight:700; }
div[data-testid="stSelectbox"] [data-baseweb="select"],div[data-testid="stMultiSelect"] [data-baseweb="select"] { background:#fff!important;color:#172b29!important;border-radius:8px; }
div[data-testid="stSelectbox"] [data-baseweb="select"] > div,div[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
div[data-testid="stSelectbox"] [role="combobox"],div[data-testid="stMultiSelect"] [role="combobox"] { border-radius:8px;background:#fff!important;color:#172b29!important; }
div[data-testid="stSelectbox"] [data-baseweb="select"] *,div[data-testid="stMultiSelect"] [data-baseweb="select"] * { color:#172b29!important; }
[data-testid="stSelectbox"] input,[data-testid="stMultiSelect"] input { color:#172b29!important;background:transparent!important; }
[data-testid="stNumberInputContainer"] { background:#fff!important;border:1px solid #cbd9d2!important;border-radius:8px!important;overflow:hidden; }
[data-testid="stNumberInputContainer"] input { background:#fff!important;color:#172b29!important;caret-color:#14786f; }
[data-testid="stNumberInputContainer"] button { color:#37564f!important;background:#f3f7f4!important;border:0!important; }
[data-testid="stNumberInputContainer"] button:hover { color:#0e625a!important;background:#e5efea!important; }
[data-testid="stTextAreaRootElement"] { background:#fff!important;border:1px solid #cbd9d2!important;border-radius:9px!important;overflow:hidden; }
[data-testid="stTextAreaRootElement"]:focus-within { border-color:#14786f!important;box-shadow:0 0 0 1px #14786f!important; }
[data-testid="stTextAreaRootElement"] textarea { background:#fff!important;color:#172b29!important;caret-color:#14786f!important; }
[data-testid="stTextAreaRootElement"] textarea::placeholder,[data-testid="stNumberInputContainer"] input::placeholder { color:#778982!important; }
[data-testid="stSlider"] [role="group"] > div > div[data-rac][style*="position: absolute"] { background:#14786f!important; }
[data-testid="stCaptionContainer"] { color:#536a64;font-size:14px;line-height:1.5; }
.team-credit { margin-top:14px;padding:15px 17px;border:1px solid #d5e1db;border-radius:10px;background:#e9efeb;
  color:#304c46;font-size:15px;font-weight:600;line-height:1.75; }
.team-credit strong { color:#173a33;font-weight:800; }
div[data-testid="stAlert"] { border-radius:9px; }
@media(max-width:900px) {
 [data-testid="stAppViewContainer"] > .main .block-container { padding:1.2rem 1rem 2rem; }
 .hero-title { font-size:27px; }
 [data-testid="stTabs"] [role="tablist"] { gap:6px;flex-wrap:wrap; }
 [data-testid="stTabs"] [data-testid="stTab"] { padding:10px 11px!important;font-size:14px!important; }
}
</style>
""", unsafe_allow_html=True)


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

header_left, header_right = st.columns([5, 1.4], vertical_alignment="center")
with header_left:
    st.markdown('<div class="eyebrow">Fleet intelligence · CS-370 OEL</div>'
                '<div class="hero-title">Autonomous Delivery Operations</div>'
                '<div class="hero-subtitle">Plan terrain-aware routes, monitor live movement, and evaluate decisions against a competing agent.</div>',
                unsafe_allow_html=True)
with header_right:
    st.markdown('<div class="status-pill"><span class="status-dot"></span>SIMULATION READY</div>',
                unsafe_allow_html=True)

st.markdown(f'<div class="team-credit"><strong>Project team</strong><br>{html.escape(STAMP)}</div>', unsafe_allow_html=True)
st.write("")
overview = st.columns(4)
overview[0].metric("Map coverage", f"{w.n} × {w.n}", "grid cells")
overview[1].metric("Delivery stops", len(w.stops), "available parcels")
overview[2].metric("Charging hubs", len(w.stations), "energy network")
overview[3].metric("Route uncertainty", f"{w.slip_p:.0%}", "movement slip")
st.caption(f"Instance {w.inst['instance_id']}  ·  Rival policy: {w.inst['adversary']}  ·  Active twists: {', '.join(w.inst['twists'])}")
st.write("")

tab_plan, tab_game, tab_dispatcher = st.tabs(["Route planner", "Play against the rival", "LLM dispatcher"])

# ------------------------------------------------------------------------------------------- planner tab
with tab_plan:
    with st.container(border=True):
        st.markdown("#### Route configuration")
        st.caption("Choose a search strategy, delivery stops, and the courier’s fuel limit.")
        c1, c2, c3 = st.columns([1.2, 1.6, 1])
        algo = c1.selectbox("Algorithm", ["A* (own heuristic)", "A* (Manhattan farthest)", "A* (Manhattan nearest)",
                                          "Weighted A* (inadmissible)", "UCS", "BFS", "Greedy best-first"])
        stops = c2.multiselect("Parcels to deliver", list(range(1, len(w.stops) + 1)), default=list(range(1, len(w.stops) + 1)))
        cap = c3.slider("Fuel capacity", 20, 200, int(w.fuel_cap), 5)
        weight_col, button_col = st.columns([2, 1])
        weight = weight_col.slider("Heuristic weight W (for weighted A*)", 1.0, 5.0, 2.0, 0.5)
        go = button_col.button("Plan route", type="primary", use_container_width=True)
    planner_signature = (algo, tuple(stops), int(cap), float(weight))
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
            st.session_state["last_signature"] = planner_signature
    route_is_current = ("last" in st.session_state and
                        st.session_state.get("last_signature") == planner_signature)
    if not route_is_current:
        st.info("Route settings changed. Select **Plan route** to calculate and display the route for these settings.")
    else:
        algo_u, stops_u, r = st.session_state["last"]
        planned_route = route_cells(w, r["actions"]) if r["solved"] else None
        st.markdown("#### Live route view")
        visualization = st.radio("Route visualization", ["3D world", "Route graph"], horizontal=True,
                                 label_visibility="collapsed", key="route_visualization")
        if planned_route:
            if visualization == "3D world":
                st.subheader("Live 3D delivery world")
                st.caption("The courier follows the path produced by the selected search algorithm. Drag to orbit and scroll to zoom.")
                show_delivery_scene(w, planned_route, height=740)
            else:
                st.subheader("Directed route graph")
                st.caption("Every node marks a visited grid cell; directed edges trace the algorithm's exact movement path.")
                show_route_graph(w, planned_route, height=800)
            show_action_graph(planned_route, r["actions"])
            st.write("")
            route_metrics = st.columns(5)
            route_metrics[0].metric("Route cost", r["cost"])
            route_metrics[1].metric("Moves", len(r["actions"]))
            route_metrics[2].metric("Search time", f"{r['ms']:.1f} ms")
            route_metrics[3].metric("Nodes expanded", f"{r['expanded']:,}")
            route_metrics[4].metric("Peak frontier", f"{r['peak_stored']:,}")
            with st.expander("View generated action sequence (text)"):
                st.code("  →  ".join(r["actions"]) or "(already at destination)", language=None)
        else:
            st.error(f"No route found ({r['note']}). Try a larger fuel capacity or select fewer parcels.")

# ------------------------------------------------------------------------------------------- rival tab
with tab_game:
    gw_world, gw, GP, gh = get_game()
    g1, g2, g3 = st.columns(3)
    slip = g1.slider("Slip probability", 0.0, 0.5, float(gw_world.slip_p), 0.02)
    if "game_seed_value" not in st.session_state:
        st.session_state["game_seed_value"] = 14781596
    if g2.button("New random seed", key="new_game_seed", use_container_width=True):
        st.session_state["game_seed_value"] = secrets.randbelow(1_000_000_001)
    seed = g2.number_input(
        "Game seed",
        0,
        10**9,
        key="game_seed_value",
        help=("Enter any whole number from 0 to 1,000,000,000. The seed picks the rival's "
              "starting cell. Keep the same number to replay the same setup, or change it "
              "to try a different rival starting location."),
    )
    g2.caption("Same seed and settings replay the same simulation. Generate a new seed to change the rival's start and random slips.")
    depth = g3.slider("Rival search depth", 2, 6, int(gw_world.inst["adversary_search_depth"]))
    play = st.button("Simulate one game", type="primary")
    game_signature = (float(slip), int(seed), int(depth))
    if play:
        trace = []
        with st.spinner("Simulating decisions and preparing the live 3D replay…"):
            stats = play_game(gw, GP, gh, DEFAULT_W, slip, depth, int(seed), trace=trace)
        st.session_state["game"] = (trace, stats)
        st.session_state["game_signature"] = game_signature
    game_is_current = ("game" in st.session_state and
                       st.session_state.get("game_signature") == game_signature)
    if "game" in st.session_state and not game_is_current:
        st.info("Simulation settings changed. Click **Simulate one game** to generate a new run; the previous replay is hidden.")
    if game_is_current:
        trace, stats = st.session_state["game"]
        st.subheader("Live 3D match")
        st.caption("Each run is computed from the current settings, then replayed here. The same seed reproduces the same run. Follow both agents; pause, replay, orbit, or change playback speed at any time.")
        agent_route = [gw.rc[agent] for agent, _ in trace]
        rival_route = [gw.rc[rival] for _, rival in trace]
        show_delivery_scene(
            gw_world,
            agent_route,
            height=760,
            opponent_route=rival_route,
            game_result=stats,
            autoplay=True,
        )
        with st.expander("Recorded game statistics"):
            stat_cols = st.columns(3)
            stat_cols[0].metric("Turns", stats["turns"])
            stat_cols[1].metric("Travel cost", stats["cost"])
            stat_cols[2].metric("Slips", stats["slips"])
            stat_cols = st.columns(3)
            stat_cols[0].metric("Replans after slips", stats["replan_slip"])
            stat_cols[1].metric("Evasion replans", stats["replan_evade"])
            stat_cols[2].metric("Waits", stats["waits"])

# ------------------------------------------------------------------------------------------- LLM dispatcher tab
with tab_dispatcher:
    st.subheader("Ask the delivery dispatcher")
    st.caption("The configured language model can interpret a delivery request and call the route planner.")
    dispatch_request = st.text_area(
        "Delivery request", "Deliver stop 3 first, then complete the remaining deliveries.",
        key="llm_dispatch_request")
    dispatch_signature = dispatch_request.strip()
    if st.button("Ask dispatcher", type="primary", key="ask_llm_dispatcher"):
        llm = make_backend()
        if llm.name == "mock":
            st.warning("No live LLM is configured. Copy .env.example to .env, add your OPENAI_API_KEY, and restart Streamlit.")
        elif not dispatch_request.strip():
            st.warning("Enter a delivery request first.")
        else:
            dispatch_env = DispatchEnv()
            with st.spinner(f"Asking {llm.model} to plan the delivery…"):
                try:
                    dispatch_result, dispatch_trace = run_agent(
                        dispatch_env, dispatch_request.strip(), llm, validate=True)
                    st.session_state["llm_dispatch_result"] = (dispatch_result, dispatch_trace,
                                                               dispatch_env.last_plan, llm.model)
                    st.session_state["llm_dispatch_signature"] = dispatch_signature
                except Exception as exc:
                    message = str(exc)
                    if any(token in message.lower() for token in ("credential", "authentication", "incorrect api key")) or any(
                        status in message for status in ("401", "403")
                    ):
                        st.error(
                            "The model provider rejected the API key loaded by the app. The local .env key now takes "
                            "precedence over inherited environment values. Confirm it is an active key for the selected "
                            "provider; if it still returns 401, create a fresh key and replace OPENAI_API_KEY in .env. "
                            "Never paste the key into this request box."
                        )
                    elif "404" in message:
                        st.error("The provider could not find this model. Check OEL_LLM_MODEL and the selected provider.")
                    elif "429" in message:
                        st.error("The provider rate-limited the request or the account has no available API quota. Check the provider's usage and billing settings.")
                    else:
                        st.error("The dispatcher request failed. Check the provider connection and model settings, then try again. Raw provider responses are hidden to protect credentials.")
    dispatch_is_current = ("llm_dispatch_result" in st.session_state and
                           st.session_state.get("llm_dispatch_signature") == dispatch_signature)
    if "llm_dispatch_result" in st.session_state and not dispatch_is_current:
        st.info("The delivery request changed. Click **Ask dispatcher** to plan the updated request.")
    if dispatch_is_current:
        dispatch_result, dispatch_trace, dispatch_plan, dispatch_model = st.session_state["llm_dispatch_result"]
        st.caption(f"Model: {dispatch_model} | Validated tool calls: {dispatch_result['tool_calls']}")
        if dispatch_plan:
            st.success(f"Route planned: {len(dispatch_plan['actions'])} moves, cost {dispatch_plan['cost']}.")
            delivery_order = dispatch_plan.get("delivery_order", [])
            if delivery_order:
                order_label = " → ".join(f"Stop {stop_id}" for stop_id in delivery_order)
                st.markdown(f"**Delivery order:** {order_label}")
                st.caption("A courier may pass through another stop cell without delivering there. The requested order controls when each parcel is marked delivered.")
            st.subheader("3D delivery route")
            st.caption("The courier is following the validated route. Pause, orbit, zoom, or adjust playback in the scene.")
            dispatch_route = route_cells(w, dispatch_plan["actions"])
            show_delivery_scene(w, dispatch_route, height=740, autoplay=True)
            with st.expander("View planned action sequence"):
                st.code("  →  ".join(dispatch_plan["actions"]) or "(already at destination)", language=None)
        elif dispatch_result.get("final"):
            st.info(dispatch_result["final"].get("message", str(dispatch_result["final"])))
        with st.expander("Dispatcher reasoning and tool trace"):
            for step, thought, action, observation in dispatch_trace:
                st.markdown(f"**Step {step}** — {thought}")
                st.code(action, language=None)
                if observation:
                    st.caption(observation)
