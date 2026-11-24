"""SVG route graph for showing a planned delivery path as connected grid nodes."""
import streamlit.components.v1 as components


def show_route_graph(world, route, height=800):
    n = world.n
    cell = 27
    pad = 28
    size = n * cell
    terrain_colors = ["#d98b3a", "#648d49", "#94684d", "#52b9c8"]
    cells = []
    for row in range(n):
        for col in range(n):
            terrain = int(world.grid[row, col])
            color = "#777a7d" if terrain < 0 else terrain_colors[terrain]
            cells.append(
                f'<rect x="{pad + col * cell}" y="{pad + row * cell}" width="{cell}" height="{cell}" '
                f'fill="{color}" stroke="#fff2d9" stroke-width="1"/>'
            )

    points = [(pad + col * cell + cell / 2, pad + row * cell + cell / 2) for row, col in route]
    path_data = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    route_path = (
        f'<path d="{path_data}" fill="none" stroke="#1558d6" stroke-width="5" '
        'stroke-linecap="round" stroke-linejoin="round" opacity=".9"/>'
    )

    arrows = []
    nodes = []
    last_index = len(points) - 1
    for i, (x, y) in enumerate(points):
        if i and i < last_index and i % 3 == 0:
            x0, y0 = points[i - 1]
            x1, y1 = points[i + 1]
            dx, dy = x1 - x0, y1 - y0
            length = max((dx * dx + dy * dy) ** .5, 1)
            ux, uy = dx / length, dy / length
            arrows.append(
                f'<path d="M{x - ux * 5:.1f},{y - uy * 5:.1f} L{x + ux * 5:.1f},{y + uy * 5:.1f}" '
                'stroke="#fff" stroke-width="2.4" marker-end="url(#move-arrow)"/>'
            )
        radius = 6.2 if i in (0, last_index) or i % 10 == 0 else 3.2
        fill = "#fff" if radius > 4 else "#9bc0ff"
        nodes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="#1558d6" stroke-width="2"/>')
        if i in (0, last_index) or i % 10 == 0:
            label = "START" if i == 0 else "END" if i == last_index else str(i)
            nodes.append(
                f'<g transform="translate({x + 8:.1f},{y - 16:.1f})"><rect x="0" y="0" width="{max(30, len(label) * 8 + 10)}" height="19" rx="7" fill="#172b47"/>'
                f'<text x="6" y="13.5" fill="#fff" font-size="10" font-weight="700">{label}</text></g>'
            )

    special = []
    for i, (row, col) in enumerate(world.stops, 1):
        x, y = pad + col * cell + cell / 2, pad + row * cell + cell / 2
        special.append(f'<circle cx="{x}" cy="{y}" r="9" fill="#f4bf53" stroke="#fff" stroke-width="2"/>')
        special.append(f'<text x="{x}" y="{y + 3.5}" text-anchor="middle" fill="#263449" font-size="9" font-weight="800">{i}</text>')
    for row, col in world.stations:
        x, y = pad + col * cell + cell / 2, pad + row * cell + cell / 2
        special.append(f'<path d="M{x},{y-8} L{x+8},{y} L{x},{y+8} L{x-8},{y} Z" fill="#35cabc" stroke="#fff" stroke-width="2"/>')
    start_x, start_y = pad + world.start[1] * cell + cell / 2, pad + world.start[0] * cell + cell / 2
    special.append(f'<rect x="{start_x-7}" y="{start_y-7}" width="14" height="14" rx="2" fill="#ef6257" stroke="#fff" stroke-width="2"/>')

    legend = '''<div class="legend"><span><i class="path"></i>Planned route</span><span><i class="stop">1</i>Delivery stop</span><span><i class="station"></i>Charging station</span><span><i class="depot"></i>Depot</span></div>'''
    svg = (
        f'<svg viewBox="0 0 {size + pad * 2} {size + pad * 2}" role="img" aria-label="Directed route graph on the delivery grid">'
        '<defs><marker id="move-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#fff"/></marker></defs>'
        + "".join(cells) + route_path + "".join(arrows) + "".join(nodes) + "".join(special) + "</svg>"
    )
    graph_html = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:#203049}}
.card{{background:#fff;border:1px solid #e1e8f0;border-radius:16px;padding:16px;box-shadow:0 8px 24px #182c440b}}
.head{{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin:0 0 12px}}
h3{{font-size:16px;margin:0 0 4px}}.sub{{font-size:11px;color:#74849a}}
.badge{{background:#eaf1ff;color:#315fc2;border-radius:99px;padding:7px 10px;font-size:10px;font-weight:750;white-space:nowrap}}
.map{{width:100%;max-height:660px;overflow:auto;display:flex;justify-content:center;background:#f7f2e7;border-radius:11px;padding:8px}}
svg{{display:block;width:min(100%,650px);height:auto;min-width:0}}
.legend{{display:flex;flex-wrap:wrap;gap:16px;margin-top:13px;color:#60718a;font-size:11px}}.legend span{{display:inline-flex;align-items:center;gap:7px}}
.path{{width:20px;height:4px;background:#1558d6;border-radius:4px}}.stop,.station,.depot{{width:15px;height:15px;display:inline-grid;place-items:center;font-style:normal;font-size:9px;font-weight:800}}
.stop{{border-radius:50%;background:#f4bf53;color:#263449}}.station{{background:#35cabc;transform:rotate(45deg);border:2px solid #fff}}.depot{{background:#ef6257;border-radius:3px}}
@media(max-width:520px){{.card{{padding:10px}}.badge{{display:none}}.legend{{gap:10px;font-size:10px}}svg{{min-width:0;width:100%}}}}
</style></head><body><div class="card"><div class="head"><div><h3>Route graph</h3><div class="sub">Each node is a visited grid cell; arrows show movement order.</div></div><div class="badge">{len(route) - 1} MOVES · {len(route)} NODES</div></div>
<div class="map">{svg}</div>{legend}</div></body></html>'''
    components.html(graph_html, height=height, scrolling=False)
