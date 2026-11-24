"""Render the algorithm's exact action sequence as a directed node graph."""
import html

import streamlit.components.v1 as components


def show_action_graph(route, actions, height=None):
    """Draw each route state as a node, connected in movement order."""
    route = list(route)
    actions = list(actions)
    count = len(route)
    if not count:
        return

    columns = 16
    node_w, node_h, gap_x, gap_y = 76, 48, 14, 24
    step_x, step_y = node_w + gap_x, node_h + gap_y
    rows = (count + columns - 1) // columns
    width = 24 + columns * step_x - gap_x
    height = height or (100 + rows * step_y)
    points = []
    for index in range(count):
        row, offset = divmod(index, columns)
        col = offset if row % 2 == 0 else columns - 1 - offset
        x = 12 + col * step_x
        y = 22 + row * step_y
        points.append((x, y, col, row))

    edges = []
    for i in range(count - 1):
        x1, y1, col1, row1 = points[i]
        x2, y2, col2, row2 = points[i + 1]
        if row1 == row2:
            start_x = x1 + (node_w if col2 > col1 else 0)
            end_x = x2 + (0 if col2 > col1 else node_w)
            edges.append(f'<path d="M{start_x},{y1 + node_h/2} L{end_x},{y2 + node_h/2}"/>')
        else:
            # Turn at the edge of the row, then point down to the next state.
            turn_x = x1 + (node_w if row1 % 2 == 0 else 0)
            edges.append(
                f'<path d="M{turn_x},{y1 + node_h/2} L{turn_x},{y1 + node_h + 9} '
                f'L{x2 + node_w/2},{y2 - 4}"/>'
            )

    nodes = []
    for i, (x, y, _, _) in enumerate(points):
        if i == 0:
            title, action, color = "START", f"({route[i][0]}, {route[i][1]})", "#2459c7"
        else:
            title = f"STEP {i}"
            action = f"{html.escape(actions[i - 1])} · ({route[i][0]}, {route[i][1]})"
            color = "#187c73" if i == count - 1 else "#203b61"
        nodes.append(
            f'<g transform="translate({x},{y})"><rect width="{node_w}" height="{node_h}" rx="11" fill="{color}"/>'
            f'<text x="{node_w/2}" y="18" text-anchor="middle" class="step">{title}</text>'
            f'<text x="{node_w/2}" y="36" text-anchor="middle" class="action">{action}</text></g>'
        )

    svg = (
        f'<svg viewBox="0 0 {width} {22 + rows * step_y}" role="img" '
        f'aria-label="Directed action graph with {count} route states">'
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#7590b2"/></marker></defs>'
        f'<g class="edges">{"".join(edges)}</g>{"".join(nodes)}</svg>'
    )
    page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{{box-sizing:border-box}}body{{margin:0;padding:12px;background:#f3f6fb;font-family:Inter,ui-sans-serif,system-ui,"Segoe UI",sans-serif;color:#1c304d}}
.card{{background:#fff;border:1px solid #dce5f0;border-radius:16px;padding:16px;box-shadow:0 8px 24px #19314e0d}}
.head{{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}}
h3{{font-size:18px;margin:0 0 4px;font-weight:750}}.sub{{font-size:13px;color:#526780}}
.badge{{padding:8px 11px;border-radius:999px;background:#e8efff;color:#204eb1;font-size:12px;font-weight:800;white-space:nowrap}}
.viewport{{overflow:auto;background:#f7f9fc;border:1px solid #e7edf5;border-radius:12px;padding:8px}}
svg{{display:block;width:100%;min-width:900px;height:auto}}.edges path{{fill:none;stroke:#7590b2;stroke-width:2.2;marker-end:url(#arrow)}}
.step{{fill:#e5edfa;font-size:10px;font-weight:800;letter-spacing:.04em}}.action{{fill:#fff;font-size:10px;font-weight:700}}
@media(max-width:600px){{.card{{padding:11px}}.sub{{font-size:12px}}.badge{{font-size:10px}}svg{{min-width:900px}}}}
</style></head><body><div class="card"><div class="head"><div><h3>Algorithm movement graph</h3>
<div class="sub">Every node is a route state; arrows show the planner’s move order and grid coordinates.</div></div>
<div class="badge">{count - 1} MOVES · {count} STATES</div></div><div class="viewport">{svg}</div></div></body></html>'''
    components.html(page, height=height, scrolling=True)
