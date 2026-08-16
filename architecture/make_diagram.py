"""Generate architecture/graph.svg from the COMPILED LangGraph.

The node sequence is read from the compiled graph's edges (never hand-listed),
so the diagram cannot drift from the pipeline. Styling is ours; topology is code.

    uv run python architecture/make_diagram.py
"""
from pathlib import Path

from pipeline.graph import build_graph

AGENT_NODES = {"research", "estimate", "reconcile"}   # model-driven stages
NOTES = {
    "research": "cached tools",
    "estimate": "blind — no consensus",
    "reconcile": "config-gated",
    "finalize": "gates + fallback ladder",
}

# geometry
W, H = 800, 208
NODE_W, NODE_H, GAP_Y = 148, 44, 96
ROW1_Y, ROW2_Y = 28, 28 + GAP_Y


def node_order() -> list[str]:
    """Follow the compiled graph's edges from START to END — the single chain."""
    g = build_graph().get_graph()
    nxt = {e.source: e.target for e in g.edges}
    order, cur = [], "__start__"
    while cur in nxt:
        cur = nxt[cur]
        if cur != "__end__":
            order.append(cur)
    return order


def render(order: list[str]) -> str:
    row1, row2 = order[:4], order[4:]
    xs1 = [18 + i * (NODE_W + 62) for i in range(len(row1))]
    xs2 = [18 + i * (NODE_W + 62) for i in range(len(row2))][::-1]  # snake back

    def node(x, y, name):
        agent = name in AGENT_NODES
        fill = "#1f3a5f" if agent else "#ffffff"
        text = "#ffffff" if agent else "#1f3a5f"
        parts = [
            f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="8" '
            f'fill="{fill}" stroke="#1f3a5f" stroke-width="1.6"/>',
            f'<text x="{x + NODE_W / 2}" y="{y + 20}" text-anchor="middle" '
            f'font-family="Georgia,serif" font-size="15" fill="{text}">{name}</text>',
        ]
        if agent:
            parts.append(
                f'<text x="{x + NODE_W / 2}" y="{y + 35}" text-anchor="middle" '
                f'font-family="ui-monospace,monospace" font-size="9.5" '
                f'fill="#b8c4d8">agent</text>')
        if name in NOTES:
            parts.append(
                f'<text x="{x + NODE_W / 2}" y="{y + NODE_H + 14}" text-anchor="middle" '
                f'font-family="ui-monospace,monospace" font-size="9.5" '
                f'fill="#5a6472">{NOTES[name]}</text>')
        return "".join(parts)

    def h_arrow(x1, x2, y):
        return (f'<line x1="{x1}" y1="{y}" x2="{x2 - 7}" y2="{y}" '
                f'stroke="#5a6472" stroke-width="1.4" marker-end="url(#arr)"/>')

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'role="img" aria-label="Pipeline diagram">',
        '<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" '
        'markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M0 0 L8 4 L0 8 z" fill="#5a6472"/></marker></defs>',
    ]
    for x, n in zip(xs1, row1):
        svg.append(node(x, ROW1_Y, n))
    for x, n in zip(xs2, row2):
        svg.append(node(x, ROW2_Y, n))
    for i in range(len(row1) - 1):
        svg.append(h_arrow(xs1[i] + NODE_W, xs1[i + 1], ROW1_Y + NODE_H / 2))
    # wrap: last of row1 down to first of row2 (right edge)
    xw = xs1[-1] + NODE_W / 2
    svg.append(f'<line x1="{xw}" y1="{ROW1_Y + NODE_H}" x2="{xw}" y2="{ROW2_Y - 7}" '
               f'stroke="#5a6472" stroke-width="1.4" marker-end="url(#arr)"/>')
    for i in range(len(row2) - 1):  # row 2 flows right-to-left
        x1 = xs2[i]                    # left edge of current node
        x2 = xs2[i + 1] + NODE_W + 7   # right edge of next node
        svg.append(f'<line x1="{x1}" y1="{ROW2_Y + NODE_H / 2}" x2="{x2}" '
                   f'y2="{ROW2_Y + NODE_H / 2}" stroke="#5a6472" stroke-width="1.4" '
                   f'marker-end="url(#arr)"/>')
    svg.append("</svg>")
    return "".join(svg)


if __name__ == "__main__":
    order = node_order()
    out = Path(__file__).parent / "graph.svg"
    out.write_text(render(order))
    print("nodes from compiled graph:", " -> ".join(order))
    print("wrote", out)
