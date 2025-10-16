"""
Utility to render CSS code Tanner graphs as raw SVG (no D3 dependency).

- Matches the API and layout behavior of draw_css_code_tanner_graph_d3.
- Supports width/height or margin-based autosizing.
- SVG-native tooltips via <title> (works in <img>, <object>, and inline).
"""

from html import escape


def draw_css_code_tanner_graph_svg(
    css_code,
    file_name,
    qubit_radius=8,
    check_radius=10,
    qubit_label="Q",
    x_check_label="X",
    z_check_label="Z",
    qubit_label_offset=0,
    check_label_offset=0,
    qubit_color="#1f77b4",
    qubit_fill="white",
    x_check_color="#ff7f0e",
    x_check_fill="white",
    z_check_color="#2ca02c",
    z_check_fill="white",
    x_edge_color="#1f77b4",
    z_edge_color="#d62728",
    x_edge_style="solid",
    z_edge_style="dashed",
    edge_width=2,
    spacing=50,
    width=None,
    height=None,
    margin=None,
    show_labels=True,
    label_fontsize=12,
):
    """
    Render a CSS code Tanner graph as a standalone SVG file.

    Parameters
    ----------
    css_code : object
        Must expose:
          - qubit_coordinates, x_check_coordinates, z_check_coordinates (list[(x,y)])
          - x_edge_coordinates, z_edge_coordinates (list[((x1,y1),(x2,y2))])
        If methods get_node_coordinates/get_x_edge_coordinates/get_z_edge_coordinates exist,
        they will be called to populate the above.
    file_name : str
        Output .svg file path.
    spacing : float
        Scale factor for all coordinates (SVG px per code unit).
    width, height : int or None
        Fixed SVG size. Ignored if margin is provided.
    margin : (left, top, right, bottom) or None
        If provided, the SVG will be auto-sized to fit the graph plus margins.
        When omitted and width/height not given, defaults to (0,0,0,0).
    Node and edge styling
    ---------------------
    qubit_radius, check_radius : float
    qubit_color, qubit_fill : str
    x_check_color, x_check_fill : str
    z_check_color, z_check_fill : str
    x_edge_color, z_edge_color : str
    x_edge_style, z_edge_style : {'solid','dashed'}
    edge_width : float
    Labels
    ------
    qubit_label, x_check_label, z_check_label : str
    qubit_label_offset, check_label_offset : int
    show_labels : bool
    label_fontsize : int

    Notes
    -----
    - Y axis is “up” (larger logical Y renders higher on the canvas), matching the D3 version.
    - Native <title> tooltips work in <img>. If you need custom interactive popups, load the SVG
      via <object> or inline and attach JS externally.
    """
    # Populate coordinates/edges if helper methods exist
    if hasattr(css_code, "get_node_coordinates"):
        css_code.get_node_coordinates()
    if hasattr(css_code, "get_x_edge_coordinates"):
        css_code.get_x_edge_coordinates()
    if hasattr(css_code, "get_z_edge_coordinates"):
        css_code.get_z_edge_coordinates()

    qubit_coords = getattr(css_code, "qubit_coordinates", []) or []
    x_check_coords = getattr(css_code, "x_check_coordinates", []) or []
    z_check_coords = getattr(css_code, "z_check_coordinates", []) or []
    x_edge_coords = getattr(css_code, "x_edge_coordinates", []) or []
    z_edge_coords = getattr(css_code, "z_edge_coordinates", []) or []

    all_nodes = qubit_coords + x_check_coords + z_check_coords
    if all_nodes:
        xs = [x for x, _ in all_nodes]
        ys = [y for _, y in all_nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x = max_x = min_y = max_y = 0.0

    # Default margin if nothing else specified
    if height is None and width is None and margin is None:
        margin = (0, 0, 0, 0)

    # Compute transform and canvas size
    if margin is not None:
        if len(margin) != 4:
            raise ValueError("margin must be a 4-tuple: (left, top, right, bottom)")
        ml, mt, mr, mb = margin
        max_qubit_r = qubit_radius if qubit_coords else 0
        max_check_r = check_radius if (x_check_coords or z_check_coords) else 0
        max_r = max(max_qubit_r, max_check_r)
        edge_pad = edge_width / 2.0
        pad = max_r + edge_pad

        graph_w = (max_x - min_x) * spacing if max_x > min_x else spacing
        graph_h = (max_y - min_y) * spacing if max_y > min_y else spacing
        width = int(graph_w + ml + mr + 2 * pad)
        height = int(graph_h + mt + mb + 2 * pad)

        def transform(x, y):
            tx = (x - min_x) * spacing + ml + pad
            ty = (max_y - y) * spacing + mt + pad  # Y-up
            return tx, ty
    else:
        width = int(width if width is not None else 800)
        height = int(height if height is not None else 600)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        def transform(x, y):
            tx = (x - cx) * spacing + width / 2.0
            ty = (cy - y) * spacing + height / 2.0  # Y-up
            return tx, ty

    # Build edge and node elements
    def line_element(x1, y1, x2, y2, color, style, w):
        dash = ' stroke-dasharray="5,5"' if style == "dashed" else ""
        return f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{escape(color)}" stroke-width="{w}" stroke-linecap="round"{dash} />'

    def circle_element(cx, cy, r, stroke, fill, title=None, label=None):
        t = f"<title>{escape(title)}</title>" if title else ""
        lbl = ""
        if show_labels and label:
            lbl = f'<text x="{r + 5:.3f}" y="5" font-size="{label_fontsize}" text-anchor="start" fill="#111">{escape(label)}</text>'
        return (
            f'<g transform="translate({cx:.3f},{cy:.3f})">'
            f'{t}<circle r="{r}" fill="{escape(fill)}" stroke="{escape(stroke)}" stroke-width="2" />'
            f"{lbl}</g>"
        )

    def square_element(cx, cy, r, stroke, fill, title=None, label=None):
        t = f"<title>{escape(title)}</title>" if title else ""
        lbl = ""
        if show_labels and label:
            lbl = f'<text x="{r + 5:.3f}" y="5" font-size="{label_fontsize}" text-anchor="start" fill="#111">{escape(label)}</text>'
        return (
            f'<g transform="translate({cx:.3f},{cy:.3f})">'
            f'{t}<rect x="{-r}" y="{-r}" width="{2*r}" height="{2*r}" fill="{escape(fill)}" stroke="{escape(stroke)}" stroke-width="2" />'
            f"{lbl}</g>"
        )

    edge_elems = []
    for (p, q) in x_edge_coords:
        x1, y1 = transform(p[0], p[1])
        x2, y2 = transform(q[0], q[1])
        edge_elems.append(line_element(x1, y1, x2, y2, x_edge_color, x_edge_style, edge_width))
    for (p, q) in z_edge_coords:
        x1, y1 = transform(p[0], p[1])
        x2, y2 = transform(q[0], q[1])
        edge_elems.append(line_element(x1, y1, x2, y2, z_edge_color, z_edge_style, edge_width))

    node_elems = []
    for i, (x, y) in enumerate(qubit_coords):
        tx, ty = transform(x, y)
        lbl = f"{qubit_label}_{i + qubit_label_offset}" if show_labels else ""
        title = lbl if lbl else f"qubit {i}"
        node_elems.append(circle_element(tx, ty, qubit_radius, qubit_color, qubit_fill, title=title, label=lbl))

    for i, (x, y) in enumerate(x_check_coords):
        tx, ty = transform(x, y)
        lbl = f"{x_check_label}_{i + check_label_offset}" if show_labels else ""
        title = lbl if lbl else f"X check {i}"
        node_elems.append(square_element(tx, ty, check_radius, x_check_color, x_check_fill, title=title, label=lbl))

    for i, (x, y) in enumerate(z_check_coords):
        tx, ty = transform(x, y)
        lbl = f"{z_check_label}_{i + check_label_offset}" if show_labels else ""
        title = lbl if lbl else f"Z check {i}"
        node_elems.append(square_element(tx, ty, check_radius, z_check_color, z_check_fill, title=title, label=lbl))

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<desc>CSS code Tanner graph</desc>',
        '<g id="edges">',
        *edge_elems,
        "</g>",
        '<g id="nodes">',
        *node_elems,
        "</g>",
        "</svg>",
    ]
    with open(file_name, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

    print(f"SVG written to: {file_name}")


if __name__ == "__main__":
    # Example usage: visualize a rotated surface code (same example parameters as D3 utility)
    from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode

    code = RotatedSurfaceCode(5)
    code.get_node_coordinates()
    code.get_x_edge_coordinates()
    code.get_z_edge_coordinates()

    out_svg = "rotated_xy_surface_l5.svg"
    draw_css_code_tanner_graph_svg(
        code,
        out_svg,
        qubit_radius=8,
        check_radius=10,
        spacing=75,
        qubit_label="Q",
        x_check_label="SX",
        z_check_label="SZ",
        show_labels=True,
        label_fontsize=10,
        x_edge_color="black",
        x_check_fill="white",
        z_edge_color="black",
        z_check_fill="white",
        qubit_fill="#0091ff",
        x_check_color="black",
        z_check_color="black",
        edge_width=4,
        margin=(20, 20, 20, 20),  # or set width/height instead
    )
    print(f"Open {out_svg} directly in a browser or include via <img src='{out_svg}'>")