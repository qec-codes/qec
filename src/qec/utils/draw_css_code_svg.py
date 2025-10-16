"""
Utility functions for drawing CSS code Tanner graphs using raw SVG and vanilla JavaScript tooltips.
"""

from __future__ import annotations

import json
from typing import Iterable, Tuple


def draw_css_code_tanner_graph_svg(
    css_code,
    file_name: str,
    qubit_radius: float = 8,
    check_radius: float = 10,
    qubit_label: str = "Q",
    x_check_label: str = "X",
    z_check_label: str = "Z",
    qubit_label_offset: int = 0,
    check_label_offset: int = 0,
    qubit_color: str = "#1f77b4",
    qubit_fill: str = "white",
    x_check_color: str = "#ff7f0e",
    x_check_fill: str = "white",
    z_check_color: str = "#2ca02c",
    z_check_fill: str = "white",
    x_edge_color: str = "#1f77b4",
    z_edge_color: str = "#d62728",
    x_edge_style: str = "solid",
    z_edge_style: str = "dashed",
    edge_width: float = 2,
    spacing: float = 50,
    width: int | None = None,
    height: int | None = None,
    margin: Tuple[int, int, int, int] | None = None,
    show_labels: bool = True,
    label_fontsize: int = 12,
) -> None:
    """Draw a CSS code Tanner graph into a standalone HTML file using raw SVG.

    Parameters
    ----------
    css_code : object
        Any object exposing the same coordinate/edge API as other utilities
        in this package (``qubit_coordinates``, ``x_check_coordinates``,
        ``z_check_coordinates``, ``x_edge_coordinates``, ``z_edge_coordinates``).
    file_name : str
        Output path for the generated HTML file.
    qubit_radius, check_radius : float
        Node radii for qubits and stabilizers in pixels.
    qubit_label, x_check_label, z_check_label : str
        Prefixes used when generating node labels.
    qubit_label_offset, check_label_offset : int
        Index offsets applied to automatically generated labels.
    qubit_color, x_check_color, z_check_color : str
        Stroke colors for each node class.
    qubit_fill, x_check_fill, z_check_fill : str
        Fill colors for each node class.
    x_edge_color, z_edge_color : str
        Edge colors for X- and Z-type checks.
    x_edge_style, z_edge_style : str
        Edge style indicator ("solid" or "dashed").
    edge_width : float
        Width of drawn edges in pixels.
    spacing : float
        Scaling factor applied to the logical coordinates.
    width, height : int or None
        Explicit SVG dimensions. Ignored when ``margin`` is supplied.
    margin : tuple[int, int, int, int] or None
        Optional margins ``(left, top, right, bottom)`` in pixels. When
        provided the SVG canvas is auto-sized to hug the layout plus margin.
    show_labels : bool
        Whether to display node labels.
    label_fontsize : int
        Font size for rendered labels in pixels.
    """

    if hasattr(css_code, "get_node_coordinates"):
        css_code.get_node_coordinates()
    if hasattr(css_code, "get_x_edge_coordinates"):
        css_code.get_x_edge_coordinates()
    if hasattr(css_code, "get_z_edge_coordinates"):
        css_code.get_z_edge_coordinates()

    qubit_coords: Iterable[Tuple[float, float]] = getattr(css_code, "qubit_coordinates", [])
    x_check_coords: Iterable[Tuple[float, float]] = getattr(css_code, "x_check_coordinates", [])
    z_check_coords: Iterable[Tuple[float, float]] = getattr(css_code, "z_check_coordinates", [])
    x_edge_coords = getattr(css_code, "x_edge_coordinates", [])
    z_edge_coords = getattr(css_code, "z_edge_coordinates", [])

    all_coord_pairs = list(qubit_coords) + list(x_check_coords) + list(z_check_coords)
    all_x = [x for x, _ in all_coord_pairs]
    all_y = [y for _, y in all_coord_pairs]

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
    else:
        min_x = max_x = min_y = max_y = 0.0

    if margin is not None:
        if len(margin) != 4:
            raise ValueError("margin must be a tuple of 4 integers: (left, top, right, bottom)")
        margin_left, margin_top, margin_right, margin_bottom = margin
        max_qubit_radius = qubit_radius if qubit_coords else 0
        max_check_radius = check_radius if (x_check_coords or z_check_coords) else 0
        max_node_radius = max(max_qubit_radius, max_check_radius)
        pad = max_node_radius + edge_width / 2

        min_x_pad = min_x
        max_x_pad = max_x
        min_y_pad = min_y
        max_y_pad = max_y

        graph_width = (max_x_pad - min_x_pad) * spacing if max_x_pad > min_x_pad else spacing
        graph_height = (max_y_pad - min_y_pad) * spacing if max_y_pad > min_y_pad else spacing
        width = int(graph_width + margin_left + margin_right + 2 * pad)
        height = int(graph_height + margin_top + margin_bottom + 2 * pad)

        def transform_coord(x: float, y: float) -> Tuple[float, float]:
            tx = (x - min_x_pad) * spacing + margin_left + pad
            ty = (max_y_pad - y) * spacing + margin_top + pad
            return tx, ty

    else:
        width = 800 if width is None else width
        height = 600 if height is None else height
        center_x = (min_x + max_x) / 2 if all_x else 0.0
        center_y = (min_y + max_y) / 2 if all_y else 0.0

        def transform_coord(x: float, y: float) -> Tuple[float, float]:
            tx = (x - center_x) * spacing + width / 2
            ty = (center_y - y) * spacing + height / 2
            return tx, ty

    nodes_data = []
    edges_data = []

    for idx, (x, y) in enumerate(qubit_coords):
        tx, ty = transform_coord(x, y)
        nodes_data.append(
            {
                "id": f"q{idx}",
                "x": tx,
                "y": ty,
                "type": "qubit",
                "label": f"{qubit_label}_{idx + qubit_label_offset}" if show_labels else "",
                "radius": qubit_radius,
                "color": qubit_color,
                "fill": qubit_fill,
            }
        )

    for idx, (x, y) in enumerate(x_check_coords):
        tx, ty = transform_coord(x, y)
        nodes_data.append(
            {
                "id": f"x{idx}",
                "x": tx,
                "y": ty,
                "type": "x_check",
                "label": f"{x_check_label}_{idx + check_label_offset}" if show_labels else "",
                "radius": check_radius,
                "color": x_check_color,
                "fill": x_check_fill,
            }
        )

    for idx, (x, y) in enumerate(z_check_coords):
        tx, ty = transform_coord(x, y)
        nodes_data.append(
            {
                "id": f"z{idx}",
                "x": tx,
                "y": ty,
                "type": "z_check",
                "label": f"{z_check_label}_{idx + check_label_offset}" if show_labels else "",
                "radius": check_radius,
                "color": z_check_color,
                "fill": z_check_fill,
            }
        )

    for q_coord, c_coord in x_edge_coords:
        qx, qy = transform_coord(q_coord[0], q_coord[1])
        cx, cy = transform_coord(c_coord[0], c_coord[1])
        edges_data.append(
            {
                "x1": qx,
                "y1": qy,
                "x2": cx,
                "y2": cy,
                "color": x_edge_color,
                "style": x_edge_style,
                "width": edge_width,
            }
        )

    for q_coord, c_coord in z_edge_coords:
        qx, qy = transform_coord(q_coord[0], q_coord[1])
        cx, cy = transform_coord(c_coord[0], c_coord[1])
        edges_data.append(
            {
                "x1": qx,
                "y1": qy,
                "x2": cx,
                "y2": cy,
                "color": z_edge_color,
                "style": z_edge_style,
                "width": edge_width,
            }
        )

    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)

    html_template = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>CSS Code Tanner Graph (SVG)</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        #container {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            display: inline-block;
        }}
        .node-label {{
            font-size: {label_fontsize}px;
            pointer-events: none;
            user-select: none;
        }}
        .edge {{
            pointer-events: none;
        }}
        .node {{
            cursor: pointer;
        }}
        .svg-tooltip {{
            position: absolute;
            text-align: left;
            padding: 6px 10px;
            font-size: 13px;
            background: #fff;
            color: #111;
            border: 1.5px solid #111;
            border-radius: 4px;
            pointer-events: none;
            z-index: 1000;
            visibility: hidden;
            opacity: 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: opacity 0.08s ease;
        }}
    </style>
</head>
<body>
    <div id=\"container\">
        <svg id=\"graph\" width=\"{width}\" height=\"{height}\" xmlns=\"http://www.w3.org/2000/svg\"></svg>
    </div>
    <script>
        const nodesData = {nodes_json};
        const edgesData = {edges_json};

        const svgNS = "http://www.w3.org/2000/svg";
        const svg = document.getElementById("graph");

        // Remove stale tooltips before creating a new one
        document.querySelectorAll('.svg-tooltip').forEach(el => el.remove());
        const tooltip = document.body.appendChild(document.createElement('div'));
        tooltip.className = 'svg-tooltip';

        // Edges first to keep them underneath
        edgesData.forEach(edge => {{
            const line = document.createElementNS(svgNS, 'line');
            line.setAttribute('class', 'edge');
            line.setAttribute('x1', edge.x1);
            line.setAttribute('y1', edge.y1);
            line.setAttribute('x2', edge.x2);
            line.setAttribute('y2', edge.y2);
            line.setAttribute('stroke', edge.color);
            line.setAttribute('stroke-width', edge.width);
            if (edge.style === 'dashed') {{
                line.setAttribute('stroke-dasharray', '5,5');
            }}
            svg.appendChild(line);
        }});

        const formatTooltip = node => `<b>${{node.label}}</b><br>Type: ${{node.type}}`;

        const positionTooltip = event => {{
            const tooltipNode = tooltip;
            const pageX = event.pageX !== undefined ? event.pageX : event.clientX + window.scrollX;
            const pageY = event.pageY !== undefined ? event.pageY : event.clientY + window.scrollY;
            const tooltipWidth = tooltipNode.offsetWidth || 0;
            const tooltipHeight = tooltipNode.offsetHeight || 0;

            let x = pageX + 16;
            let y = pageY - tooltipHeight / 2;

            const maxX = window.pageXOffset + window.innerWidth - tooltipWidth - 12;
            const minX = window.pageXOffset + 12;
            const maxY = window.pageYOffset + window.innerHeight - tooltipHeight - 12;
            const minY = window.pageYOffset + 12;

            if (x > maxX) {{ x = maxX; }}
            if (x < minX) {{ x = minX; }}
            if (y > maxY) {{ y = maxY; }}
            if (y < minY) {{ y = minY; }}

            tooltipNode.style.left = `${{x}}px`;
            tooltipNode.style.top = `${{y}}px`;
        }};

        nodesData.forEach(node => {{
            const group = document.createElementNS(svgNS, 'g');
            group.setAttribute('class', 'node');
            group.setAttribute('transform', `translate(${{node.x}},${{node.y}})`);
            svg.appendChild(group);

            if (node.type === 'qubit') {{
                const circle = document.createElementNS(svgNS, 'circle');
                circle.setAttribute('r', node.radius);
                circle.setAttribute('fill', node.fill);
                circle.setAttribute('stroke', node.color);
                circle.setAttribute('stroke-width', 2);
                group.appendChild(circle);
            }} else {{
                const rect = document.createElementNS(svgNS, 'rect');
                rect.setAttribute('x', -node.radius);
                rect.setAttribute('y', -node.radius);
                rect.setAttribute('width', node.radius * 2);
                rect.setAttribute('height', node.radius * 2);
                rect.setAttribute('fill', node.fill);
                rect.setAttribute('stroke', node.color);
                rect.setAttribute('stroke-width', 2);
                group.appendChild(rect);
            }}

            if (node.label) {{
                const text = document.createElementNS(svgNS, 'text');
                text.setAttribute('class', 'node-label');
                text.setAttribute('x', node.radius + 5);
                text.setAttribute('y', 5);
                text.setAttribute('text-anchor', 'start');
                text.textContent = node.label;
                group.appendChild(text);
            }}

            group.addEventListener('mouseenter', event => {{
                tooltip.innerHTML = formatTooltip(node);
                tooltip.style.visibility = 'visible';
                tooltip.style.opacity = 1;
                positionTooltip(event);
            }});
            group.addEventListener('mousemove', event => {{
                positionTooltip(event);
            }});
            group.addEventListener('mouseleave', () => {{
                tooltip.style.opacity = 0;
                tooltip.style.visibility = 'hidden';
            }});
        }});
    </script>
</body>
</html>"""

    html_content = html_template.format(
        label_fontsize=label_fontsize,
        width=width,
        height=height,
        nodes_json=nodes_json,
        edges_json=edges_json,
    )

    with open(file_name, "w", encoding="utf-8") as fh:
        fh.write(html_content)

    print(f"Successfully created SVG visualization: {file_name}")


# Example usage
if __name__ == "__main__":
    from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode

    code = RotatedSurfaceCode(31)
    code.get_node_coordinates()
    code.get_x_edge_coordinates()
    code.get_z_edge_coordinates()

    output_file = "rotated_xy_surface_l31_svg.html"
    draw_css_code_tanner_graph_svg(
        code,
        output_file,
        qubit_radius=16,
        check_radius=20,
        spacing=150,
        width=1600,
        height=1600,
        qubit_label="q",
        x_check_label="S^X",
        z_check_label="S^Y",
        show_labels=True,
        label_fontsize=10,
        x_edge_color="black",
        z_edge_color="black",
        x_check_color="black",
        z_check_color="black",
        edge_width=3,
    )
    print(f"Open {output_file} in a web browser to view the SVG visualization.")
