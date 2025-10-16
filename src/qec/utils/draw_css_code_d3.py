"""
Utility functions for drawing CSS code Tanner graphs using D3.js.
"""


def draw_css_code_tanner_graph_d3(
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
    label_fontsize=12
):
    """
    Draws a CSS code Tanner graph using D3.js, creating an interactive HTML visualization.

    Parameters
    ----------
    css_code : object
        CSS code object with required attributes:
            - qubit_coordinates: list of (x, y) positions for qubits
            - x_check_coordinates: list of (x, y) positions for X stabilizers
            - z_check_coordinates: list of (x, y) positions for Z stabilizers
            - x_edge_coordinates: list of ((x1, y1), (x2, y2)) edges for X checks
            - z_edge_coordinates: list of ((x1, y1), (x2, y2)) edges for Z checks
    file_name : str
        Output file name for the HTML code.
    qubit_radius, check_radius : float
        Node radii for qubits and checks (in pixels).
    qubit_label, x_check_label, z_check_label : str
        Label prefixes for qubits and checks.
    qubit_label_offset, check_label_offset : int
        Index offset for node labels.
    qubit_color, x_check_color, z_check_color : str
        Node border colors (CSS color strings).
    qubit_fill, x_check_fill, z_check_fill : str
        Node fill colors (CSS color strings).
    x_edge_color, z_edge_color : str
        Edge colors (CSS color strings).
    x_edge_style, z_edge_style : str
        Edge styles ("solid" or "dashed").
    edge_width : float
        Edge line width in pixels.
    spacing : float
        Spacing factor applied to all coordinates.
    width, height : int or None
        SVG canvas dimensions in pixels. Ignored if margin is specified.
    margin : tuple or None
        Optional. Tuple (left, top, right, bottom) specifying margins in pixels. If provided, SVG size is computed to fit the code with these margins.
    show_labels : bool
        Whether to display node labels.
    label_fontsize : int
        Font size for labels in pixels.

    Notes
    -----
    If margin is specified, width and height are ignored and the SVG size is computed to fit the code with the given margins.
    """
    # Ensure coordinates and edges are available
    if hasattr(css_code, 'get_node_coordinates'):
        css_code.get_node_coordinates()
    if hasattr(css_code, 'get_x_edge_coordinates'):
        css_code.get_x_edge_coordinates()
    if hasattr(css_code, 'get_z_edge_coordinates'):
        css_code.get_z_edge_coordinates()

    # Get coordinates
    qubit_coords = getattr(css_code, "qubit_coordinates", [])
    x_check_coords = getattr(css_code, "x_check_coordinates", [])
    z_check_coords = getattr(css_code, "z_check_coordinates", [])
    x_edge_coords = getattr(css_code, "x_edge_coordinates", [])
    z_edge_coords = getattr(css_code, "z_edge_coordinates", [])


    # Calculate bounds for layout
    all_x = [x for x, y in qubit_coords + x_check_coords + z_check_coords]
    all_y = [y for x, y in qubit_coords + x_check_coords + z_check_coords]

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
    else:
        min_x = max_x = min_y = max_y = 0

    # Margin logic
    if margin is not None:
        if len(margin) != 4:
            raise ValueError("margin must be a tuple of 4 integers: (left, top, right, bottom)")
        margin_left, margin_top, margin_right, margin_bottom = margin
        # Find max node radius for each node type
        max_qubit_radius = qubit_radius if qubit_coords else 0
        max_check_radius = check_radius if (x_check_coords or z_check_coords) else 0
        max_node_radius = max(max_qubit_radius, max_check_radius)
        edge_pad = edge_width / 2
        # Expand bounds by node radius and edge width (in SVG units, not code units)
        pad = (max_node_radius + edge_pad)
        min_x_pad = min_x
        max_x_pad = max_x
        min_y_pad = min_y
        max_y_pad = max_y
        graph_width = (max_x_pad - min_x_pad) * spacing if max_x_pad > min_x_pad else spacing
        graph_height = (max_y_pad - min_y_pad) * spacing if max_y_pad > min_y_pad else spacing
        width = int(graph_width + margin_left + margin_right + 2 * pad)
        height = int(graph_height + margin_top + margin_bottom + 2 * pad)
        def transform_coord(x, y):
            tx = (x - min_x_pad) * spacing + margin_left + pad
            ty = (y - min_y_pad) * spacing + margin_top + pad
            return tx, ty
    else:
        # Default: center in width/height
        width = width if width is not None else 800
        height = height if height is not None else 600
        center_x = (min_x + max_x) / 2 if all_x else 0
        center_y = (min_y + max_y) / 2 if all_y else 0
        def transform_coord(x, y):
            tx = (x - center_x) * spacing + width / 2
            ty = (y - center_y) * spacing + height / 2
            return tx, ty

    # Build node and edge data
    nodes_data = []
    edges_data = []

    # Add qubit nodes
    for idx, (x, y) in enumerate(qubit_coords):
        tx, ty = transform_coord(x, y)
        nodes_data.append({
            'id': f'q{idx}',
            'x': tx,
            'y': ty,
            'type': 'qubit',
            'label': f'{qubit_label}_{idx + qubit_label_offset}' if show_labels else '',
            'radius': qubit_radius,
            'color': qubit_color,
            'fill': qubit_fill
        })

    # Add X check nodes
    for idx, (x, y) in enumerate(x_check_coords):
        tx, ty = transform_coord(x, y)
        nodes_data.append({
            'id': f'x{idx}',
            'x': tx,
            'y': ty,
            'type': 'x_check',
            'label': f'{x_check_label}_{idx + check_label_offset}' if show_labels else '',
            'radius': check_radius,
            'color': x_check_color,
            'fill': x_check_fill
        })

    # Add Z check nodes
    for idx, (x, y) in enumerate(z_check_coords):
        tx, ty = transform_coord(x, y)
        nodes_data.append({
            'id': f'z{idx}',
            'x': tx,
            'y': ty,
            'type': 'z_check',
            'label': f'{z_check_label}_{idx + check_label_offset}' if show_labels else '',
            'radius': check_radius,
            'color': z_check_color,
            'fill': z_check_fill
        })

    # Add X edges
    for (q, c) in x_edge_coords:
        qx, qy = transform_coord(q[0], q[1])
        cx, cy = transform_coord(c[0], c[1])
        edges_data.append({
            'x1': qx,
            'y1': qy,
            'x2': cx,
            'y2': cy,
            'color': x_edge_color,
            'style': x_edge_style,
            'width': edge_width
        })

    # Add Z edges
    for (q, c) in z_edge_coords:
        qx, qy = transform_coord(q[0], q[1])
        cx, cy = transform_coord(c[0], c[1])
        edges_data.append({
            'x1': qx,
            'y1': qy,
            'x2': cx,
            'y2': cy,
            'color': z_edge_color,
            'style': z_edge_style,
            'width': edge_width
        })

    # Generate HTML with embedded D3.js
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS Code Tanner Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
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
            font-size: %dpx;
            pointer-events: none;
            user-select: none;
        }}
        .edge {{
            pointer-events: none;
        }}
        .node {{
            cursor: pointer;
        }}
        /* Remove hover opacity effect */
        .d3-tooltip {{
            position: absolute;
            text-align: left;
            padding: 6px 10px;
            font-size: 13px;
            background: rgba(0,0,0,0.85);
            color: #fff;
            border-radius: 4px;
            pointer-events: none;
            z-index: 10;
            visibility: hidden;
        }}
    </style>
</head>
<body>
    <div id="container">
        <svg id="graph" width="%d" height="%d"></svg>
    </div>
    <div class="d3-tooltip" id="d3-tooltip"></div>
    <script>
        const nodesData = %s;
        const edgesData = %s;

        const svg = d3.select("#graph");
        const tooltip = d3.select("#d3-tooltip");

        // Draw edges first (bottom layer)
        const edges = svg.selectAll(".edge")
            .data(edgesData)
            .enter()
            .append("line")
            .attr("class", "edge")
            .attr("x1", d => d.x1)
            .attr("y1", d => d.y1)
            .attr("x2", d => d.x2)
            .attr("y2", d => d.y2)
            .attr("stroke", d => d.color)
            .attr("stroke-width", d => d.width)
            .attr("stroke-dasharray", d => d.style === "dashed" ? "5,5" : "none");

        // Create node groups
        const nodes = svg.selectAll(".node")
            .data(nodesData)
            .enter()
            .append("g")
            .attr("class", "node")
            .attr("transform", d => `translate(${d.x},${d.y})`);

        // Draw node shapes
        nodes.each(function(d) {
            const node = d3.select(this);
            if (d.type === "qubit") {
                // Circle for qubits
                node.append("circle")
                    .attr("r", d.radius)
                    .attr("fill", d.fill)
                    .attr("stroke", d.color)
                    .attr("stroke-width", 2);
            } else {
                // Square for checks
                node.append("rect")
                    .attr("x", -d.radius)
                    .attr("y", -d.radius)
                    .attr("width", d.radius * 2)
                    .attr("height", d.radius * 2)
                    .attr("fill", d.fill)
                    .attr("stroke", d.color)
                    .attr("stroke-width", 2);
            }
        });

        // Add labels
        nodes.append("text")
            .attr("class", "node-label")
            .attr("x", d => d.radius + 5)
            .attr("y", 5)
            .attr("text-anchor", "start")
            .text(d => d.label);

        // Custom tooltip on hover (immediate)
        nodes.on("mouseover", function(event, d) {
            tooltip.style("visibility", "visible")
                .html(`<b>${d.label}</b><br>Type: ${d.type}`)
                .style("left", (event.pageX + 12) + "px")
                .style("top", (event.pageY - 12) + "px");
        })
        .on("mousemove", function(event) {
            tooltip.style("left", (event.pageX + 12) + "px")
                .style("top", (event.pageY - 12) + "px");
        })
        .on("mouseout", function() {
            tooltip.style("visibility", "hidden");
        });
    </script>
</body>
</html>""" % (
        label_fontsize,
        width,
        height,
        repr(nodes_data),
        repr(edges_data)
    )

    # Write to file
    with open(file_name, "w") as f:
        f.write(html_content)
    
    print(f"Successfully created D3.js visualization: {file_name}")


# Example usage: visualize a CSS code using D3.js
if __name__ == "__main__":
    from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
    
    code = RotatedSurfaceCode(5)
    code.get_node_coordinates()
    code.get_x_edge_coordinates()
    code.get_z_edge_coordinates()
    
    output_file = "rotated_xy_surface_l31_d3.html"
    # Example 1: Specify width/height (centered)
    draw_css_code_tanner_graph_d3(
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
        edge_width=3
    )
    print(f"Open {output_file} in a web browser to view the interactive visualization.")

    # Example 2: Specify margin (auto-size SVG)
    output_file2 = "rotated_xy_surface_l31_d3_margin.html"
    draw_css_code_tanner_graph_d3(
        code,
        output_file2,
        qubit_radius=16,
        check_radius=20,
        spacing=150,
        margin=(0, 0, 0, 0),
        qubit_label="q",
        x_check_label="S^X",
        z_check_label="S^Y",
        show_labels=True,
        label_fontsize=10,
        x_edge_color="black",
        z_edge_color="black",
        x_check_color="black",
        z_check_color="black",
        edge_width=3
    )
    print(f"Open {output_file2} in a web browser to view the margin-based interactive visualization.")
