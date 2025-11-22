"""
Utility functions for drawing CSS code Tanner graphs using D3.js.

This module provides functions for creating modular, embeddable D3.js visualizations
of CSS code Tanner graphs. It supports both standalone HTML files and embeddable
JavaScript modules that can be integrated into larger applications.
"""

import json
import math


def prepare_css_code_visualization_data(
    css_code,
    qubit_radius=8,
    check_radius=10,
    qubit_label="Q",
    x_check_label="X",
    z_check_label="Z",
    qubit_index_offset=0,
    check_label_offset=0,
    qubit_color="#000000",
    qubit_fill="white",
    x_check_color="#000000",
    x_check_fill="white",
    z_check_color="#000000",
    z_check_fill="white",
    x_edge_color="#000000",
    z_edge_color="#000000",
    x_edge_style="solid",
    z_edge_style="dashed",
    edge_width=2,
    spacing=50,
    width=None,
    height=None,
    margin=None,
    show_labels=True,
    label_fontsize=12,
    qubit_label_position="NE",
    x_check_label_position="NE",
    z_check_label_position="NE",
    qubit_label_xy_offset=None,
    x_check_label_xy_offset=None,
    z_check_label_xy_offset=None,
    background_color="transparent"
):
    """
    Prepare visualization data for a CSS code Tanner graph.

    This function extracts and transforms the CSS code data into a format
    suitable for D3.js rendering. It returns structured data that can be
    used either to generate a standalone HTML file or to embed in a larger
    D3.js visualization.

    Parameters
    ----------
    css_code : object
        CSS code object with required attributes:
            - qubit_coordinates: list of (x, y) positions for qubits
            - x_check_coordinates: list of (x, y) positions for X stabilizers
            - z_check_coordinates: list of (x, y) positions for Z stabilizers
            - x_edge_coordinates: list of ((x1, y1), (x2, y2)) edges for X checks
            - z_edge_coordinates: list of ((x1, y1), (x2, y2)) edges for Z checks
    (other parameters same as draw_css_code_tanner_graph_d3)

    Returns
    -------
    dict
        A dictionary containing:
            - 'nodes': list of node dictionaries with position, style, and label info
            - 'edges': list of edge dictionaries with coordinates and style info
            - 'config': dictionary with SVG dimensions, styling, and layout info
            - 'metadata': additional metadata about the code
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

    label_position_map = {
        "N": (0.0, -1.0, "middle", "alphabetic"),
        "NE": (1.0, -1.0, "start", "alphabetic"),
        "E": (1.0, 0.0, "start", "middle"),
        "SE": (1.0, 1.0, "start", "hanging"),
        "S": (0.0, 1.0, "middle", "hanging"),
        "SW": (-1.0, 1.0, "end", "hanging"),
        "W": (-1.0, 0.0, "end", "middle"),
        "NW": (-1.0, -1.0, "end", "alphabetic"),
        "C": (0.0, 0.0, "middle", "middle")
    }

    def normalise_label_position(position):
        if not position:
            return "NE"
        key = position.strip().upper()
        if key in label_position_map:
            return key
        if key in {"CENTER", "CENTRE"}:
            return "C"
        return "NE"

    def normalise_offset(offset, label_name):
        if offset is None:
            return None
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError(f"{label_name} must be a 2-element tuple like (dx, dy) or None")
        try:
            return float(offset[0]), float(offset[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label_name} must contain numeric values") from exc

    def estimate_text_width(text: str) -> float:
        if not text:
            return 0.0
        # Empirical factor for typical font width
        return max(label_fontsize * 0.6, len(text) * label_fontsize * 0.6)

    def compute_label_info(label_text: str, radius: float, position: str, node_shape: str, custom_offset):
        tooltip_label = label_text
        display_label = label_text if (show_labels and label_text) else ""
        pos_key = normalise_label_position(position)
        mult_x, mult_y, anchor, baseline = label_position_map[pos_key]
        gap = max(6.0, label_fontsize * 0.35)

        if custom_offset is not None:
            dx, dy = custom_offset
        else:
            dir_len = math.sqrt(mult_x * mult_x + mult_y * mult_y)
            if dir_len == 0.0:
                dx = dy = 0.0
            else:
                ux = mult_x / dir_len
                uy = mult_y / dir_len

                if node_shape == "square":
                    distances = []
                    if ux != 0.0:
                        distances.append(radius / abs(ux))
                    if uy != 0.0:
                        distances.append(radius / abs(uy))
                    if not distances:
                        distances.append(radius)
                    distance_to_edge = min(distances)
                else:
                    distance_to_edge = radius

                total_distance = distance_to_edge + gap
                dx = ux * total_distance
                dy = uy * total_distance

        bbox = {
            "left": 0.0,
            "right": 0.0,
            "top": 0.0,
            "bottom": 0.0
        }

        if display_label:
            text_width = estimate_text_width(display_label)
            text_height = float(label_fontsize)

            if anchor == "start":
                x_min = dx
                x_max = dx + text_width
            elif anchor == "end":
                x_min = dx - text_width
                x_max = dx
            else:  # middle
                x_min = dx - text_width / 2
                x_max = dx + text_width / 2

            if baseline == "alphabetic":
                y_min = dy - text_height
                y_max = dy
            elif baseline == "hanging":
                y_min = dy
                y_max = dy + text_height
            else:  # middle
                y_min = dy - text_height / 2
                y_max = dy + text_height / 2

            bbox = {
                "left": x_min,
                "right": x_max,
                "top": y_min,
                "bottom": y_max
            }

        return {
            "display_label": display_label,
            "tooltip_label": tooltip_label,
            "dx": dx,
            "dy": dy,
            "anchor": anchor,
            "baseline": baseline,
            "bbox": bbox
        }

    node_specs = []

    qubit_offset_vec = normalise_offset(qubit_label_xy_offset, "qubit_label_xy_offset")
    x_check_offset_vec = normalise_offset(x_check_label_xy_offset, "x_check_label_xy_offset")
    z_check_offset_vec = normalise_offset(z_check_label_xy_offset, "z_check_label_xy_offset")

    def add_node_spec(node_type, coords, radius, color, fill, label_prefix, label_offset, position, index, node_shape, custom_offset):
        if label_prefix:
            label_text = f"{label_prefix}_{index + label_offset}"
        else:
            label_text = f"{index + label_offset}"
        label_info = compute_label_info(label_text, radius, position, node_shape, custom_offset)
        node_specs.append({
            "type": node_type,
            "coords": coords,
            "radius": radius,
            "color": color,
            "fill": fill,
            "label_info": label_info,
            "id": f"{node_type[0]}{index}",
            "index": index,
        })

    for idx, coord in enumerate(qubit_coords):
        add_node_spec(
            "qubit",
            coord,
            qubit_radius,
            qubit_color,
            qubit_fill,
            qubit_label,
            qubit_index_offset,
            qubit_label_position,
            idx,
            "circle",
            qubit_offset_vec
        )

    for idx, coord in enumerate(x_check_coords):
        add_node_spec(
            "x_check",
            coord,
            check_radius,
            x_check_color,
            x_check_fill,
            x_check_label,
            check_label_offset,
            x_check_label_position,
            idx,
            "square",
            x_check_offset_vec
        )

    for idx, coord in enumerate(z_check_coords):
        add_node_spec(
            "z_check",
            coord,
            check_radius,
            z_check_color,
            z_check_fill,
            z_check_label,
            check_label_offset,
            z_check_label_position,
            idx,
            "square",
            z_check_offset_vec
        )

    # Calculate bounds for layout
    all_x = [x for x, y in qubit_coords + x_check_coords + z_check_coords]
    all_y = [y for x, y in qubit_coords + x_check_coords + z_check_coords]

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
    else:
        min_x = max_x = min_y = max_y = 0

    edge_pad = edge_width / 2.0
    if node_specs:
        pad_left_req = pad_right_req = pad_top_req = pad_bottom_req = 0.0
        for spec in node_specs:
            radius_with_edge = spec["radius"] + edge_pad
            left_extent = right_extent = top_extent = bottom_extent = radius_with_edge
            label_info = spec["label_info"]
            if label_info["display_label"]:
                bbox = label_info["bbox"]
                left_extent = max(left_extent, max(0.0, -bbox["left"]))
                right_extent = max(right_extent, max(0.0, bbox["right"]))
                top_extent = max(top_extent, max(0.0, -bbox["top"]))
                bottom_extent = max(bottom_extent, max(0.0, bbox["bottom"]))
            pad_left_req = max(pad_left_req, left_extent)
            pad_right_req = max(pad_right_req, right_extent)
            pad_top_req = max(pad_top_req, top_extent)
            pad_bottom_req = max(pad_bottom_req, bottom_extent)
    else:
        default_pad = spacing / 2.0
        pad_left_req = pad_right_req = pad_top_req = pad_bottom_req = default_pad

    if height is None and width is None and margin is None:
        margin = (0, 0, 0, 0)

    # Margin logic
    if margin is not None:
        if len(margin) != 4:
            raise ValueError("margin must be a tuple of 4 integers: (left, top, right, bottom)")
        margin_left, margin_top, margin_right, margin_bottom = margin
        pad_left = pad_left_req or spacing / 2.0
        pad_right = pad_right_req or spacing / 2.0
        pad_top = pad_top_req or spacing / 2.0
        pad_bottom = pad_bottom_req or spacing / 2.0

        min_x_pad = min_x
        max_x_pad = max_x
        min_y_pad = min_y
        max_y_pad = max_y

        graph_width = (max_x_pad - min_x_pad) * spacing if max_x_pad > min_x_pad else spacing
        graph_height = (max_y_pad - min_y_pad) * spacing if max_y_pad > min_y_pad else spacing

        width = int(math.ceil(graph_width + margin_left + margin_right + pad_left + pad_right))
        height = int(math.ceil(graph_height + margin_top + margin_bottom + pad_top + pad_bottom))

        def transform_coord(x, y):
            tx = (x - min_x_pad) * spacing + margin_left + pad_left
            ty = (max_y_pad - y) * spacing + margin_top + pad_top
            return tx, ty
    else:
        # Default: center in width/height
        width = width if width is not None else 800
        height = height if height is not None else 600
        center_x = (min_x + max_x) / 2 if all_x else 0
        center_y = (min_y + max_y) / 2 if all_y else 0

        def transform_coord(x, y):
            tx = (x - center_x) * spacing + width / 2
            ty = (center_y - y) * spacing + height / 2
            return tx, ty

    # Build node and edge data
    nodes_data = []
    edges_data = []

    type_label_map = {
        "qubit": "Qubit",
        "x_check": "X stabilizer",
        "z_check": "Z stabilizer"
    }

    for spec in node_specs:
        tx, ty = transform_coord(spec["coords"][0], spec["coords"][1])
        label_info = spec["label_info"]
        nodes_data.append({
            'id': spec['id'],
            'x': tx,
            'y': ty,
            'type': spec['type'],
            'type_label': type_label_map.get(spec['type'], spec['type'].title()),
            'tooltip_label': label_info['tooltip_label'],
            'display_label': label_info['display_label'],
            'label_dx': label_info['dx'],
            'label_dy': label_info['dy'],
            'label_anchor': label_info['anchor'],
            'label_baseline': label_info['baseline'],
            'radius': spec['radius'],
            'color': spec['color'],
            'fill': spec['fill']
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
            'width': edge_width,
            'type': 'x_check'
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
            'width': edge_width,
            'type': 'z_check'
        })

    return {
        'nodes': nodes_data,
        'edges': edges_data,
        'config': {
            'width': width,
            'height': height,
            'label_fontsize': label_fontsize,
            'background_color': background_color
        },
        'metadata': {
            'num_qubits': len(qubit_coords),
            'num_x_checks': len(x_check_coords),
            'num_z_checks': len(z_check_coords),
            'num_x_edges': len(x_edge_coords),
            'num_z_edges': len(z_edge_coords)
        }
    }


def draw_css_code_tanner_graph_d3(
    css_code,
    file_name,
    qubit_radius=8,
    check_radius=10,
    qubit_label="Q",
    x_check_label="X",
    z_check_label="Z",
    qubit_index_offset=0,
    check_label_offset=0,
    qubit_color="#000000",
    qubit_fill="white",
    x_check_color="#000000",
    x_check_fill="white",
    z_check_color="#000000",
    z_check_fill="white",
    x_edge_color="#000000",
    z_edge_color="#000000",
    x_edge_style="solid",
    z_edge_style="dashed",
    edge_width=2,
    spacing=50,
    width=None,
    height=None,
    margin=None,
    show_labels=True,
    label_fontsize=12,
    qubit_label_position="NE",
    x_check_label_position="NE",
    z_check_label_position="NE",
    qubit_label_xy_offset=None,
    x_check_label_xy_offset=None,
    z_check_label_xy_offset=None,
    background_color="transparent"
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
    qubit_index_offset, check_label_offset : int
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
        Whether to draw text labels next to the nodes (tooltips always keep the
        identifiers, even when labels are hidden on the canvas).
    label_fontsize : int
        Font size for labels in pixels.
    qubit_label_position, x_check_label_position, z_check_label_position : str
        Cardinal direction for label placement relative to the node (one of
        "N", "NE", "E", "SE", "S", "SW", "W", "NW", or "C"). Default is "NE".
    qubit_label_xy_offset, x_check_label_xy_offset, z_check_label_xy_offset : tuple(float, float) or None
        Optional explicit (dx, dy) offsets in pixels from the node centre. When
        supplied, these replace the automatic offset calculated from the node
        size and label position for the respective node family.
    background_color : str
        CSS color string for the page and container background. Defaults to
        transparent so the generated page can be overlaid easily.

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

    label_position_map = {
        "N": (0.0, -1.0, "middle", "alphabetic"),
        "NE": (1.0, -1.0, "start", "alphabetic"),
        "E": (1.0, 0.0, "start", "middle"),
        "SE": (1.0, 1.0, "start", "hanging"),
        "S": (0.0, 1.0, "middle", "hanging"),
        "SW": (-1.0, 1.0, "end", "hanging"),
        "W": (-1.0, 0.0, "end", "middle"),
        "NW": (-1.0, -1.0, "end", "alphabetic"),
        "C": (0.0, 0.0, "middle", "middle")
    }

    def normalise_label_position(position):
        if not position:
            return "NE"
        key = position.strip().upper()
        if key in label_position_map:
            return key
        if key in {"CENTER", "CENTRE"}:
            return "C"
        return "NE"

    def normalise_offset(offset, label_name):
        if offset is None:
            return None
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError(f"{label_name} must be a 2-element tuple like (dx, dy) or None")
        try:
            return float(offset[0]), float(offset[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label_name} must contain numeric values") from exc

    def estimate_text_width(text: str) -> float:
        if not text:
            return 0.0
        # Empirical factor for typical font width
        return max(label_fontsize * 0.6, len(text) * label_fontsize * 0.6)

    def compute_label_info(label_text: str, radius: float, position: str, node_shape: str, custom_offset):
        tooltip_label = label_text
        display_label = label_text if (show_labels and label_text) else ""
        pos_key = normalise_label_position(position)
        mult_x, mult_y, anchor, baseline = label_position_map[pos_key]
        gap = max(6.0, label_fontsize * 0.35)

        if custom_offset is not None:
            dx, dy = custom_offset
        else:
            dir_len = math.sqrt(mult_x * mult_x + mult_y * mult_y)
            if dir_len == 0.0:
                dx = dy = 0.0
            else:
                ux = mult_x / dir_len
                uy = mult_y / dir_len

                if node_shape == "square":
                    distances = []
                    if ux != 0.0:
                        distances.append(radius / abs(ux))
                    if uy != 0.0:
                        distances.append(radius / abs(uy))
                    if not distances:
                        distances.append(radius)
                    distance_to_edge = min(distances)
                else:
                    distance_to_edge = radius

                total_distance = distance_to_edge + gap
                dx = ux * total_distance
                dy = uy * total_distance

        bbox = {
            "left": 0.0,
            "right": 0.0,
            "top": 0.0,
            "bottom": 0.0
        }

        if display_label:
            text_width = estimate_text_width(display_label)
            text_height = float(label_fontsize)

            if anchor == "start":
                x_min = dx
                x_max = dx + text_width
            elif anchor == "end":
                x_min = dx - text_width
                x_max = dx
            else:  # middle
                x_min = dx - text_width / 2
                x_max = dx + text_width / 2

            if baseline == "alphabetic":
                y_min = dy - text_height
                y_max = dy
            elif baseline == "hanging":
                y_min = dy
                y_max = dy + text_height
            else:  # middle
                y_min = dy - text_height / 2
                y_max = dy + text_height / 2

            bbox = {
                "left": x_min,
                "right": x_max,
                "top": y_min,
                "bottom": y_max
            }

        return {
            "display_label": display_label,
            "tooltip_label": tooltip_label,
            "dx": dx,
            "dy": dy,
            "anchor": anchor,
            "baseline": baseline,
            "bbox": bbox
        }

    node_specs = []

    qubit_offset_vec = normalise_offset(qubit_label_xy_offset, "qubit_label_xy_offset")
    x_check_offset_vec = normalise_offset(x_check_label_xy_offset, "x_check_label_xy_offset")
    z_check_offset_vec = normalise_offset(z_check_label_xy_offset, "z_check_label_xy_offset")

    def add_node_spec(node_type, coords, radius, color, fill, label_prefix, label_offset, position, index, node_shape, custom_offset):
        if label_prefix:
            label_text = f"{label_prefix}_{index + label_offset}"
        else:
            label_text = f"{index + label_offset}"
        label_info = compute_label_info(label_text, radius, position, node_shape, custom_offset)
        node_specs.append({
            "type": node_type,
            "coords": coords,
            "radius": radius,
            "color": color,
            "fill": fill,
            "label_info": label_info,
            "id": f"{node_type[0]}{index}",
            "index": index,
        })

    for idx, coord in enumerate(qubit_coords):
        add_node_spec(
            "qubit",
            coord,
            qubit_radius,
            qubit_color,
            qubit_fill,
            qubit_label,
            qubit_index_offset,
            qubit_label_position,
            idx,
            "circle",
            qubit_offset_vec
        )

    for idx, coord in enumerate(x_check_coords):
        add_node_spec(
            "x_check",
            coord,
            check_radius,
            x_check_color,
            x_check_fill,
            x_check_label,
            check_label_offset,
            x_check_label_position,
            idx,
            "square",
            x_check_offset_vec
        )

    for idx, coord in enumerate(z_check_coords):
        add_node_spec(
            "z_check",
            coord,
            check_radius,
            z_check_color,
            z_check_fill,
            z_check_label,
            check_label_offset,
            z_check_label_position,
            idx,
            "square",
            z_check_offset_vec
        )

    # Calculate bounds for layout
    all_x = [x for x, y in qubit_coords + x_check_coords + z_check_coords]
    all_y = [y for x, y in qubit_coords + x_check_coords + z_check_coords]

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
    else:
        min_x = max_x = min_y = max_y = 0

    edge_pad = edge_width / 2.0
    if node_specs:
        pad_left_req = pad_right_req = pad_top_req = pad_bottom_req = 0.0
        for spec in node_specs:
            radius_with_edge = spec["radius"] + edge_pad
            left_extent = right_extent = top_extent = bottom_extent = radius_with_edge
            label_info = spec["label_info"]
            if label_info["display_label"]:
                bbox = label_info["bbox"]
                left_extent = max(left_extent, max(0.0, -bbox["left"]))
                right_extent = max(right_extent, max(0.0, bbox["right"]))
                top_extent = max(top_extent, max(0.0, -bbox["top"]))
                bottom_extent = max(bottom_extent, max(0.0, bbox["bottom"]))
            pad_left_req = max(pad_left_req, left_extent)
            pad_right_req = max(pad_right_req, right_extent)
            pad_top_req = max(pad_top_req, top_extent)
            pad_bottom_req = max(pad_bottom_req, bottom_extent)
    else:
        default_pad = spacing / 2.0
        pad_left_req = pad_right_req = pad_top_req = pad_bottom_req = default_pad

    if height is None and width is None and margin is None:
        margin = (0,0,0,0)

    # Margin logic
    if margin is not None:
        if len(margin) != 4:
            raise ValueError("margin must be a tuple of 4 integers: (left, top, right, bottom)")
        margin_left, margin_top, margin_right, margin_bottom = margin
        pad_left = pad_left_req or spacing / 2.0
        pad_right = pad_right_req or spacing / 2.0
        pad_top = pad_top_req or spacing / 2.0
        pad_bottom = pad_bottom_req or spacing / 2.0

        min_x_pad = min_x
        max_x_pad = max_x
        min_y_pad = min_y
        max_y_pad = max_y

        graph_width = (max_x_pad - min_x_pad) * spacing if max_x_pad > min_x_pad else spacing
        graph_height = (max_y_pad - min_y_pad) * spacing if max_y_pad > min_y_pad else spacing

        width = int(math.ceil(graph_width + margin_left + margin_right + pad_left + pad_right))
        height = int(math.ceil(graph_height + margin_top + margin_bottom + pad_top + pad_bottom))

        def transform_coord(x, y):
            tx = (x - min_x_pad) * spacing + margin_left + pad_left
            ty = (max_y_pad - y) * spacing + margin_top + pad_top
            return tx, ty
    else:
        # Default: center in width/height
        width = width if width is not None else 800
        height = height if height is not None else 600
        center_x = (min_x + max_x) / 2 if all_x else 0
        center_y = (min_y + max_y) / 2 if all_y else 0
        def transform_coord(x, y):
            tx = (x - center_x) * spacing + width / 2
            ty = (center_y - y) * spacing + height / 2
            return tx, ty

    # Build node and edge data
    nodes_data = []
    edges_data = []

    type_label_map = {
        "qubit": "Qubit",
        "x_check": "X stabilizer",
        "z_check": "Z stabilizer"
    }

    for spec in node_specs:
        tx, ty = transform_coord(spec["coords"][0], spec["coords"][1])
        label_info = spec["label_info"]
        nodes_data.append({
            'id': spec['id'],
            'x': tx,
            'y': ty,
            'type': spec['type'],
            'type_label': type_label_map.get(spec['type'], spec['type'].title()),
            'tooltip_label': label_info['tooltip_label'],
            'display_label': label_info['display_label'],
            'label_dx': label_info['dx'],
            'label_dy': label_info['dy'],
            'label_anchor': label_info['anchor'],
            'label_baseline': label_info['baseline'],
            'radius': spec['radius'],
            'color': spec['color'],
            'fill': spec['fill']
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

    nodes_json = json.dumps(nodes_data, ensure_ascii=False)
    edges_json = json.dumps(edges_data, ensure_ascii=False)

    # Generate HTML with embedded D3.js
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS Code Tanner Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: %s;
        }
        #container {
            background-color: %s;
            border: none;
            padding: 0;
            display: inline-block;
        }
        .node-label {
            font-size: %dpx;
            pointer-events: none;
            user-select: none;
        }
        .edge {
            pointer-events: none;
        }
        .node {
            cursor: pointer;
        }
        .d3-tooltip {
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
        }
    </style>
</head>
<body>
    <div id="container">
        <svg id="graph" width="%d" height="%d"></svg>
    </div>
    <script>
    const nodesData = %s;
    const edgesData = %s;

        const svg = d3.select("#graph");
        d3.selectAll(".d3-tooltip").remove();
        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "d3-tooltip");

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
            .attr("x", d => d.label_dx)
            .attr("y", d => d.label_dy)
            .attr("text-anchor", d => d.label_anchor)
            .attr("dominant-baseline", d => d.label_baseline)
            .text(d => d.display_label)
            .style("visibility", d => d.display_label ? "visible" : "hidden");

        // Tooltip helpers
        function positionTooltip(event) {
            const tooltipNode = tooltip.node();
            if (!tooltipNode) {
                return;
            }
            const pageX = (event.pageX !== undefined ? event.pageX : event.clientX + window.scrollX);
            const pageY = (event.pageY !== undefined ? event.pageY : event.clientY + window.scrollY);
            const tooltipWidth = tooltipNode.offsetWidth || 0;
            const tooltipHeight = tooltipNode.offsetHeight || 0;

            let x = pageX + 16;
            let y = pageY - tooltipHeight / 2;

            const maxX = window.pageXOffset + window.innerWidth - tooltipWidth - 12;
            const minX = window.pageXOffset + 12;
            const maxY = window.pageYOffset + window.innerHeight - tooltipHeight - 12;
            const minY = window.pageYOffset + 12;

            if (x > maxX) {
                x = maxX;
            }
            if (x < minX) {
                x = minX;
            }
            if (y > maxY) {
                y = maxY;
            }
            if (y < minY) {
                y = minY;
            }

            tooltip.style("left", x + "px")
                   .style("top", y + "px");
        }

        nodes.on("mouseover", function(event, d) {
            const tooltipLabel = d.tooltip_label || d.display_label || d.id;
            tooltip.html("<b>" + tooltipLabel + "</b><br>Type: " + d.type_label)
                .style("visibility", "visible")
                .style("opacity", 1);
            positionTooltip(event);
        })
        .on("mousemove", function(event) {
            positionTooltip(event);
        })
        .on("mouseout", function() {
            tooltip.style("opacity", 0)
                .style("visibility", "hidden");
        });
    </script>
</body>
</html>""" % (
    background_color,
    background_color,
    label_fontsize,
    width,
    height,
    nodes_json,
    edges_json
    )

    # Write to file
    with open(file_name, "w") as f:
        f.write(html_content)
    
    print(f"Successfully created D3.js visualization: {file_name}")


# Example usage: visualize a CSS code using D3.js
if __name__ == "__main__":
    from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
    
    code = RotatedSurfaceCode(35)
    code.get_node_coordinates()
    code.get_x_edge_coordinates()
    code.get_z_edge_coordinates()
    
    output_file = "rs.html"
    # Example 1: Specify width/height (centered)
    draw_css_code_tanner_graph_d3(
        code,
        output_file,
        qubit_radius=12,
        check_radius=15,
        spacing=90,
        qubit_label="Q",
        x_check_label="SX",
        z_check_label="SZ",
        show_labels=False,
        label_fontsize=10,
        x_edge_color="black",
        x_check_fill="white",
        z_edge_color="black",
        z_check_fill="white",
        qubit_fill="#0091ff",
        x_check_color="black",
        z_check_color="black",
    edge_width=4,
        qubit_label_position="NE",
        x_check_label_position="E",
    z_check_label_position="W",
    background_color="#ffffff"
    )
    print(f"Open {output_file} in a web browser to view the interactive visualization.")



