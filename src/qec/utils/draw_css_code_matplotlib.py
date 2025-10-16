import matplotlib.pyplot as plt
def draw_css_code_matplotlib(
    css_code,
    ax=None,
    qubit_color="C0",
    x_check_color="C2",
    z_check_color="C1",
    qubit_size=100,
    check_size=140,
    zorder_qubits=2,
    zorder_checks=1,
    label_qubits=True,
    label_checks=True,
    label_fontsize=8,
    label_color="k",
    label_offset=(0.0, 0.0),
    x_edge_color="#1f77b4",  # blue
    z_edge_color="#d62728",  # red
    edge_alpha=0.7,
    edge_lw=2,
):
    """
    Visualize a CSS code using matplotlib.

    This function draws qubit nodes (circles), X/Z check nodes (squares), and edges on a white background.
    The CSS code object must provide attributes for coordinates and edges, e.g.:
        - qubit_coordinates: list of (x, y) positions for qubits
        - x_check_coordinates: list of (x, y) positions for X stabilizers
        - z_check_coordinates: list of (x, y) positions for Z stabilizers
        - x_edge_coordinates: list of ((x1, y1), (x2, y2)) edges for X checks
        - z_edge_coordinates: list of ((x1, y1), (x2, y2)) edges for Z checks
    Optionally, the code object may have methods to generate these attributes:
        - get_node_coordinates(), get_x_edge_coordinates(), get_z_edge_coordinates()

    Parameters
    ----------
    css_code : object
        CSS code object with required attributes.
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on. If None, creates a new figure and axes.
    qubit_color : str
        Color for qubit nodes.
    x_check_color : str
        Color for X-type check nodes.
    z_check_color : str
        Color for Z-type check nodes.
    qubit_size : int
        Marker size for qubit nodes.
    check_size : int
        Marker size for check nodes.
    zorder_qubits : int
        Z-order for qubit markers.
    zorder_checks : int
        Z-order for check markers.
    label_qubits : bool
        If True, annotate qubit nodes with their indices.
    label_checks : bool
        If True, annotate check nodes with their indices.
    label_fontsize : int
        Font size for node labels.
    label_color : str
        Color for node labels.
    label_offset : tuple[float, float]
        (dx, dy) offset applied to labels, in data coordinates.
    x_edge_color : str
        Color for X (hx) edges.
    z_edge_color : str
        Color for Z (hz) edges.
    edge_alpha : float
        Alpha for edge lines.
    edge_lw : float
        Line width for edges.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    # Ensure coordinates and edges are available
    if hasattr(css_code, 'get_node_coordinates'):
        css_code.get_node_coordinates()
    if hasattr(css_code, 'get_x_edge_coordinates'):
        css_code.get_x_edge_coordinates()
    if hasattr(css_code, 'get_z_edge_coordinates'):
        css_code.get_z_edge_coordinates()

    # Prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
        fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Unpack coordinates
    if getattr(css_code, 'qubit_coordinates', None):
        qx, qy = zip(*css_code.qubit_coordinates)
    else:
        qx, qy = [], []
    if getattr(css_code, 'x_check_coordinates', None):
        xcx, xcy = zip(*css_code.x_check_coordinates)
    else:
        xcx, xcy = [], []
    if getattr(css_code, 'z_check_coordinates', None):
        zcx, zcy = zip(*css_code.z_check_coordinates)
    else:
        zcx, zcy = [], []

    # Draw edges first (bottom layer)
    for (q, c) in getattr(css_code, "x_edge_coordinates", []):
        ax.plot([q[0], c[0]], [q[1], c[1]], color=x_edge_color, alpha=edge_alpha, lw=edge_lw, zorder=0)
    for (q, c) in getattr(css_code, "z_edge_coordinates", []):
        ax.plot([q[0], c[0]], [q[1], c[1]], color=z_edge_color, alpha=edge_alpha, lw=edge_lw, zorder=0)

    # Plot check nodes: X-type (squares, one color), Z-type (squares, another color)
    if xcx and xcy:
        ax.scatter(xcx, xcy, s=check_size, c=x_check_color, marker="s", edgecolors="none", zorder=zorder_checks)
    if zcx and zcy:
        ax.scatter(zcx, zcy, s=check_size, c=z_check_color, marker="s", edgecolors="none", zorder=zorder_checks)
    if qx and qy:
        ax.scatter(qx, qy, s=qubit_size, c=qubit_color, marker="o", edgecolors="none", zorder=zorder_qubits)

    # Number the nodes
    dx, dy = label_offset
    if label_checks:
        # X-type checks
        for idx, (x, y) in enumerate(getattr(css_code, 'x_check_coordinates', [])):
            ax.text(x + dx, y + dy, f"X{idx}", ha="center", va="center", fontsize=label_fontsize, color=label_color, zorder=zorder_checks + 2)
        # Z-type checks
        for idx, (x, y) in enumerate(getattr(css_code, 'z_check_coordinates', [])):
            ax.text(x + dx, y + dy, f"Z{idx}", ha="center", va="center", fontsize=label_fontsize, color=label_color, zorder=zorder_checks + 2)
    if label_qubits and getattr(css_code, 'qubit_coordinates', None):
        for idx, (x, y) in enumerate(css_code.qubit_coordinates):
            ax.text(x + dx, y + dy, str(idx), ha="center", va="center", fontsize=label_fontsize, color=label_color, zorder=zorder_qubits + 2)

    # Formatting: equal aspect, no axes, no grid
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.grid(False)

    # Add a small padding
    all_x = list(qx) + list(xcx) + list(zcx)
    all_y = list(qy) + list(xcy) + list(zcy)
    if all_x and all_y:
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        pad = 0.5
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    if created_fig:
        plt.tight_layout()

    return ax

# Example usage: visualize an L=3 rotated surface code
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
    code = RotatedSurfaceCode(5)
    ax = draw_css_code_matplotlib(code)
    ax.figure.savefig("rotated_surface_L3.png", dpi=200, bbox_inches="tight")
    print("Saved visualization to rotated_surface_L3.png")