import numpy as np
import subprocess

def rep_code(num_reps, standard_form=True):
    """ Outputs repetition code parity check matrix given number of repetitions (code distance) """

    pc_matrix = np.zeros((num_reps - 1, num_reps), dtype=int)

    if standard_form:

        for i in range(num_reps - 1):
            pc_matrix[i, i] = 1
            pc_matrix[i, num_reps - 1] = 1

    else:

        for i in range(num_reps - 1):
            pc_matrix[i, i] = 1
            pc_matrix[i, i + 1] = 1

    return pc_matrix


def data_node(x, y, r, label_index=-1, d_label="D", label_position="above right", colour="black", fill_colour="white", value="X", data_label_offset=0):
    """
    Generates a LaTeX TikZ command for a data qubit node.

    Args:
        x, y (float): Coordinates of the node.
        r (float): Radius of the node.
        label_index (int): Index for the label. If -1, no label is added.
        d_label (str): Prefix for the label.
        label_position (str): Position of the label relative to the node.
        colour (str): Border colour of the node.
        fill_colour (str): Fill colour of the node.
        value (str): Value to display inside the node.
        data_label_offset (int): Offset to apply to the data label index.

    Returns:
        str: LaTeX TikZ command for the data node.
    """
    uid = f"{d_label}{np.random.randint(100000)}"

    if label_index == -1:
        label = ""
    else:
        label = f"$\\scriptstyle {d_label}_{{{label_index + data_label_offset}}} $"

    value_label = f"${value}$" if value != -1 else ""

    return f'\\filldraw[fill={fill_colour}, draw={colour}] ({x},{y}) circle ({r}) node [label={label_position}:{{{label}}}]({uid}){{{value_label}}};'


def parity_node(x, y, r, label_index=-1, p_label="A", colour="black", fill_colour="white", label_shift="-0.3", value=-1, parity_label_offset=0):
    """
    Generates a LaTeX TikZ command for a parity qubit node.

    Args:
        x, y (float): Coordinates of the node.
        r (float): Radius of the node.
        label_index (int): Index for the label. If -1, no label is added.
        p_label (str): Prefix for the label.
        colour (str): Border colour of the node.
        fill_colour (str): Fill colour of the node.
        label_shift (str): Distance of the label from the node.
        value (int): Value to display inside the node.
        parity_label_offset (int): Offset to apply to the parity label index.

    Returns:
        str: LaTeX TikZ command for the parity node.
    """
    r = 1.3 * r
    x1, y1 = (x - r / 2, y - r / 2)
    x2, y2 = (x + r / 2, y + r / 2)

    uid = f"{p_label}{np.random.randint(100000)}"

    if label_index == -1:
        label = ""
    else:
        label = f"$\\scriptstyle {p_label}_{{{label_index + parity_label_offset}}}$"

    value_label = f"${int(value)}$" if value != -1 else ""

    return f'\\filldraw[fill={fill_colour}, draw={colour}] ({x1},{y1}) rectangle ({x2},{y2}) node [label={{[label distance={label_shift}cm]30:{{{label}}}}}]({uid}){{}} node at ({x},{y}) {{{value_label}}};'


def draw_line(c1, c2, colour="black", line_width="thick", line_pattern="solid", out_bend=0, in_bend=0):
    x1, y1 = c1
    x2, y2 = c2

    if in_bend == 0 and out_bend == 0:
        return f"\\draw[{colour},{line_width},{line_pattern}] ({x1},{y1}) -- ({x2},{y2});"
    
    return f"\\draw[{colour},{line_width},{line_pattern}] ({x1},{y1}) to[relative,out={out_bend},in={in_bend}] ({x2},{y2});"


def draw_line_dashed(c1, c2, colour):
    x1, y1 = c1
    x2, y2 = c2
    return f"\\draw[{colour},thick,dashed] ({x1},{y1}) -- ({x2},{y2});"



def draw_css_code_tanner_graph_tikz(
    css_code,
    file_name,
    qubit_radius=0.3,
    check_radius=0.39,
    qubit_label="Q",
    x_check_label="X",
    z_check_label="Z",
    qubit_index_offset=0,
    check_label_offset=0,
    qubit_colour="black",
    qubit_fill="white",
    check_colour="black",
    check_fill="white",
    x_edge_style="black,thick,solid",
    z_edge_style="black,thick,dashed",
    spacing=1.0,
    qubit_label_position="above right",
    x_check_label_position="above right",
    z_check_label_position="above right",
    qubit_label_xy_offset=None,
    x_check_label_xy_offset=None,
    z_check_label_xy_offset=None
):
    """
    Draws a CSS code using TikZ, similar to draw_css_code_matplotlib.

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
        Output file name for the LaTeX code.
    qubit_radius, check_radius : float
        Node radii for qubits and checks.
    qubit_label, x_check_label, z_check_label : str
        Label prefixes for qubits and checks.
    qubit_index_offset, check_label_offset : int
        Index offset for node labels.
    qubit_colour, qubit_fill, check_colour, check_fill : str
        Node border and fill colors.
    x_edge_style, z_edge_style : str
        TikZ style for X/Z edges.
    spacing : float
        Spacing factor applied to all coordinates.
    qubit_label_position, x_check_label_position, z_check_label_position : str
        Position of the label relative to the node (e.g. "above right", "right", "below", "south east", etc.).
        These are passed directly to TikZ's node positioning. Default is "above right" (northeast).
        Single letter compass directions (N, S, E, W, NE, NW, SE, SW) are also supported.
    qubit_label_xy_offset, x_check_label_xy_offset, z_check_label_xy_offset : tuple(float, float)
        Additional (x, y) offset applied to the label position for each node type, in TikZ coordinates.
    """
    # Convert single letter compass directions to TikZ positions
    position_map = {
        "N": "above",
        "S": "below",
        "E": "right",
        "W": "left",
        "NE": "above right",
        "NW": "above left",
        "SE": "below right",
        "SW": "below left"
    }
    
    qubit_label_position = position_map.get(qubit_label_position, qubit_label_position)
    x_check_label_position = position_map.get(x_check_label_position, x_check_label_position)
    z_check_label_position = position_map.get(z_check_label_position, z_check_label_position)
    
    f = open(file_name, "w+")
    print(r"\documentclass[tikz, border=0]{standalone}", file=f)
    print(r"\usepackage{tikz}", file=f)
    print(r"\begin{document}", file=f)
    print(r"\begin{tikzpicture}", file=f)


    # Draw X edges
    for (q, c) in getattr(css_code, "x_edge_coordinates", []):
        qx, qy = q[0] * spacing, q[1] * spacing
        cx, cy = c[0] * spacing, c[1] * spacing
        print(f"\\draw[{x_edge_style}] ({qx},{qy}) -- ({cx},{cy});", file=f)
    # Draw Z edges
    for (q, c) in getattr(css_code, "z_edge_coordinates", []):
        qx, qy = q[0] * spacing, q[1] * spacing
        cx, cy = c[0] * spacing, c[1] * spacing
        print(f"\\draw[{z_edge_style}] ({qx},{qy}) -- ({cx},{cy});", file=f)

    # Set default offsets if not provided
    if qubit_label_xy_offset is None:
        qubit_label_xy_offset = (qubit_radius * 0.7, 0)
    if x_check_label_xy_offset is None:
        x_check_label_xy_offset = (check_radius * 0.55, 0)
    if z_check_label_xy_offset is None:
        z_check_label_xy_offset = (check_radius * 0.55, 0)

    # Draw qubit nodes
    for idx, (x, y) in enumerate(getattr(css_code, "qubit_coordinates", [])):
        x, y = x * spacing, y * spacing
        label = f"$\\scriptstyle {qubit_label}_{{{idx + qubit_index_offset}}}$"
        lx, ly = qubit_label_xy_offset
        print(f"\\filldraw[fill={qubit_fill}, draw={qubit_colour}] ({x},{y}) circle ({qubit_radius}) node at ({x+lx},{y+ly}) [{qubit_label_position}] {{{label}}};", file=f)

    # Draw X check nodes
    for idx, (x, y) in enumerate(getattr(css_code, "x_check_coordinates", [])):
        x, y = x * spacing, y * spacing
        label = f"$\\scriptstyle {x_check_label}_{{{idx + check_label_offset}}}$"
        x1, y1 = x - check_radius / 2, y - check_radius / 2
        x2, y2 = x + check_radius / 2, y + check_radius / 2
        lx, ly = x_check_label_xy_offset
        print(f"\\filldraw[fill={check_fill}, draw={check_colour}] ({x1},{y1}) rectangle ({x2},{y2}) node at ({x+lx},{y+ly}) [{x_check_label_position}] {{{label}}};", file=f)

    # Draw Z check nodes
    for idx, (x, y) in enumerate(getattr(css_code, "z_check_coordinates", [])):
        x, y = x * spacing, y * spacing
        label = f"$\\scriptstyle {z_check_label}_{{{idx + check_label_offset}}}$"
        x1, y1 = x - check_radius / 2, y - check_radius / 2
        x2, y2 = x + check_radius / 2, y + check_radius / 2
        lx, ly = z_check_label_xy_offset
        print(f"\\filldraw[fill={check_fill}, draw={check_colour}] ({x1},{y1}) rectangle ({x2},{y2}) node at ({x+lx},{y+ly}) [{z_check_label_position}] {{{label}}};", file=f)

    print(r"\end{tikzpicture}", file=f)
    print(r"\end{document}", file=f)
    f.close()


def compile_latex(file_name):
    """
    Compile the LaTeX file to generate an SVG output.

    Args:
        file_name (str): Name of the LaTeX file to compile.

    Returns:
        None
    """
    try:
        # Run pdflatex to generate the PDF
        subprocess.run(['pdflatex', '-shell-escape', file_name], check=True)
        # Convert the PDF to SVG using pdf2svg
        pdf_file = file_name.replace(".tex", ".pdf")
        svg_file = file_name.replace(".tex", ".svg")
        subprocess.run(['pdf2svg', pdf_file, svg_file], check=True)
        print(f"Successfully compiled {file_name} to {svg_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling {file_name}: {e}")

# Example Usage

# Example usage: visualize a CSS code as TikZ
if __name__ == "__main__":
    from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
    code = RotatedSurfaceCode(31)
    code.get_node_coordinates()
    code.get_x_edge_coordinates()
    code.get_z_edge_coordinates()
    output_file = "rotated_xy_surface_l3.tex"
    draw_css_code_tanner_graph_tikz(code, output_file, qubit_radius=0.25, check_radius=0.4, spacing=1.8, qubit_label="q", x_check_label="S^X",z_check_label="S^Y", qubit_label_position="right", x_check_label_position="right", z_check_label_position="right")
    compile_latex(output_file)