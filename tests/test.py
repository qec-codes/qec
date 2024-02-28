import numpy as np
from ldpc.codes import rep_code, hamming_code, ring_code
from ldpc import mod2
from qec.css import CssCode
from qec.hgp import HyperGraphProductCode
from qec.codes import *
from qec.stab_code import StabCode
from qec.lifted_hgp import LiftedHypergraphProduct
import scipy.sparse
import ldpc
import qec


import numpy as np
from PIL import Image


def pcm_to_image(pcm, square_width):
    """
    Convert a binary parity check matrix to an image.

    Parameters:
    pcm (numpy.ndarray): Binary parity check matrix.
    square_width (int): Width of each square in pixels.

    Returns:
    Image: The generated image.
    """

    if isinstance(pcm, scipy.sparse.spmatrix):
        pcm = pcm.toarray()

    # Dimensions of the matrix
    rows, cols = pcm.shape

    # Create an empty image with light blue background
    img_size = (cols * square_width, rows * square_width)
    img = Image.new("RGB", img_size, color="lightblue")

    # Draw black squares for non-zero elements
    for i in range(rows):
        for j in range(cols):
            if pcm[i, j] != 0:
                top_left = (j * square_width, i * square_width)
                bottom_right = ((j + 1) * square_width, (i + 1) * square_width)
                img.paste(
                    "black",
                    [top_left[0], top_left[1], bottom_right[0], bottom_right[1]],
                )

    return img


# Example usage
pcm = hamming_code(3)
img = pcm_to_image(pcm, 10)
img.save("pcm_image.png")

a1 = ldpc.protograph.array(
    [
        [(0), (11), (7), (12)],
        [(1), (8), (1), (8)],
        [(11), (0), (4), (8)],
        [(6), (2), (4), (12)],
    ]
)

H = a1.to_binary(13)
print(H.shape)

pcm = H
img = pcm_to_image(pcm, 3)
img.save("seed.png")


qcode = qec.lifted_hgp.LiftedHypergraphProduct(13, a1, a1.T)

pcm = qcode.hz
img = pcm_to_image(pcm, 3)
img.save("hz.png")

pcm = qcode.hx
img = pcm_to_image(pcm, 3)
img.save("hx.png")
