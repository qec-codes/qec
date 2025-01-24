<h1>
<p align="center">
  <br>QEC
</h1>
  <p align="center">
    Python Tools for Quantum Error Correction
    <br />
    <a href="https://qec.codes/">Website</a>
    ·
    <a href="#features">Features</a>
    ·
    <a href="#installation">Installation</a>
    ·
    <a href="#examples">Examples</a>
  </p>
</p>

# Features

## Code constructions and fundamental analysis
We currently provide the construction methods for the stabilizer generators of the QEC code families below, along with optimized methods to obtain their fundamental properties. All the classes allow to:

- obtain the physical qubit count
- obtain the logical qubit count
- calculate the exact code distance
- estimate the minimum code distance
- obtain a logical basis

The classes (along with their extre methods) are:

- Hyperprgarph Product (HGP) codes, with methods:
  - construct the x and z stabiliser matrices from the seed codes
  - obtain the canonical basis (work in progress)
    
- Calderbank-Shor-Steane (CSS) codes
  - check that the seed codes satisfy the CSS criteria
    
- Stabiliser codes
  - check that the input stabiliser matrix is valid

## Circuit compilation
Work in progress.

# Installation 

Simply do:
```bash
pip install qec
```

or obtain a local copy of the package by cloning it, then navigate into the created folder: 

```bash
git clone git@github.com:qec-codes/qec.git
cd qec
```

Finally, install the package:

```bash
pip install -e .
```

You are all set! To import the package use:

```python
In  [1]: import qec

In  [2]: qec.__version__
Out [2]: '0.1.0'

```

# Examples

In this example we are going to create the Steane code, and obtain its fundamental code properties. We start by initialising its seed matrices (the [7, 4, 3] Hamming code):
```python
In [1]: import numpy as np
In [2]: from qec.code_constructions import CSSCode

In [3]: hamming_code = np.array([[1, 0, 0, 1, 0, 1, 1],
                                 [0, 1, 0, 1, 1, 0, 1],
                                 [0, 0, 1, 0, 1, 1, 1]])
```
as the Steane code is part of the CSS code family, we can use the `CSSCode` class:

```python
In [4]: steane_code = CSSCode(x_stabilizer_matrix = hamming_code,
                              z_stabilizer_matrix = hamming_code,
                              name = "Steane")
In [5]: print(steane_code)
Out [6]: Steane Code: [[N=7, K=1, dx<=3, dz<=3]]
```

we can see that all the fundamental properties (N - physical qubit number, K - logical qubit number, dx - X distance, dz - Z distance) are pre-calculated for us. We can continue our analysis by taking a look at the x and z logical basis of the code: 

```python
In [7]: print(steane_code.x_logical_operator_basis.toarray())
Out [8]: [[1 1 0 1 0 0 0]]
In [9]: print(steane_code.z_logical_operator_basis.toarray())
Out [10]: [[1 1 0 1 0 0 0]]
```

                              





