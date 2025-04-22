![qec_banner](https://github.com/user-attachments/assets/25dcf89f-d0df-4eb7-a57a-9df09827eb76)

  <p align="center">
    QEC: Python Tools for Quantum Error Correction
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

## Base functionality: code constructions and "fundamental" analysis
We currently provide the construction methods for the stabilizer generators of the QEC code families below, along with optimized methods to obtain their fundamental properties. All the classes allow you to:

- obtain the physical qubit count
- obtain the logical qubit count
- calculate the exact code distance
- estimate the minimum code distance
- obtain a logical basis

The classes (along with their extra methods) are:

- Hyperprgarph Product (HGP) codes, with methods:
  - construct the x and z stabiliser matrices from the seed codes
  - obtain the canonical basis (work in progress)
 
- Surface codes
  - Unrotated Surface code
  - Periodic Surface XZZX code
  - Rotated Surface XZZX code
 
- Toric code
    
- Calderbank-Shor-Steane (CSS) codes
  - check that the seed codes satisfy the CSS criteria
    
- Stabiliser codes
  - check that the input stabiliser matrix is valid

## Circuit compilation 

>
> _Note:_ this functionality is still work in progress. The corresponding code is not part of the `main` branch - you can find it on the `circuit_compilation` branch to play around.  
>

Currently we only support circuit compilation for memory experiments of:

- HGP codes with:
  - "coloration circuit" stabilizer schedule (twice the depth of the most optimal "cardinal circuit" method)
  - _under development:_ "cardinal circuit" stabilizer schedule

One can either compile noisy or noisless circuits (for further compilation to one's own needs). To create a nosiy circuit one needs to construct their own noise model, or use one of the available presets:
- `uniform_depolarizing_noise` 
- `non_uniform_depolarizing_noise` under development 
- `phenomenological_noise` 

For a more detailed example please see the `demo.ipynb` [notebook](https://github.com/qec-codes/qec/blob/circuit_compilation/demo/demo.ipynb), inside the `demo/` folder of the `circuit_compilation` branch.

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

# Examples

## Base functionality:

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
In [5]: steane_code.compute_exact_code_distance()
In [6]: print(steane_code)

Out [7]: Steane Code: [[N=7, K=1, dx<=3, dz<=3]]
```

we can see that all the fundamental properties (N - physical qubit number, K - logical qubit number, dx - X distance, dz - Z distance). We can continue our analysis by taking a look at the x and z logical basis of the code: 

```python
In [7]: print(steane_code.x_logical_operator_basis.toarray())
Out [8]: [[1 1 0 1 0 0 0]]
In [9]: print(steane_code.z_logical_operator_basis.toarray())
Out [10]: [[1 1 0 1 0 0 0]]
```

## Circuit compilation:

```python
In [10]: from qec.code_constructions import HypergraphProductCode
In [11]: from qec.circuit_compilation import MemoryExperiment

In [12]: hgp_example_code = HypergraphProductCode(hamming_code, hamming_code)

In [12]: hgp_memory_example = MemoryExperiment(hgp_example_code)

In [13]: hgp_X_mem_circuit = hgp_memory_example.circuit(basis = 'X', rounds = 1, noise = False)
```

To see the output circuit and a more detailed example (including a simulation) see the `demo.ipynb` [notebook](https://github.com/qec-codes/qec/blob/circuit_compilation/demo/demo.ipynb), inside the `demo/` folder of the `circuit_compilation` branch.

                              





