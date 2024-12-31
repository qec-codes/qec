# Contributing to QEC
---

## Repository structure

### Production code
All core code resides in the `src/qec/` directory, with the following main subfolders:
- **`quantum_codes/`** – For code related to specific quantum code constructions (e.g., `five_qubit_code.py`).
- **`stabilizer_code/`** – For stabilizer-code-specific implementations (e.g., `css_code.py`, `stabilizer_code.py`).
- **`utils/`** – For utility modules used across the various code constructions (e.g., `binary_pauli_utils.py`, `sparse_binary_utils.py`).

### Testing
All tests live in the `tests/` directory and **must mirror** the structure of `src/qec/`. This means:
- If a new folder or module is added under `src/qec/…`, create a corresponding folder or module under `tests/…`.
- For instance, `src/qec/quantum_codes/five_qubit_code.py` is tested by `tests/quantum_codes/test_five_qubit_code.py`.

### Examples
More detailed examples, such as Jupyter notebooks or demonstrations, should be placed in the `examples/` folder.

## Testing

We use [pytest](https://docs.pytest.org/en/stable/getting-started.html) for all testing needs. Ensure that all tests pass locally before submitting a pull request.

## Stylistic guideline

- Configure your editor to use [Ruff](https://docs.astral.sh/ruff/) (a [VSCode extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) is available).
- Follow the NumPy docstring [standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) for all documentation.
