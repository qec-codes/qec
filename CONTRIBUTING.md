# Contributing to QEC
---

## Repository structure

### Production code
All core code resides in the `src/qec/` directory, with the following main subfolders:
- **`code_constructions/`** – For QEC code construction methods (e.g., `hgp_code.py`).
- **`code_instances/`** – For specific code instances (e.g., `five_qubit_code.py`).
- **`utils/`** – For utility modules used across the various code constructions (e.g., `binary_pauli_utils.py`, `sparse_binary_utils.py`).

If a functionality is added that can be considered as a stand-alone extension, e.g. the `codetables_de.py` it should be placed in a separate folder under `src/qec/`, e.g.  `src/qec/codetables_de/codetables_de.py`, where the name refers to the functionality.

### Testing
All tests live in the `tests/` directory and **must mirror** the structure of `src/qec/`. This means:
- If a new folder or module is added under `src/qec/…`, create a corresponding folder or module under `tests/…`.
- For instance, `src/qec/code_instances/five_qubit_code.py` is tested by `tests/code_instances/test_five_qubit_code.py`.

### Examples
More detailed examples, such as Jupyter notebooks or demonstrations, should be placed in the `examples/` folder.

## Testing

We use [pytest](https://docs.pytest.org/en/stable/getting-started.html) for all testing needs. Ensure that all tests pass locally before submitting a pull request.

## Stylistic guideline

- Configure your editor to use [Ruff](https://docs.astral.sh/ruff/) (a [VSCode extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) is available).
- Follow the NumPy docstring [standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) for all documentation.
