# Contributing to QEC
---

## Repo structure

- Code constructions go under `src/qec/codes/` into a folder with their name.
- Utilities that can be used in code constructions go under `src/qec/utils`
- More detailed examples e.g. in the form of jupyter notebooks are under `src/qec/examples`
- Finally tests go into `test/`

```bash 
├── src 
│   └── qec
│       ├── codes
│       │   └── hgp
│       └── utilities
├── examples
└── tests 
```

## Testing 

For testing use [pytest](https://docs.pytest.org/en/stable/getting-started.html)

## Stylistic guideline

- Set up your code editor to use [Ruff](https://docs.astral.sh/ruff/) -- yes there is a [VSCode extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).

- Follow the NumPy docstring [standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard). 


    
