# Contributing

Author: Fabrizio Musacchio  
Date: June 2026

ZenReg is currently a small starting point for a standalone registration package.
Contributions are welcome, especially if they improve robustness, documentation,
tests, or scientific reproducibility.

## Development setup

```bash
python -m pip install -e ".[dev]"
```

## Contribution guidelines

- Keep public functions documented with NumPy-style docstrings.
- Preserve canonical `TZCYX` stack semantics unless a function explicitly says otherwise.
- Add tests for behavior that changes registration output or public API.
- Keep examples small enough to run on a normal laptop.
- Prefer explicit, reproducible parameters over hidden defaults.

## Suggested next steps

- Add unit tests for shift estimation and stack transformations.
- Add support for additional registration models beyond translation.
- Add richer metadata and report writing for registration workflows.
- Add optional visual quality-control helpers.
