# 🛠️ Updating *ZenReg* on PyPI

After releasing a new version on GitHub:

1. **Update the version number** in `pyproject.toml`
   Make sure it matches the new GitHub tag (e.g., `version = "1.0.4"` for tag `v1.0.4`)
   Do I need to commit this change before publishing? Yes: commit the upated `pyproject.toml` file to the main branch.

2. **Publish the updated package to PyPI**:
   ```bash
   flit publish
   ```

> ℹ️ If `flit` is not yet installed, do so with:

```bash
pip install flit
```

To update a ZenReg installation (installed via pip publicly), run

```bash
pip install --upgrade zenreg
```

To list outdated packages:

```bash
pip list --outdated
```
