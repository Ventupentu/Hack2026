# Contributions Guide

Thanks for contributing to Hack2026.

## Ways To Contribute
- Fix bugs or improve model quality.
- Improve data prep, training, inference, or evaluation scripts.
- Add or improve tests and documentation.

## Local Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Development Workflow
1. Create a branch for your change (example: `feat/better-reranking` or `fix/infer-bug`).
2. Keep changes focused and easy to review.
3. If behavior changes, update documentation (for example `README.md` or config docs).
4. Run relevant scripts before opening a PR.

Example:
```bash
python -m src.infer --help
pytest
```

## Pull Request Checklist
- [ ] Clear title and short description of what changed and why.
- [ ] Reproducible commands used to test the change.
- [ ] Updated docs/configs when needed.
- [ ] No large generated artifacts, model weights, or secrets added.

## Reporting Issues
Please include:
- What you expected to happen.
- What actually happened.
- Steps to reproduce.
- Environment details (OS, Python version, relevant command).
