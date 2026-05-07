# Testing

Install dev tooling using the repo's documented dev flow when available. The
pre-commit stack uses `ruff==0.14.1`.

Focused checks:

```bash
python -m ruff check <changed files>
python -m ruff format --check <changed files>
python -m pytest <focused tests> -m unit
```

Common focused suites:

```bash
python -m pytest tests/unit/cli -m unit
python -m pytest tests/unit/sdk -m unit
python -m pytest tests/unit/generator -m unit
```

Run broader tests when touching shared SDK behavior, task validation, model
parsing, or perf database lookup logic. If tests cannot run because optional
dependencies or LFS data are missing, report the blocker and the exact command.
