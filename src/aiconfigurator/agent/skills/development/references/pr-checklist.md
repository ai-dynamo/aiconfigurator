# PR Checklist

Before committing:

```bash
git status --short --branch
python -m ruff check <changed files>
python -m ruff format --check <changed files>
python -m pytest <focused tests> -m unit
```

For commits that require DCO, use the intended human author and sign-off:

```bash
git commit -s --author="Name <email>" -m "Message"
```

If local credentials are missing, do not rewrite remote URLs permanently just to
push. Use the configured remote or a temporary credential method requested by
the user.

In the final summary, include changed areas, tests run, and any untracked files
left alone.
