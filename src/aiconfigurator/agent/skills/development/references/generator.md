# Generator Changes

Before editing anything under `src/aiconfigurator/generator/**`, read:

```bash
sed -n '1,220p' .claude/rules/generator-development.md
```

Keep generator changes traceable to one behavior:

- API/config schema changes belong near `generator/api.py` and config defaults.
- Rendering behavior belongs near the rendering engine and backend templates.
- Backend-specific output belongs in the relevant backend template tree.
- Add tests under `tests/unit/generator/` for changed rendering or arguments.

Do not edit generator templates by guesswork. Compare the generated output or
focused snapshot expected by the existing tests.
