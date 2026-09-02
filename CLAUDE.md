# CLAUDE.md — Project guidance for AI assistants

## Documentation policy

**Keep the docs in sync with the code.** Any change that affects user-facing behavior must be reflected in the documentation before the work is considered done.

The documentation lives in two places:

| File | Purpose |
|------|---------|
| `README.md` | High-level overview, installation, quick start |
| `mkdocs.yml` | MkDocs configuration (nav, theme, plugins) |
| `docs/index.md` | GitHub Pages landing page |
| `docs/getting-started.md` | Step-by-step tutorial for analysts |

### What always requires a doc update

- Adding, renaming, or removing a keyboard shortcut → update the shortcuts table in `docs/getting-started.md`
- Changing a CLI argument or the `cochleogram` entry point → update `README.md` and `docs/getting-started.md`
- Adding support for a new file format → update the installation extras table in both `README.md` and `docs/getting-started.md`
- Adding or removing a species → update the species table in `docs/index.md`, `docs/getting-started.md`, and the nav in `mkdocs.yml` if a new page is added
- Changing the analysis workflow (new step, reordered steps, renamed modes) → update `docs/getting-started.md`
- Changing default channel names or stain colors → update `docs/getting-started.md` where channels are described
- Adding a new export format option → update Step 6 in `docs/getting-started.md`
- Changing the save-file format or naming convention → update the relevant steps in `docs/getting-started.md`

### What does NOT require a doc update

- Internal refactors that have no user-visible effect
- Performance improvements with no behavior change
- Test additions or changes
- Changes to `cochleogram/plot.py` internals that don't affect output file formats or UI options

## Code conventions

- Entry point: `cochleogram.main:main`
- Config constants: `cochleogram/config.py` — update `CELLS`, `CHANNEL_CONFIG`, key maps, and species settings there
- Serialization: JSON via `get_state()` / `set_state()` on model objects in `cochleogram/model.py`
- File format detection is automatic based on file extension / folder contents (see `cochleogram/readers.py`)

## Commit messages

- Do not include `Claude` or `Claude-Session` in commit messages or PR bodies.
