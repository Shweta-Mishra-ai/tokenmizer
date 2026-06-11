## What this PR does

<!-- Brief description -->

## Type
- [ ] Bug fix
- [ ] Extraction improvement (graph_memory/)
- [ ] New provider
- [ ] Performance
- [ ] Documentation

## Tests
- [ ] `pytest tests/ -v` passes
- [ ] `ruff check tokenmizer/` clean
- [ ] Memory accuracy test added/updated (if extraction change)

## Checklist
- [ ] No raw dicts crossing layer boundaries (use DTOs)
- [ ] No `os.getenv()` outside `config/settings.py`
- [ ] External imports are lazy (inside functions, with try/except ImportError)
