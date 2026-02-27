# Phase 1: Infrastructure Unblock - Research

**Researched:** 2026-02-27
**Domain:** Python merge conflict resolution, secrets management via environment variables
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BUG-01 | Bot can start without errors — git merge conflict in `Scripts/main.py` resolved | Conflict is on lines 11-17; exact resolution identified; both sides examined |
| BUG-02 | Database credentials loaded from environment variables — hardcoded AWS RDS password and hostname removed from `Scripts/config.py` | Exact lines identified (13-14); standard `os.getenv` without fallback pattern documented; git history contains credentials (5 commits); rotation required |
</phase_requirements>

## Summary

Phase 1 has two concrete, fully-understood bugs. Neither requires a library, design pattern, or architectural decision — both are single-file edits with known correct resolutions. The merge conflict in `Scripts/main.py` lines 11-17 is blocking Python import, making the file unparseable. The credential exposure in `Scripts/config.py` lines 13-14 places a plaintext AWS RDS password and hostname as `os.getenv()` default fallbacks, meaning they are committed to git history and will be used silently if env vars are absent.

The fix for BUG-01 is to remove the conflict markers and keep the "Updated upstream" side's imports (`.database` and `.data_fetch` with relative import syntax), because the "Stashed changes" side uses the incorrect absolute `Scripts.data_fetch` path that would fail when `main.py` is run as a module. The fix for BUG-02 is to call `os.getenv("DB_PASSWORD")` and `os.getenv("DB_HOST")` with no default, then raise a descriptive `ValueError` at module load time if either is `None`. A `.env.example` file should be added (with placeholder values) to document required variables without exposing real values.

A critical prerequisite for BUG-02 that is not a code change: the exposed credentials are in git history across multiple commits and the AWS RDS password must be rotated before any new commits are made. The planner must include this as a manual step that cannot be automated.

**Primary recommendation:** Fix the merge conflict first (BUG-01), then remove credential defaults and add startup validation (BUG-02). Rotate the RDS password as a manual prerequisite before committing BUG-02 changes.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python `os` module | stdlib | Read environment variables | No install; `os.getenv()` is the canonical approach |
| Python-dotenv (`python-dotenv`) | 1.x | Load `.env` file into environment for local dev | De facto standard for Python local dev env management |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `python-dotenv` | 1.x | `load_dotenv()` at app entry point to populate os.environ from `.env` | Local development only; production uses real env vars from OS/container |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `python-dotenv` | `direnv`, manual `export` | `python-dotenv` is portable across Windows/Linux/Mac; no shell config required |
| `raise ValueError` on missing var | `sys.exit(1)` with message | `ValueError` gives a traceable stack; `sys.exit` is cleaner for CLI but loses context |

**Installation (if python-dotenv not already present):**
```bash
pip install python-dotenv
```

## Architecture Patterns

### Recommended Project Structure

No structural changes required for this phase. Both fixes are confined to existing files:

```
Scripts/
├── main.py          # BUG-01: remove merge conflict markers, keep correct imports
├── config.py        # BUG-02: remove default values from DB_PASSWORD and DB_HOST
├── ...
.env                 # New: local dev credentials (gitignored — already in .gitignore)
.env.example         # New: placeholder template committed to repo
```

### Pattern 1: Merge Conflict Resolution (Choosing the Correct Side)

**What:** When a merge conflict has two versions, you must identify which is correct and keep it wholesale, removing all conflict markers.

**When to use:** The conflict markers `<<<<<<<`, `=======`, `>>>>>>>` must be removed entirely. Python will raise `SyntaxError` on any file containing them.

**The conflict in `Scripts/main.py` lines 11-17:**

```
<<<<<<< Updated upstream
from .database import engine
from .data_fetch import fetch_and_load_symbols, fetch_historical_data, insert_historical_data
=======
from .database import engine, get_session, StockData
from Scripts.data_fetch import fetch_and_load_symbols, fetch_historical_data, insert_historical_data
>>>>>>> Stashed changes
```

**Correct resolution — keep "Updated upstream" side, but also keep `get_session` and `StockData`:**

The "Stashed changes" side imports `get_session` and `StockData` from `.database`, which ARE used later in `main.py` (lines 158-161 and 277-280 in `hourly_portfolio_scan` call `get_session()` and query `StockData`). The "Updated upstream" side omits those, which would cause `NameError` at runtime. The "Stashed changes" side uses `Scripts.data_fetch` (absolute) which is wrong for module execution. The correct merge is:

```python
from .database import engine, get_session, StockData
from .data_fetch import fetch_and_load_symbols, fetch_historical_data, insert_historical_data
```

This takes the symbols from the stashed side and the import path syntax from the upstream side.

### Pattern 2: Fail-Fast Environment Variable Validation

**What:** Call `os.getenv()` without a default, then immediately raise if the result is `None`.

**When to use:** Any credential or infrastructure-specific value that must never have a hardcoded fallback.

**Example (correct pattern):**
```python
import os

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill in the values."
        )
    return value

DB_PASSWORD = _require_env("DB_PASSWORD")
DB_HOST = _require_env("DB_HOST")
```

This raises at import time with a clear message, satisfying success criterion 3.

### Pattern 3: python-dotenv Integration at Entry Point

**What:** Call `load_dotenv()` once at the top of `main.py` before any imports that read config.

**Example:**
```python
from dotenv import load_dotenv
load_dotenv()  # reads .env in cwd or parent dirs, populates os.environ

# All subsequent imports now see the env vars
from Scripts.config import IB_HOST, IB_PORT, ...
```

### Anti-Patterns to Avoid

- **Keeping any default in `os.getenv("DB_PASSWORD", "...")`:** Even `os.getenv("DB_PASSWORD", "")` is dangerous — an empty string will connect to the DB with a blank password rather than failing clearly.
- **Putting `load_dotenv()` inside `config.py`:** Config is a module that may be imported in multiple contexts (tests, CLI tools). Entry point (`main.py`) is the right place.
- **Resolving the conflict by combining both import lines:** The absolute import `from Scripts.data_fetch import ...` will fail when running `python Scripts/main.py` because Python doesn't add `Scripts/` to `sys.path` in that way — it will produce `ModuleNotFoundError`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Loading `.env` into environment | Custom file parser | `python-dotenv` | Handles quoting, multiline values, comments, encoding edge cases |
| Checking for missing env vars | One-off inline checks | Centralized `_require_env()` helper | Consistent error messages; single place to update format |

**Key insight:** For this phase, all problems are already solved by the stdlib and one small library. The goal is deletion and simplification, not new code.

## Common Pitfalls

### Pitfall 1: Conflict Resolution Creates a New NameError

**What goes wrong:** The developer resolves the conflict by keeping only the "Updated upstream" side, which drops `get_session` and `StockData`. `main.py` then fails at runtime with `NameError: name 'get_session' is not defined` inside `hourly_portfolio_scan`.

**Why it happens:** The conflict resolution tools and `CONCERNS.md` note says "keep upstream side" but don't account for that the stashed side added new symbols to the database import.

**How to avoid:** After resolving, grep the file for all usages of `get_session` and `StockData` before running. Both appear in `hourly_portfolio_scan` (lines 158, 277 in the original file).

**Warning signs:** `NameError` on `get_session` or `StockData` immediately after starting the bot.

### Pitfall 2: Credentials Remain in Git History After Code Fix

**What goes wrong:** `config.py` is edited to remove defaults, but the password `Stocks123` and hostname `database-2.ctwgq2kqgrl6.us-east-2.rds.amazonaws.com` remain readable in prior commits via `git log -p` or `git show`.

**Why it happens:** Git history is permanent. Editing a file does not rewrite history.

**How to avoid:** The RDS password MUST be rotated in AWS before or simultaneously with the code commit. History rewriting (BFG Repo Cleaner / `git filter-repo`) is optional for a private repo with a single developer, but the password rotation is mandatory regardless.

**Warning signs:** `git log -S "Stocks123" --all` returns results (confirmed: it does return 5 commits as of this research).

### Pitfall 3: `.env` File Accidentally Committed

**What goes wrong:** Developer creates `.env` with real credentials, then runs `git add .` and commits it.

**Why it happens:** `.env` is a new file that `git add .` will pick up if not already tracked.

**How to avoid:** The `.gitignore` already contains `.env` (line 108). Verify `.env` is listed in `.gitignore` before creating the file. Run `git status` after creating `.env` to confirm it appears as "untracked/ignored" rather than staged.

**Warning signs:** `git status` shows `.env` as a modified/new file (it should be invisible).

### Pitfall 4: Silent Failure When .env Not Present in Production

**What goes wrong:** `load_dotenv()` silently does nothing if `.env` doesn't exist. If `_require_env()` is not used, the app starts with `None` credentials and fails at database connection time with a cryptic `psycopg2` error instead of a clear startup message.

**Why it happens:** `load_dotenv()` is designed to be a no-op when no file exists — that's correct behavior for production. But it means the fail-fast validation in `config.py` is the only safeguard.

**How to avoid:** Always use `_require_env()` (raising `ValueError`) rather than `os.getenv()` alone for required credentials.

## Code Examples

### Exact Conflict Resolution for `Scripts/main.py` lines 11-17

```python
# Source: Direct analysis of Scripts/main.py conflict markers
# Remove lines 11-17 entirely and replace with:
from .database import engine, get_session, StockData
from .data_fetch import fetch_and_load_symbols, fetch_historical_data, insert_historical_data
```

### Credential Removal for `Scripts/config.py` lines 12-16

Current (insecure):
```python
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Stocks123")
DB_HOST = os.getenv("DB_HOST", "database-2.ctwgq2kqgrl6.us-east-2.rds.amazonaws.com")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
```

Target (secure with fail-fast):
```python
def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and set your credentials."
        )
    return value

DB_USER = os.getenv("DB_USER", "postgres")      # non-sensitive, default is fine
DB_PASSWORD = _require_env("DB_PASSWORD")         # required, no default
DB_HOST = _require_env("DB_HOST")                 # required, no default
DB_PORT = os.getenv("DB_PORT", "5432")           # non-sensitive, default is fine
DB_NAME = os.getenv("DB_NAME", "postgres")       # non-sensitive, default is fine
```

### `.env.example` Template

```bash
# Copy this file to .env and fill in your values.
# .env is gitignored — never commit credentials.

# Database (AWS RDS or local PostgreSQL)
DB_PASSWORD=your_database_password_here
DB_HOST=your_rds_hostname.us-east-2.rds.amazonaws.com

# Optional overrides (defaults shown)
# DB_USER=postgres
# DB_PORT=5432
# DB_NAME=postgres
```

### Verifying the Fix

```bash
# Confirm no merge conflict markers remain
grep -n "<<<<<<\|=======\|>>>>>>>" Scripts/main.py
# Expected: no output

# Confirm no hardcoded credentials remain
grep -n "Stocks123\|ctwgq2kqgrl6" Scripts/config.py
# Expected: no output

# Confirm credentials are still in history (rotation reminder)
git log -S "Stocks123" --all --oneline
# Expected: still shows commits — that's why rotation is mandatory
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded credentials | `os.getenv()` with default fallback | Common beginner pattern | Still exposes credentials; just one step better than literals |
| `os.getenv()` with default | `os.getenv()` with no default + explicit validation | Current best practice | Fails clearly at startup instead of silently using wrong value |
| Manual `.env` management | `python-dotenv` | ~2014 (library exists since 2012) | Portable, handles edge cases, works on Windows |

**Deprecated/outdated:**
- `os.getenv("SECRET", "hardcoded_fallback")`: Considered an antipattern for any value that is secret or environment-specific. The fallback defeats the purpose of environment variable externalization.

## Open Questions

1. **Should git history be rewritten to remove credentials?**
   - What we know: The password and hostname appear in 5 commits. The repo appears to be a single-developer private repo on the local machine.
   - What's unclear: Whether the repo has any remote origin (no remote configured at time of research); whether it has ever been pushed to GitHub or another host.
   - Recommendation: Rotate the password first (mandatory). For history rewriting, check `git remote -v` — if no remote exists, history rewriting is low priority. If a remote exists, use `git filter-repo --path Scripts/config.py --invert-paths` or BFG Repo Cleaner on the specific string, then force-push and rotate all collaborators.

2. **Does `python-dotenv` need to be added to requirements?**
   - What we know: No `requirements.txt` or `pyproject.toml` was found in the repo root (not searched exhaustively).
   - What's unclear: Whether `python-dotenv` is already installed in the `.venv`.
   - Recommendation: Planner should include a task to check for `requirements.txt` and add `python-dotenv` if not present, or install it directly.

3. **Is there a `Scripts/__init__.py` that imports from `config.py` at package load time?**
   - What we know: `Scripts/__init__.py` exists. If it imports `config`, then the `_require_env` validation fires at import of the `Scripts` package.
   - What's unclear: Contents of `Scripts/__init__.py`.
   - Recommendation: Read `Scripts/__init__.py` during planning to confirm the import chain and ensure the validation fires before any network connections are attempted.

## Sources

### Primary (HIGH confidence)

- Direct file analysis: `Scripts/main.py` lines 11-17 — conflict markers and both sides inspected
- Direct file analysis: `Scripts/config.py` lines 13-14 — hardcoded values confirmed
- Direct file analysis: `.gitignore` lines 108, 244-245 — `.env` already ignored
- `git log -S "Stocks123" --all --oneline` — confirmed credentials in 5 commits
- `git log -S "ctwgq2kqgrl6" --all --oneline` — confirmed RDS hostname in 3 commits

### Secondary (MEDIUM confidence)

- Python `os.getenv` documentation: stdlib, no version concerns; behavior is stable
- `python-dotenv` library: widely used, stable API; `load_dotenv()` is the standard entry point

### Tertiary (LOW confidence)

- None.

## Metadata

**Confidence breakdown:**
- BUG-01 resolution: HIGH — conflict markers and both import sides inspected directly; correct merge identified by cross-referencing usages of `get_session`/`StockData` in `hourly_portfolio_scan`
- BUG-02 resolution: HIGH — credential lines confirmed; `_require_env` pattern is standard Python; `.gitignore` already covers `.env`
- Git history exposure: HIGH — confirmed by direct `git log -S` query; 5 commits contain password

**Research date:** 2026-02-27
**Valid until:** 2026-03-29 (stable domain — no library version concerns; valid indefinitely until codebase changes)
