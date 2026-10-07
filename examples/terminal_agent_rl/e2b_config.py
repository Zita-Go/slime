"""Load E2B credentials from environment or a protected JSON file."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _protected_config() -> tuple[str, str | None, str | None] | None:
    raw_path = os.environ.get("TERMINAL_RL_E2B_CONFIG_FILE", "").strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    info = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"E2B config must be a regular non-symlink file: {path}")
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        raise RuntimeError(f"E2B config permissions must not exceed 0600: {path} mode={mode:o}")
    if info.st_size <= 0 or info.st_size > 16 * 1024:
        raise RuntimeError(f"E2B config size is invalid: {path} size={info.st_size}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("E2B config must be a JSON object")
    api_key = str(payload.get("api_key") or "").strip()
    api_url = str(payload.get("api_url") or "").strip() or None
    domain = str(payload.get("domain") or "").strip() or None
    if not api_key:
        raise RuntimeError("E2B config has no api_key")
    return api_key, api_url, domain


def resolve_e2b_connection() -> tuple[str, str | None, str | None]:
    api_key = _first_env(
        "NEXTASK_BENCH_E2B_API_KEY",
        "NEXTASK_SOLVE_E2B_API_KEY",
        "NEXTASK_GEN_E2B_API_KEY",
        "E2B_API_KEY",
        "E2B_API_KEY_CODEX",
    )
    api_url = _first_env(
        "NEXTASK_BENCH_E2B_API_URL",
        "NEXTASK_SOLVE_E2B_API_URL",
        "NEXTASK_GEN_E2B_API_URL",
        "E2B_API_URL",
    )
    domain = _first_env(
        "NEXTASK_BENCH_E2B_DOMAIN",
        "NEXTASK_SOLVE_E2B_DOMAIN",
        "NEXTASK_GEN_E2B_DOMAIN",
        "E2B_DOMAIN",
    )
    if api_key:
        return api_key, api_url, domain
    protected = _protected_config()
    if protected is not None:
        return protected
    raise RuntimeError("E2B credentials are not configured")
