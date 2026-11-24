"""Check GitHub once at startup for a newer Nocturnomath version."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version

import httpx

from .auth import env_flag_enabled

PYPROJECT_URL = (
    "https://raw.githubusercontent.com/fmottes/nocturnomath/main/pyproject.toml"
)
REPO_URL = "https://github.com/fmottes/nocturnomath"
UPGRADE_COMMAND = "uv tool upgrade nocturnomath"
DISABLE_ENV = "NOCTURNOMATH_NO_UPDATE_CHECK"


def _version_key(raw: str) -> tuple[int, ...] | None:
    parts = raw.strip().lstrip("v").split(".")
    if not all(part.isdigit() for part in parts):
        return None
    nums = [int(part) for part in parts]
    while len(nums) > 1 and nums[-1] == 0:
        nums.pop()
    return tuple(nums)


async def check_for_update(*, current: str | None = None) -> dict[str, str] | None:
    """Return update info when main's pyproject.toml has a newer version."""
    if env_flag_enabled(DISABLE_ENV):
        return None
    try:
        local = current if current is not None else version("nocturnomath")
    except PackageNotFoundError:
        return None
    local_key = _version_key(local)
    if local_key is None:
        return None
    try:
        async with httpx.AsyncClient(timeout=3.0, follow_redirects=True) as client:
            response = await client.get(PYPROJECT_URL)
            if response.status_code != 200:
                return None
            latest = tomllib.loads(response.text).get("project", {}).get("version")
    except (httpx.HTTPError, tomllib.TOMLDecodeError, ValueError, OSError):
        return None
    if not isinstance(latest, str):
        return None
    latest = latest.strip()
    latest_key = _version_key(latest)
    if latest_key is None or latest_key <= local_key:
        return None
    return {
        "current": local,
        "latest": latest,
        "url": REPO_URL,
        "command": UPGRADE_COMMAND,
    }
