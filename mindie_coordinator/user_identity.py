"""Read a consumer-supplied GitHub identity snapshot for local attribution.

Shared-root deployments use this configured user for naming and coordination.
It is not an authentication token or an authorization boundary. This module
performs no network requests and never searches for a consumer workspace.
The confirmed login is sufficient for local attribution. A numeric GitHub ID
may accompany an authenticated snapshot, but is never required for local use.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from mindie_coordinator.client_paths import client_path

IDENTITY_FILE_ENV = "MINDIE_GITHUB_IDENTITY_FILE"


def load_github_identity(identity_file=None) -> dict | None:
    supplied = identity_file if identity_file is not None else os.environ.get(IDENTITY_FILE_ENV)
    if supplied is None:
        return None
    if not os.fspath(supplied).strip():
        raise ValueError(f"{IDENTITY_FILE_ENV} is configured but its path is empty")
    path = Path(client_path(supplied)).expanduser()
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ValueError(f"Cannot read configured GitHub identity file {path}: {exc.strerror or exc}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Configured GitHub identity file {path} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != "mindie.github.v1":
        raise ValueError(f"Configured GitHub identity file {path} must use schema mindie.github.v1")
    login = value.get("login")
    if not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", login):
        raise ValueError(f"Configured GitHub identity file {path} has an invalid personal login")
    user_id = value.get("github_user_id")
    if "github_user_id" in value and (type(user_id) is not int or user_id <= 0):
        raise ValueError(f"Configured GitHub identity file {path} requires a positive numeric github_user_id")
    return {"schema": "mindie.github.v1", "login": login.lower(),
            **({"github_user_id": user_id} if "github_user_id" in value else {})}
