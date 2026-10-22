"""Image selection and host image-resolution primitives for provisioning.

[Role]
Resolves Ascend vllm-ascend image selectors and explicit references into
registry-pinned candidates, and owns the shared non-interactive local command
helpers used by host provisioning.

[Invariants]
- Never accepts bare repositories, moving `:latest` tags, or the legacy `auto`
  selector.
- Machine suffixes follow `IMAGE_SUFFIX_BY_MACHINE_TYPE`; registry mirror order
  is fixed by `IMAGE_REGISTRY_MIRRORS`.
- Shared constants (`SENTINEL`, `PROGRESS_SENTINEL`, `SOC_TO_MACHINE_TYPE`) stay
  importable without importing host transport modules.

[Timeouts & Locks]
- Release metadata lookups use `LATEST_RELEASE_TIMEOUT_SECONDS` (10s).
- `run_local` defaults to a bounded 120s subprocess timeout and never uses a
  shell.

[Side-Effect Policy]
- Network reads are limited to the pinned GitHub release endpoints; no remote
  host state is mutated here.
- Local command failures raise `MachineManagementError`; `TimeoutError` is never
  swallowed.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from mindie_coordinator.device_inventory import SOC_TO_MACHINE_TYPE

PROGRESS_SENTINEL = "__MINDIE_PROGRESS__="


IMAGE_REGISTRY_NJU = "quay.nju.edu.cn/ascend/vllm-ascend"
IMAGE_REGISTRY_OFFICIAL = "quay.io/ascend/vllm-ascend"
IMAGE_REGISTRY_MIRRORS = (IMAGE_REGISTRY_NJU, IMAGE_REGISTRY_OFFICIAL)
IMAGE_SELECTOR_RC = "rc"
IMAGE_SELECTOR_MAIN = "main"
IMAGE_SELECTOR_STABLE = "stable"
IMAGE_SELECTOR_LOCAL_LATEST = "local-latest"
IMAGE_POLICY_LOCAL_LATEST = "latest-local-vllm-ascend"
IMAGE_SELECTOR_ALIASES = {
    IMAGE_SELECTOR_RC: IMAGE_SELECTOR_RC,
    "release-candidate": IMAGE_SELECTOR_RC,
    "release-candidates": IMAGE_SELECTOR_RC,
    "latest-rc": IMAGE_SELECTOR_RC,
    "latest-prerelease": IMAGE_SELECTOR_RC,
    "latest-pre-release": IMAGE_SELECTOR_RC,
    "prerelease": IMAGE_SELECTOR_RC,
    "pre-release": IMAGE_SELECTOR_RC,
    IMAGE_SELECTOR_MAIN: IMAGE_SELECTOR_MAIN,
    "main-branch": IMAGE_SELECTOR_MAIN,
    IMAGE_SELECTOR_STABLE: IMAGE_SELECTOR_STABLE,
    "release": IMAGE_SELECTOR_STABLE,
    "latest-release": IMAGE_SELECTOR_STABLE,
    "latest-official": IMAGE_SELECTOR_STABLE,
    IMAGE_SELECTOR_LOCAL_LATEST: IMAGE_SELECTOR_LOCAL_LATEST,
}
LEGACY_IMAGE_SELECTORS = {"auto"}
FORBIDDEN_IMAGE_TAGS = {"latest"}
DEFAULT_IMAGE = IMAGE_SELECTOR_RC
DEFAULT_IMAGE_CANDIDATES = tuple(f"{repo}:main" for repo in IMAGE_REGISTRY_MIRRORS)
RELEASES_API = (
    "https://api.github.com/repos/vllm-project/vllm-ascend/releases?per_page=100"
)
RELEASES_PAGE = "https://github.com/vllm-project/vllm-ascend/releases"
LATEST_RELEASE_PAGE = "https://github.com/vllm-project/vllm-ascend/releases/latest"
LATEST_RELEASE_TIMEOUT_SECONDS = 10
IMAGE_RESOLVER_USER_AGENT = "mindie-machine-management/1.0"
DEFAULT_HOST_USER = "root"
DEFAULT_HOST_PORT = 22

SENTINEL = "__MINDIE_JSON__="

SEMVER_TAG_PATTERN = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:(?P<kind>rc)(?P<kind_number>\d+))?$",
    re.IGNORECASE,
)
MACHINE_TYPE_CHOICES = ("A2", "A3", "A5", "310P")
IMAGE_SUFFIX_BY_MACHINE_TYPE = {
    "A2": "",
    "A3": "-a3",
    "A5": "-a5",
    "310P": "-310p",
}


LOCAL_IMAGE_DISCOVERY_PY = r'''
def _mindie_image_ref_repo(ref):
    without_digest = ref.split("@", 1)[0]
    last_slash = without_digest.rfind("/")
    last_colon = without_digest.rfind(":")
    if last_colon > last_slash:
        return without_digest[:last_colon]
    return without_digest


def _mindie_image_ref_machine_type(ref):
    if "@" in ref:
        return None
    last_slash = ref.rfind("/")
    last_colon = ref.rfind(":")
    if last_colon <= last_slash:
        return None
    tag = ref[last_colon + 1:].lower()
    if any(marker in tag for marker in ("-a5", "-950dt", "-950pr")):
        return "A5"
    if "-310p" in tag:
        return "310P"
    if "-a3" in tag:
        return "A3"
    return "A2"


def _mindie_local_ref_rank(ref):
    is_digest = "@" in ref
    tag = ""
    if not is_digest:
        last_slash = ref.rfind("/")
        last_colon = ref.rfind(":")
        if last_colon > last_slash:
            tag = ref[last_colon + 1:].lower()
    return (1 if is_digest else 0, 1 if tag == "latest" else 0, ref)


def select_latest_local_vllm_ascend_image(image_inspects, machine_type=None):
    matched_images = []
    eligible_images = []
    for image in image_inspects:
        if not isinstance(image, dict):
            continue
        references = []
        for ref in list(image.get("RepoTags") or []) + list(image.get("RepoDigests") or []):
            if not isinstance(ref, str) or not ref.strip():
                continue
            repo = _mindie_image_ref_repo(ref.strip())
            if "vllm-ascend" not in repo.rsplit("/", 1)[-1].lower():
                continue
            references.append(ref.strip())
        references = sorted(set(references), key=_mindie_local_ref_rank)
        if not references:
            continue
        compatible_references = [
            ref
            for ref in references
            if machine_type is None or _mindie_image_ref_machine_type(ref) == machine_type
        ]
        entry = {
            "image_id": str(image.get("Id") or ""),
            "created": str(image.get("Created") or ""),
            "references": references,
            "compatible_references": compatible_references,
            "compatible": bool(compatible_references),
        }
        matched_images.append(entry)
        if compatible_references:
            eligible = dict(entry)
            eligible["selected_reference"] = compatible_references[0]
            eligible_images.append(eligible)

    matched_images.sort(
        key=lambda item: (item["created"], item["image_id"]), reverse=True
    )
    eligible_images.sort(
        key=lambda item: (item["created"], item["image_id"]), reverse=True
    )
    selected = eligible_images[0] if eligible_images else None
    return {
        "machine_type": machine_type,
        "matched_count": len(matched_images),
        "eligible_count": len(eligible_images),
        "matched_images": matched_images,
        "selected_reference": selected.get("selected_reference") if selected else None,
        "selected_image_id": selected.get("image_id") if selected else None,
        "selected_created": selected.get("created") if selected else None,
    }
'''


class MachineManagementError(RuntimeError):
    """Raised for deterministic, user-facing failures."""


@dataclass(frozen=True)
class SshTarget:
    host: str
    user: str = DEFAULT_HOST_USER
    port: int = DEFAULT_HOST_PORT


@dataclass(frozen=True)
class ImageResolution:
    selector: str
    requested: str
    policy: str
    candidates: tuple[str, ...]
    resolved_tag: str | None = None
    mirror_order: tuple[str, ...] = IMAGE_REGISTRY_MIRRORS
    machine_type: str | None = None


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def normalize_machine_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper().replace("_", "")
    if normalized == "310P":
        return "310P"
    if normalized in {"A2", "A3", "A5"}:
        return normalized
    raise MachineManagementError(
        f"unsupported machine type {value!r}; expected one of: {', '.join(MACHINE_TYPE_CHOICES)}"
    )


def image_tag_for_machine(base_tag: str, machine_type: str | None) -> str:
    normalized = normalize_machine_type(machine_type) or "A2"
    suffix = IMAGE_SUFFIX_BY_MACHINE_TYPE[normalized]
    if not suffix:
        return base_tag
    if base_tag.endswith("-openeuler"):
        stem = base_tag[: -len("-openeuler")]
        if stem.endswith(suffix):
            return base_tag
        return f"{stem}{suffix}-openeuler"
    if base_tag.endswith(suffix):
        return base_tag
    return f"{base_tag}{suffix}"


def image_candidates_for_tag(
    tag: str, machine_type: str | None = None
) -> tuple[str, ...]:
    resolved_tag = image_tag_for_machine(tag, machine_type)
    return tuple(f"{repo}:{resolved_tag}" for repo in IMAGE_REGISTRY_MIRRORS)


def docker_ref_tag(ref: str) -> str | None:
    if "@" in ref:
        return None
    last_slash = ref.rfind("/")
    last_colon = ref.rfind(":")
    if last_colon > last_slash:
        return ref[last_colon + 1 :]
    return None


def docker_ref_repo(ref: str) -> str:
    without_digest = ref.split("@", 1)[0]
    last_slash = without_digest.rfind("/")
    last_colon = without_digest.rfind(":")
    if last_colon > last_slash:
        return without_digest[:last_colon]
    return without_digest


def infer_machine_type_from_image(ref: str | None) -> str | None:
    if not ref:
        return None
    tag = docker_ref_tag(ref)
    if tag is None:
        return None
    lowered = tag.lower()
    if any(marker in lowered for marker in ("-a5", "-950dt", "-950pr")):
        return "A5"
    if "-310p" in lowered:
        return "310P"
    if "-a3" in lowered:
        return "A3"
    if docker_ref_repo(ref).endswith("ascend/vllm-ascend"):
        return "A2"
    return None


def validate_explicit_image_for_machine(ref: str, machine_type: str | None) -> str:
    candidate = require_explicit_image_ref(ref)
    normalized_machine_type = normalize_machine_type(machine_type)
    if normalized_machine_type is None:
        return candidate
    repo = docker_ref_repo(candidate)
    if not repo.endswith("ascend/vllm-ascend"):
        return candidate
    inferred = infer_machine_type_from_image(candidate)
    if inferred is not None and inferred != normalized_machine_type:
        raise MachineManagementError(
            f"image {candidate!r} appears to target {inferred}, but the machine type is {normalized_machine_type}"
        )
    return candidate


def require_explicit_image_ref(ref: str) -> str:
    candidate = ref.strip()
    if not candidate:
        raise MachineManagementError(
            "image reference is empty; choose `local-latest`, `rc`, `main`, `stable`, or an explicit non-latest image reference"
        )
    normalized = candidate.lower()
    if normalized in LEGACY_IMAGE_SELECTORS:
        raise MachineManagementError(
            "legacy image selector `auto` is no longer allowed; ask the user to choose `local-latest`, `rc`, `main`, `stable`, or a concrete image reference"
        )
    if normalized in IMAGE_SELECTOR_ALIASES:
        raise MachineManagementError(
            "semantic image selectors must be handled directly, not mixed into a custom candidate list"
        )
    if "@" in candidate:
        return candidate
    tag = docker_ref_tag(candidate)
    if tag is None:
        raise MachineManagementError(
            "explicit image references must include a concrete tag or digest; bare repositories implicitly resolve to `latest` and are not allowed"
        )
    if tag.lower() in FORBIDDEN_IMAGE_TAGS:
        raise MachineManagementError(
            "the moving `latest` tag is not allowed as a direct image choice; choose `local-latest`, `rc`, `main`, `stable`, or a concrete version tag"
        )
    return candidate


def parse_release_tag(tag: str) -> tuple[int, int, int, int | None] | None:
    match = SEMVER_TAG_PATTERN.fullmatch(tag.strip())
    if match is None:
        return None
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))
    kind = (match.group("kind") or "").lower()
    kind_number = match.group("kind_number")
    if kind == "rc":
        return major, minor, patch, int(kind_number or "0")
    return major, minor, patch, None


def choose_latest_tag(tags: Sequence[str], *, prerelease: bool) -> str | None:
    parsed_tags: list[tuple[tuple[int, int, int, int], str]] = []
    for tag in tags:
        parsed = parse_release_tag(tag)
        if parsed is None:
            continue
        major, minor, patch, rc_number = parsed
        is_prerelease = rc_number is not None
        if prerelease != is_prerelease:
            continue
        sort_key = (major, minor, patch, rc_number or 0)
        parsed_tags.append((sort_key, tag.strip()))
    if not parsed_tags:
        return None
    parsed_tags.sort(key=lambda item: item[0], reverse=True)
    return parsed_tags[0][1]


def fetch_release_catalog(
    timeout_seconds: int = LATEST_RELEASE_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": IMAGE_RESOLVER_USER_AGENT,
    }
    request = urllib.request.Request(RELEASES_API, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise MachineManagementError(
            "GitHub releases API returned an unexpected payload"
        )
    releases = [item for item in payload if isinstance(item, dict)]
    if not releases:
        raise MachineManagementError("GitHub releases API returned no releases")
    return releases


def choose_release_tag_from_catalog(
    catalog: Sequence[dict[str, Any]], *, prerelease: bool
) -> str | None:
    tags: list[str] = []
    for release in catalog:
        if release.get("draft"):
            continue
        tag = str(release.get("tag_name") or "").strip()
        if not tag:
            continue
        parsed = parse_release_tag(tag)
        if parsed is None:
            continue
        is_prerelease = parsed[3] is not None
        if prerelease:
            if release.get("prerelease") or is_prerelease:
                tags.append(tag)
        else:
            if not release.get("prerelease") and not is_prerelease:
                tags.append(tag)
    return choose_latest_tag(tags, prerelease=prerelease)


def choose_release_tag_from_page(html: str, *, prerelease: bool) -> str | None:
    matches = re.findall(r'/vllm-project/vllm-ascend/releases/tag/([^"?#]+)', html)
    tags = [urllib.parse.unquote(item).strip() for item in matches]
    return choose_latest_tag(tags, prerelease=prerelease)


def fetch_latest_release_tag(
    timeout_seconds: int = LATEST_RELEASE_TIMEOUT_SECONDS,
    *,
    _catalog_fetcher: Callable[..., list[dict[str, Any]]] | None = None,
) -> str:
    errors: list[str] = []
    try:
        fetch_catalog = _catalog_fetcher or fetch_release_catalog
        catalog = fetch_catalog(timeout_seconds=timeout_seconds)
        tag = choose_release_tag_from_catalog(catalog, prerelease=False)
        if tag:
            return tag
        errors.append("GitHub releases catalog did not contain a final release tag")
    except TimeoutError:
        raise
    except (
        MachineManagementError,
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
    ) as exc:
        errors.append(f"GitHub API: {exc}")

    page_request = urllib.request.Request(
        LATEST_RELEASE_PAGE, headers={"User-Agent": IMAGE_RESOLVER_USER_AGENT}
    )
    try:
        with urllib.request.urlopen(page_request, timeout=timeout_seconds) as response:
            final_url = response.geturl()
        match = re.search(r"/tag/([^/?#]+)$", final_url)
        if match:
            tag = urllib.parse.unquote(match.group(1))
            if choose_latest_tag([tag], prerelease=False):
                return tag
        errors.append("release redirect did not expose a tag name")
    except TimeoutError:
        raise
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        errors.append(f"GitHub releases page: {exc}")

    page_request = urllib.request.Request(
        RELEASES_PAGE, headers={"User-Agent": IMAGE_RESOLVER_USER_AGENT}
    )
    try:
        with urllib.request.urlopen(page_request, timeout=timeout_seconds) as response:
            html = response.read().decode("utf-8", errors="replace")
        tag = choose_release_tag_from_page(html, prerelease=False)
        if tag:
            return tag
        errors.append("releases page did not expose a final release tag")
    except TimeoutError:
        raise
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        errors.append(f"GitHub releases listing: {exc}")

    detail = "; ".join(errors)
    raise MachineManagementError(
        "could not resolve the latest official vllm-ascend release tag; retry later, choose `rc`, `main`, or pass an explicit image reference"
        + (f" ({detail})" if detail else "")
    )


def fetch_latest_prerelease_tag(
    timeout_seconds: int = LATEST_RELEASE_TIMEOUT_SECONDS,
    *,
    _catalog_fetcher: Callable[..., list[dict[str, Any]]] | None = None,
) -> str:
    errors: list[str] = []
    try:
        fetch_catalog = _catalog_fetcher or fetch_release_catalog
        catalog = fetch_catalog(timeout_seconds=timeout_seconds)
        tag = choose_release_tag_from_catalog(catalog, prerelease=True)
        if tag:
            return tag
        errors.append("GitHub releases catalog did not contain a prerelease tag")
    except TimeoutError:
        raise
    except (
        MachineManagementError,
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
    ) as exc:
        errors.append(f"GitHub API: {exc}")

    page_request = urllib.request.Request(
        RELEASES_PAGE, headers={"User-Agent": IMAGE_RESOLVER_USER_AGENT}
    )
    try:
        with urllib.request.urlopen(page_request, timeout=timeout_seconds) as response:
            html = response.read().decode("utf-8", errors="replace")
        tag = choose_release_tag_from_page(html, prerelease=True)
        if tag:
            return tag
        errors.append("releases page did not expose a prerelease tag")
    except TimeoutError:
        raise
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        errors.append(f"GitHub releases listing: {exc}")

    detail = "; ".join(errors)
    raise MachineManagementError(
        "could not resolve the latest official vllm-ascend prerelease tag; retry later, choose `main`, `stable`, or pass an explicit image reference"
        + (f" ({detail})" if detail else "")
    )


def resolve_image_request(
    spec: str,
    *,
    machine_type: str | None = None,
    _catalog_fetcher: Callable[..., list[dict[str, Any]]] | None = None,
) -> ImageResolution:
    requested = spec.strip()
    if not requested:
        raise MachineManagementError(
            "image selector is missing; choose `local-latest`, `rc`, `main`, `stable`, or an explicit non-latest image reference"
        )
    normalized_machine_type = normalize_machine_type(machine_type)
    normalized = requested.lower()
    selector = IMAGE_SELECTOR_ALIASES.get(normalized)
    if selector == IMAGE_SELECTOR_RC:
        release_tag = fetch_latest_prerelease_tag(_catalog_fetcher=_catalog_fetcher)
        return ImageResolution(
            selector=IMAGE_SELECTOR_RC,
            requested=requested,
            policy="latest-official-prerelease",
            candidates=image_candidates_for_tag(release_tag, normalized_machine_type),
            resolved_tag=image_tag_for_machine(release_tag, normalized_machine_type),
            machine_type=normalized_machine_type,
        )
    if selector == IMAGE_SELECTOR_MAIN:
        return ImageResolution(
            selector=IMAGE_SELECTOR_MAIN,
            requested=requested,
            policy="main-branch",
            candidates=image_candidates_for_tag(
                IMAGE_SELECTOR_MAIN, normalized_machine_type
            ),
            resolved_tag=image_tag_for_machine(
                IMAGE_SELECTOR_MAIN, normalized_machine_type
            ),
            machine_type=normalized_machine_type,
        )
    if selector == IMAGE_SELECTOR_STABLE:
        release_tag = fetch_latest_release_tag(_catalog_fetcher=_catalog_fetcher)
        return ImageResolution(
            selector=IMAGE_SELECTOR_STABLE,
            requested=requested,
            policy="latest-official-release",
            candidates=image_candidates_for_tag(release_tag, normalized_machine_type),
            resolved_tag=image_tag_for_machine(release_tag, normalized_machine_type),
            machine_type=normalized_machine_type,
        )
    if selector == IMAGE_SELECTOR_LOCAL_LATEST:
        return ImageResolution(
            selector=IMAGE_SELECTOR_LOCAL_LATEST,
            requested=requested,
            policy=IMAGE_POLICY_LOCAL_LATEST,
            candidates=(),
            mirror_order=(),
            machine_type=normalized_machine_type,
        )

    candidates = tuple(
        validate_explicit_image_for_machine(item, normalized_machine_type)
        for item in requested.split(",")
        if item.strip()
    )
    if not candidates:
        raise MachineManagementError(
            "image selector is empty; choose `local-latest`, `rc`, `main`, `stable`, or an explicit non-latest image reference"
        )
    return ImageResolution(
        selector="explicit",
        requested=requested,
        policy="explicit",
        candidates=candidates,
        machine_type=normalized_machine_type,
    )


def image_request_payload(
    spec: str,
    *,
    machine_type: str | None = None,
    _catalog_fetcher: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    resolution = resolve_image_request(
        spec, machine_type=machine_type, _catalog_fetcher=_catalog_fetcher
    )
    return {
        "requested": resolution.requested,
        "selector": resolution.selector,
        "policy": resolution.policy,
        "candidates": list(resolution.candidates),
        "resolved_tag": resolution.resolved_tag,
        "mirror_order": list(resolution.mirror_order),
        "machine_type": resolution.machine_type,
    }


def run_local(
    cmd: Sequence[str],
    *,
    input_text: str | None = None,
    check: bool = False,
    env: dict[str, str] | None = None,
    stdin_source: int | None = None,
    timeout_seconds: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            list(cmd),
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            stdin=stdin_source,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise MachineManagementError(
            f"required local command not found: {cmd[0]}"
        ) from exc
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise MachineManagementError(detail)
    return proc


def shell_join(cmd: Sequence[str]) -> str:
    items = list(cmd)
    if os.name == "nt":
        return subprocess.list2cmdline(items)
    return shlex.join(items)
