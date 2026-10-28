"""Renders the managed container bootstrap host script.

[Role]
Concatenates the two literal template halves and substitutes the documented
placeholders to produce the bash/Python script that creates, reuses, and
configures a managed Ascend container on a host.

[Invariants]
- Public-key installation uses the same remote-dev primitive as host onboarding.
- Template placeholders are substituted literally without evaluating shell input.

[Timeouts & Locks]
- Rendering is pure; the outer bootstrap timeout is owned by `host_ops`
  (1800s).

[Side-Effect Policy]
- No side effects at import or render time.
"""

from __future__ import annotations

from remote_dev.auth import INSTALL_PUBLIC_KEY_SH

from mindie_coordinator.provision._host_bootstrap_template_a import (
    _BOOTSTRAP_TEMPLATE_HEAD,
)
from mindie_coordinator.provision._host_bootstrap_template_b import (
    _BOOTSTRAP_TEMPLATE_TAIL,
)
from mindie_coordinator.provision.image_resolver import (
    IMAGE_POLICY_LOCAL_LATEST,
    LOCAL_IMAGE_DISCOVERY_PY,
    PROGRESS_SENTINEL,
    SENTINEL,
)

_BOOTSTRAP_TEMPLATE = _BOOTSTRAP_TEMPLATE_HEAD + _BOOTSTRAP_TEMPLATE_TAIL


def render_bootstrap_host_script() -> str:
    return (
        _BOOTSTRAP_TEMPLATE.replace("__SENTINEL__", SENTINEL)
        .replace("__PROGRESS__", PROGRESS_SENTINEL)
        .replace("__LOCAL_IMAGE_DISCOVERY_PY__", LOCAL_IMAGE_DISCOVERY_PY)
        .replace("__LOCAL_IMAGE_DISCOVERY_POLICY__", IMAGE_POLICY_LOCAL_LATEST)
        .replace("__INSTALL_PUBLIC_KEY_SH__", INSTALL_PUBLIC_KEY_SH)
    )
