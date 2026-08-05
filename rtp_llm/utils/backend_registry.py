"""Deferred registration of out-of-tree model backends.

``rtp_llm/__init__.py`` imports optional backend packages long before the
factory modules under this package are imported, so a backend cannot call
``LinearFactory.register()`` or ``StrategyRegistry.register()`` at its own
import time -- the factory does not exist yet. Importing the factories early
from the backend is not an option either: eager imports of the MoE/attention
factories pull in communication libraries and have broken server startup for
configurations that never use them.

Backends therefore record their intent here, and each factory drains the hooks
for its own slot once it has finished building its registry:

    # backend, at import time
    register_backend_hook("linear", lambda: LinearFactory.register(MyLinear))

    # factory, at the end of its __init__
    run_backend_registrations("linear")

This lives under ``rtp_llm.utils`` rather than next to the factories on
purpose: the server argument parser also drains a slot, and importing anything
under ``models_py.modules.factory`` would execute that package's ``__init__``,
which builds every factory. That is the eager import this mechanism exists to
avoid.

Hook exceptions are intentionally not swallowed. A backend that registered a
hook needs it: silently dropping the registration leaves the factory selecting
a different implementation, which shows up as wrong numerics rather than a
startup failure.
"""

import logging
from typing import Any, Callable, Dict, List, Set

logger = logging.getLogger(__name__)

BackendHook = Callable[..., None]

_hooks: Dict[str, List[BackendHook]] = {}
_drained: Set[str] = set()


def register_backend_hook(slot: str, hook: BackendHook) -> None:
    """Record ``hook`` to run once the ``slot`` factory is initialised.

    Raises if the slot was already drained, since the hook would never run.
    """
    if slot in _drained:
        raise RuntimeError(
            f"backend slot {slot!r} was already initialised; register the hook "
            "before the factory is imported"
        )
    _hooks.setdefault(slot, []).append(hook)


def run_backend_registrations(slot: str, **context: Any) -> None:
    """Run the hooks recorded for ``slot``, passing ``context`` to each.

    Idempotent: repeated calls for the same slot are no-ops, so a factory
    re-imported under a different alias does not double-register.
    """
    if slot in _drained:
        return
    _drained.add(slot)
    hooks = _hooks.get(slot, ())
    for hook in hooks:
        logger.debug("running backend registration hook for slot %r", slot)
        hook(**context)


def reset_backend_registrations() -> None:
    """Drop all recorded hooks and drained slots. For tests only."""
    _hooks.clear()
    _drained.clear()
