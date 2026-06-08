"""Hermes plugin entry point for pip-based auto-discovery.

When installed via pip, Hermes discovers this module through the
``hermes.plugins`` entry point group and calls ``register(ctx)``.

The plugin context provides:
- ctx.register_memory_provider(provider_class)
- ctx.home_dir — the active HERMES_HOME path
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Register the LocalMemoryStack provider with Hermes.

    Called by Hermes plugin loader during startup. This is the only
    function Hermes needs — the plugin system handles the rest.
    """
    from .hermes_memory_provider import LocalMemoryStackProvider

    ctx.register_memory_provider(LocalMemoryStackProvider)
    logger.info("local-memory-stack: registered memory provider")


def post_setup(hermes_home: str, config: dict) -> None:
    """Called after 'hermes memory setup' selects this provider.

    Since local-memory-stack has zero credentials, we just create the
    data directory and print a success message.
    """
    import os

    data_dir = os.path.join(hermes_home, "memory", "local-memory-stack")
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n✅ local-memory-stack is ready!")
    print(f"   Data directory: {data_dir}")
    print(f"   No API keys needed — fully local.")
    print(f"   Add to ~/.hermes/config.yaml:")
    print(f"     memory:")
    print(f"       provider: local-memory-stack")
    print()
