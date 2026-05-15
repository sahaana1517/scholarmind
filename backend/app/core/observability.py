"""
Langfuse observability helpers.

Provides a singleton Langfuse client and graceful no-op behavior
if credentials are missing (so the app keeps working in dev).
"""

from typing import Optional

from langfuse import Langfuse

from backend.app.core.config import settings


_client: Optional[Langfuse] = None
_enabled: Optional[bool] = None


def get_langfuse() -> Optional[Langfuse]:
    """
    Returns a Langfuse client if credentials are configured, else None.

    Callers should check for None and skip tracing if disabled
    (graceful degradation for local development).
    """
    global _client, _enabled

    if _enabled is False:
        return None

    if _client is not None:
        return _client

    # Check credentials
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        _enabled = False
        print("ℹ️  Langfuse credentials not set — tracing disabled")
        return None

    try:
        _client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST or "https://cloud.langfuse.com",
        )
        _enabled = True
        return _client
    except Exception as e:
        print(f"⚠️  Langfuse init failed: {e}")
        _enabled = False
        return None


def flush() -> None:
    """Force-send pending traces (call before script exits)."""
    client = get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception as e:
            print(f"⚠️  Langfuse flush failed: {e}")