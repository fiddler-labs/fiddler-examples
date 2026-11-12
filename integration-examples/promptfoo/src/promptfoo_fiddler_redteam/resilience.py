"""Timeouts, a single retry for dropped connections, and readable Fiddler errors.

Scheduled runs (cron, CI) hit the network in ways a laptop run rarely does:
a connection that's dropped mid-request, or a server that never answers.
Without explicit timeouts, one stuck call can hold a job for the whole CI
time limit, so every Fiddler call in this tool gets a connect/read timeout
and one retry on connection-level failures. HTTP errors such as 401 or 500
are not retried here: a bad API key won't fix itself.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")

# (connect, read) timeouts in seconds for every Fiddler call. Pass these to
# fiddler_evals.init() explicitly: without them the SDK waits indefinitely.
FIDDLER_TIMEOUT: tuple[float, float] = (10.0, 60.0)
RETRY_DELAY_SECONDS = 3.0
TRANSIENT_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)

# requests formats these as e.g. "401 Client Error: Unauthorized for url: ..."
_AUTH_MARKERS = ("Unauthorized", "Forbidden")
_CONNECTION_MARKERS = ("Failed to connect", "Connection", "timed out", "Timeout")


def with_retry(fn: Callable[[], T], attempts: int = 2) -> T:
    """Call fn(), retrying on connection errors and timeouts only."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except TRANSIENT_ERRORS as e:
            if attempt == attempts:
                raise
            logger.warning(
                "Fiddler call failed (%s); retrying in %.0fs", e, RETRY_DELAY_SECONDS
            )
            time.sleep(RETRY_DELAY_SECONDS)
    raise AssertionError("unreachable")


def describe_fiddler_error(error: BaseException, url: str) -> str:
    """A one-line, actionable explanation of a failed Fiddler call."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    text = f"{type(error).__name__}: {error}"
    if status in (401, 403) or any(m in text for m in _AUTH_MARKERS):
        return f"Fiddler rejected the API key. Check FIDDLER_API_KEY. ({text})"
    if isinstance(error, TRANSIENT_ERRORS) or any(
        m in text for m in _CONNECTION_MARKERS
    ):
        return (
            f"Couldn't reach Fiddler at {url}, or it stopped responding. Check "
            f"FIDDLER_URL and that the instance is up. ({text})"
        )
    return f"Fiddler call failed. ({text})"


def error_hint(message: str) -> str:
    """What to do about a per-prompt scoring error, based on its message."""
    if any(m in message for m in _AUTH_MARKERS):
        return "Fiddler rejected the API key; check FIDDLER_API_KEY."
    if any(m in message for m in _CONNECTION_MARKERS):
        return (
            "Fiddler didn't respond or dropped the connection, even after a retry. "
            "This is usually temporary: rerun, and if it keeps happening, check that "
            "your Fiddler instance is healthy."
        )
    return "Check the logs for details."
