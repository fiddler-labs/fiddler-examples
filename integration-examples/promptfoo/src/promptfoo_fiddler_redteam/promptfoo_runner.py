"""Thin subprocess wrapper around `promptfoo redteam generate`.

We only ever call `generate`, never `eval`: this tool supplies its own
grading (Fiddler Guardrails) instead of promptfoo's. `generate` does not
contact any target LLM and does not consume red-team cloud probe credits;
it just synthesizes adversarial test cases.

promptfoo's output streams straight to your terminal, so you can see its
progress and answer its one-time email prompt on first use. For unattended
runs (cron, CI), set the email first so promptfoo never prompts:

    npx promptfoo@<PROMPTFOO_VERSION> config set email you@company.com

When stdin isn't a terminal, promptfoo gets no stdin at all, and every run
has a timeout, so a scheduled job fails instead of hanging.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# The promptfoo version this example was tested with. The parser reads
# promptfoo's unversioned output format, so pin rather than run @latest.
PROMPTFOO_VERSION = "0.124.1"


class PromptfooNotFoundError(RuntimeError):
    pass


class PromptfooGenerateError(RuntimeError):
    pass


def _promptfoo_command() -> list[str]:
    if shutil.which("npx"):
        return ["npx", "--yes", f"promptfoo@{PROMPTFOO_VERSION}"]
    if shutil.which("promptfoo"):
        return ["promptfoo"]
    raise PromptfooNotFoundError(
        "Neither `npx` nor `promptfoo` was found on PATH. Install Node.js (for `npx`) "
        f"or promptfoo itself (`npm install -g promptfoo@{PROMPTFOO_VERSION}`)."
    )


def generate(
    config_path: str | Path,
    output_path: str | Path,
    plugins: list[str] | None = None,
    strategies: list[str] | None = None,
    num_tests: int | None = None,
    force: bool = True,
    timeout: float = 1800,
) -> Path:
    """Run `promptfoo redteam generate` and return the path to the generated YAML file."""
    cmd = _promptfoo_command() + [
        "redteam",
        "generate",
        "--config",
        str(config_path),
        "-o",
        str(output_path),
    ]
    if plugins:
        cmd += ["--plugins", ",".join(plugins)]
    if strategies:
        cmd += ["--strategies", ",".join(strategies)]
    if num_tests is not None:
        cmd += ["-n", str(num_tests)]
    if force:
        cmd.append("--force")

    interactive = sys.stdin is not None and sys.stdin.isatty()
    try:
        result = subprocess.run(
            cmd,
            check=False,
            stdin=None if interactive else subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise PromptfooGenerateError(
            f"`{' '.join(cmd)}` timed out after {timeout:.0f}s. If this is the first run on this "
            "machine, promptfoo may be waiting for its one-time email prompt; run "
            f"`npx promptfoo@{PROMPTFOO_VERSION} config set email you@company.com` first."
        ) from e
    if result.returncode != 0:
        raise PromptfooGenerateError(
            f"`{' '.join(cmd)}` failed (exit {result.returncode}). See promptfoo's output above."
        )

    out_path = Path(output_path)
    if not out_path.exists():
        raise PromptfooGenerateError(
            f"promptfoo reported success but {out_path} was not created."
        )
    return out_path
