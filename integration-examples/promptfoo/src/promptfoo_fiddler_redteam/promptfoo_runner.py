"""Thin subprocess wrapper around `promptfoo redteam generate`.

We only ever call `generate`, never `eval` -- this tool supplies its own
grading (Fiddler Guardrails) instead of promptfoo's. `generate` does not
contact any target LLM and does not consume redteam cloud probe credits,
it just synthesizes adversarial test cases.

Note: promptfoo's hosted generation service requires a one-time work-email
prompt the first time it's used from a given machine/cache. Run the
`generate` subcommand interactively at least once (or pre-seed promptfoo's
cache) before wiring this into an unattended daily/weekly cron job.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class PromptfooNotFoundError(RuntimeError):
    pass


class PromptfooGenerateError(RuntimeError):
    pass


def _promptfoo_command() -> list[str]:
    if shutil.which("promptfoo"):
        return ["promptfoo"]
    if shutil.which("npx"):
        return ["npx", "--yes", "promptfoo@latest"]
    raise PromptfooNotFoundError(
        "Neither `promptfoo` nor `npx` was found on PATH. Install promptfoo "
        "(`npm install -g promptfoo`) or Node.js (for `npx`)."
    )


def generate(
    config_path: str | Path,
    output_path: str | Path,
    plugins: list[str] | None = None,
    strategies: list[str] | None = None,
    num_tests: int | None = None,
    force: bool = True,
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

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PromptfooGenerateError(
            f"`{' '.join(cmd)}` failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
        )

    out_path = Path(output_path)
    if not out_path.exists():
        raise PromptfooGenerateError(
            f"promptfoo reported success but {out_path} was not created.\n{result.stdout}"
        )
    return out_path
