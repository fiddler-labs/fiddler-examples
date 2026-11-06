"""CLI: generate adversarial prompts with promptfoo, run them through Fiddler
Guardrails as a Fiddler Dataset + Experiment, print a detection scorecard.

Unless overridden with -o/--output or --csv-out, each run is archived under
`runs/<name-prefix>-<timestamp>/` (gitignored) so the generated prompts and
scorecard/misses from a run stay together on disk and can be grepped before
generating a new, possibly-duplicate set. See AGENTS.md / README.md."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import ConfigError, FiddlerSettings
from .experiment_runner import run_experiment
from .fiddler_bridge import build_dataset, connect
from .guardrails_client import GuardrailsClient
from .promptfoo_parser import load_redteam_cases
from .promptfoo_runner import PromptfooGenerateError, PromptfooNotFoundError, generate
from .report import (
    build_scorecard,
    print_scorecard,
    write_misses_csv,
    write_scorecard_csv,
)


def _default_run_dir(name_prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{name_prefix}-{timestamp}"


def _add_generate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config", required=True, help="promptfooconfig.yaml with a redteam: section"
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Where to write generated test cases (default: runs/<prefix>-<timestamp>/redteam.yaml)",
    )
    p.add_argument(
        "--plugins", help="Comma-separated plugin ids, overrides the config file"
    )
    p.add_argument(
        "--strategies", help="Comma-separated strategy ids, overrides the config file"
    )
    p.add_argument("-n", "--num-tests", type=int, help="Test cases per plugin")
    p.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Seconds to wait for promptfoo generation before giving up (default: 1800)",
    )


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--input", required=True, help="promptfoo redteam generate output file"
    )
    p.add_argument(
        "--name-prefix",
        default="promptfoo-redteam",
        help="Dataset/Experiment name prefix",
    )
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument(
        "--csv-out",
        help="Directory to write scorecard.csv / misses.csv into (default: runs/<prefix>-<timestamp>/)",
    )


def cmd_generate(args: argparse.Namespace) -> int:
    output = args.output
    if output is None:
        run_dir = _default_run_dir(getattr(args, "name_prefix", None) or "generate")
        run_dir.mkdir(parents=True, exist_ok=True)
        output = run_dir / "redteam.yaml"

    try:
        plugins = args.plugins.split(",") if args.plugins else None
        strategies = args.strategies.split(",") if args.strategies else None
        out_path = generate(
            config_path=args.config,
            output_path=output,
            plugins=plugins,
            strategies=strategies,
            num_tests=args.num_tests,
            timeout=args.timeout,
        )
    except (PromptfooNotFoundError, PromptfooGenerateError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote generated test cases to {out_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        settings = FiddlerSettings.from_env()
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    cases = load_redteam_cases(args.input)
    if not cases:
        print(f"error: no test cases parsed from {args.input}", file=sys.stderr)
        return 1
    print(f"Parsed {len(cases)} test case(s) from {args.input}")

    application = connect(settings)
    dataset = build_dataset(application, cases, name_prefix=args.name_prefix)
    print(f"Created Fiddler dataset: {dataset.name}")

    client = GuardrailsClient(base_url=settings.url, api_key=settings.api_key)
    experiment_result = run_experiment(
        dataset, client, name_prefix=args.name_prefix, max_workers=args.max_workers
    )

    scorecard = build_scorecard(experiment_result)
    print_scorecard(scorecard)

    # Write the scorecard/misses next to the generated redteam.yaml (which
    # `generate`/`all` already default into runs/<prefix>-<timestamp>/) so
    # the prompts and results for a run stay together on disk and a future
    # `generate` call -- by a person or an AI agent -- can grep this directory
    # instead of re-synthesizing an equivalent set. See AGENTS.md.
    out_dir = Path(args.csv_out) if args.csv_out else Path(args.input).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    write_scorecard_csv(scorecard.rows, out_dir / "scorecard.csv")
    write_scorecard_csv(
        scorecard.rows_by_plugin,
        out_dir / "scorecard_by_plugin.csv",
        key_name="plugin_id",
    )
    write_misses_csv(scorecard.misses, out_dir / "misses.csv")
    print(f"\nWrote scorecard.csv, scorecard_by_plugin.csv and misses.csv to {out_dir}")

    # Non-zero exit when some prompts couldn't be scored, so a scheduled run
    # surfaces a config/API problem instead of reporting a misleading rate.
    return 2 if scorecard.errors or scorecard.unscored else 0


def cmd_all(args: argparse.Namespace) -> int:
    output = args.output
    csv_out = args.csv_out
    if output is None and csv_out is None:
        # Share a single runs/ directory between the generate and run steps.
        run_dir = _default_run_dir(args.name_prefix)
        run_dir.mkdir(parents=True, exist_ok=True)
        output = str(run_dir / "redteam.yaml")
        csv_out = str(run_dir)
    elif output is None:
        output = "redteam.yaml"

    gen_args = argparse.Namespace(
        config=args.config,
        output=output,
        plugins=args.plugins,
        strategies=args.strategies,
        num_tests=args.num_tests,
        timeout=args.timeout,
    )
    rc = cmd_generate(gen_args)
    if rc != 0:
        return rc

    run_args = argparse.Namespace(
        input=output,
        name_prefix=args.name_prefix,
        max_workers=args.max_workers,
        csv_out=csv_out,
    )
    return cmd_run(run_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="promptfoo-fiddler-redteam")
    sub = parser.add_subparsers(dest="command", required=True)

    p_generate = sub.add_parser(
        "generate", help="Generate adversarial prompts with promptfoo"
    )
    _add_generate_args(p_generate)
    p_generate.set_defaults(func=cmd_generate)

    p_run = sub.add_parser(
        "run", help="Push prompts into Fiddler and score them against Guardrails"
    )
    _add_run_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_all = sub.add_parser("all", help="generate + run in one shot")
    _add_generate_args(p_all)
    p_all.add_argument("--name-prefix", default="promptfoo-redteam")
    p_all.add_argument("--max-workers", type=int, default=4)
    p_all.add_argument("--csv-out")
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
