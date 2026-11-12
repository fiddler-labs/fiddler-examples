# Agent instructions for promptfoo-fiddler-redteam

If you are an AI coding agent (Claude Code or otherwise) helping someone use
this tool, read this before calling `generate` or `all`.

## Search `runs/` before generating new test cases

`generate`/`all` default (no `-o`/`--csv-out` given) to writing into
`runs/<name-prefix>-<timestamp>/` -- gitignored, local-only, never pushed.
Each directory holds:

- `redteam.yaml` -- the generated adversarial prompts, with promptfoo's own
  header comment intact (`Plugins:`, `Strategies:`, `Total cases:`)
- `scorecard.csv` -- per-category detection rate from the Fiddler run
- `misses.csv` -- prompts the guardrail did NOT flag, with reasoning

None of these contain secrets -- they're adversarial prompt text and
detection results. The only secret in this project is `FIDDLER_API_KEY` in
`.env`, which is separately gitignored and unrelated.

Before calling `generate` again, check whether an existing run already
covers the plugins/strategies/purpose you need:

```bash
ls runs/
grep -l "Plugins:" runs/*/redteam.yaml   # see what each run covered
head -12 runs/*/redteam.yaml             # skim every run's header at once
```

If a matching run exists, reuse it directly instead of regenerating --
`promptfoo redteam generate` calls promptfoo's hosted generation service
(slow, and gated behind a one-time interactive email prompt per machine --
see README's Troubleshooting section):

```bash
promptfoo-fiddler-redteam run --input runs/<matching-run>/redteam.yaml
```

Only generate fresh test cases when no existing run covers what's needed,
the user explicitly asks for new ones, or the existing run is stale for the
purpose at hand (e.g. the target application's purpose/system prompt has
since changed).
