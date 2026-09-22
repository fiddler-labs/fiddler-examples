# jev-evals

> Part of [fiddler-examples](https://github.com/fiddler-labs/fiddler-examples). Everything
> below is run from this folder, `experiments/jev-evals/`. It is a self-contained Node project.

Do three AI Gateway models agree on how risky a coding agent's shell commands are?

**Explore the results:** [jev-evals.vercel.app](https://jev-evals.vercel.app) is the
read-only drill-down UI over the committed artifacts. Nothing there is computed live.

This experiment lets the **real GitHub Copilot CLI** work on fabricated tasks inside a
disposable **Vercel Sandbox**, captures every Bash tool call it genuinely chose to make,
and then has **three models classify each span against one identical rubric**:

| evaluator | model | returns |
|---|---|---|
| `jev` | `typesafe-ai/jev` | `risk_level` only (classifier; no rationale requested) |
| `gemini` | `google/gemini-3.5-flash-lite` | `risk_level` + `rationale` (`thinkingLevel: minimal`) |
| `openai` | `openai/gpt-5.4-nano` | `risk_level` + `rationale` (`reasoningEffort: none`) |

`risk_level` is one of `low`, `medium`, `high`, `critical`.

## The live pipeline, exactly

For **each** of the ~12 fabricated scenarios:

1. **Provision** a fresh disposable Vercel Sandbox (`Sandbox.create`).
2. **Seed** a fabricated workspace: fake source files, a fake `package.json`, a
   credential-shaped `.env.fabricated` containing only fake values.
3. **Install** the real agent inside the sandbox: `npm install -g @github/copilot@1.0.86`.
4. **Configure BYOK routing** so the CLI talks to Vercel AI Gateway instead of GitHub's
   model routing. `COPILOT_PROVIDER_BASE_URL` activates BYOK and, per
   `copilot help providers`, **GitHub authentication is not required**. No `GH_TOKEN`,
   `GITHUB_TOKEN` or `COPILOT_GITHUB_TOKEN` is ever set.
5. **Hand Copilot the scenario prompt** (`copilot -p … --allow-all-tools --no-ask-user
   --output-format json`). **Copilot — not this harness — decides and executes every
   command.** The span count per scenario is whatever the agent actually did.
6. **Capture** the agent's own JSONL event stream plus its OTel file export, writing both
   to `/vercel/sandbox/artifacts` inside the sandbox.
7. **Retrieve artifacts before teardown** with `readFileToBuffer`, in a `finally` block so
   a crash mid-scenario still preserves the evidence. Only then `sandbox.stop()`.
8. **Normalize** each `bash` tool call into one span: command, the agent's raw tool input,
   capped and redacted output, exit code, error type, timestamps and duration.
9. **Evaluate every span** with all three models, using byte-identical shared rubric text
   and byte-identical span evidence.
10. **Preserve** raw evaluator responses alongside normalized results, each stamped with
    the rubric hash and a hash of the exact submitted input.

Total evaluation calls = **3 × number of captured Bash spans** (not a fixed 30).

> **Every artifact in this repo is a live measurement.** There is no simulated dataset: the
> only evaluator stand-in is a deterministic stub under `test/helpers/`, reachable from the
> test suite and from nothing in `src/`.

## Setup

```bash
npm install
cp .env.example .env     # then fill in AI_GATEWAY_API_KEY
```

One Gateway credential powers both halves: the Copilot agent (BYOK) and the three
evaluators. **This project never creates a Gateway key for you** — create it yourself in
the Vercel dashboard under AI Gateway → API keys, or with
`vercel ai-gateway api-keys create`.

**Tier note:** the two peer evaluators need paid AI Gateway credits on the team that owns
the key. Without them the Gateway returns 403 ("Free tier users do not have access to this
model") for current-generation models and caps the team at 5 requests per minute. Jev is an
evaluation model and is only callable through the evaluation API, not `/chat/completions`.

### Vercel Sandbox setup

Only `npm run trace:live` touches Sandbox; evaluation, summary, report and UI never do.
Each scenario provisions one sandbox (`node22` runtime, 10 minute cap) on the Vercel
team you authenticate as, and that team is billed for it. Check plan availability, pricing
and limits at [vercel.com/docs/vercel-sandbox](https://vercel.com/docs/vercel-sandbox).

The `@vercel/sandbox` SDK accepts credentials three ways. Pick one:

**A. OIDC, the simplest on a laptop.**

```bash
npm i -g vercel
vercel login
vercel link          # pick or create a project; sandboxes bill to its team
vercel env pull      # writes VERCEL_OIDC_TOKEN to .env.local
```

`src/config.js` loads `.env.local`, so nothing else is needed. The token expires after
roughly 12 hours. When `trace:live` fails with "Could not get credentials from OIDC
context", run `vercel env pull` again.

**B. Access token, for CI or anything non-interactive.** Set all three of `VERCEL_TOKEN`,
`VERCEL_TEAM_ID` and `VERCEL_PROJECT_ID` in `.env`. Create the token at
vercel.com/account/tokens; the IDs are under Settings → General for the team and project.
The harness passes them straight to `Sandbox.create` and refuses a partial set.

**C. Nothing.** In an interactive terminal the SDK opens a browser sign-in and infers the
team and project. Fine for a first try, not for a repeatable run.

Whichever you choose, the sandbox's project id is recorded in `traces.jsonl` only as a
redaction marker, never as its value.

## Run it

**Validate offline first — no credentials, no network, nothing billed:**

```bash
npm test                 # 19 unit tests over the rubric, redaction, parser and schema
```

**Then the experiment.** Both call-making steps fail closed unless `LIVE_RUN_APPROVED=yes`:

```bash
export LIVE_RUN_APPROVED=yes
npm run trace:live       # Copilot CLI in one disposable sandbox per scenario
npm run eval:live        # 3 × spans calls through AI Gateway
npm run summarize
npm run report
```

Re-evaluating the same traces with a different evaluator set takes its own marker, so the
two runs' `result_id`s never collide:

```bash
EVAL_RUN_MARKER="jevlive2-$(date -u +%Y%m%d%H%M%S)" npm run eval:live
```

**Analyse:**

```bash
npm run ui               # http://127.0.0.1:5177, or use the hosted copy at https://jev-evals.vercel.app
```

The UI reads only the preserved artifacts and is a three-pane drill-down:

1. **Scenarios** — filter by name or design band, with a contested-span count per scenario.
2. **Bash spans** — the selected scenario's spans, each showing the command, all three
   verdicts, exit code and duration. The left edge carries the agreement signal: green when
   all three agreed, amber on a split, red on a three-way split. Arrow keys walk the list.
3. **Detail** — the selected span in full: the agent's description, the command, raw tool
   input, redacted output, and the three evaluators side by side.

**Hosting the UI.** The page is static; only its data is precomputed. `npm run ui:build`
writes `ui/dist/` (gitignored, about 750 KB) with the page files, the same dataset the local
server assembles as `data.json`, and the repeatability report. Any file host can serve that
folder. `vercel.json` in this folder points Vercel at it, so `vercel deploy` run from here in a
linked project builds and publishes it in one step.

Detail that would otherwise crowd the default view is disclosed on demand: aggregates sit
behind the `overview` button, and raw tool input, scenario/run metadata and the agent event
list are collapsed sections. A `raw JSON` toggle swaps the span for its stored records.
Any field that was not captured renders as *unavailable* rather than being guessed, and
derived values say that they are derived. The panes stack on narrow screens.

## Artifacts

| file | contents |
|---|---|
| `artifacts/traces.jsonl` | one normalized trace per scenario, including every Bash span |
| `artifacts/raw/<scenario>.agent-events.jsonl` | the agent's own untouched JSONL event stream |
| `artifacts/raw/<scenario>.copilot-otel.jsonl` | Copilot's OTel file export |
| `artifacts/evaluations.raw.jsonl` | raw evaluator responses, unmodified |
| `artifacts/evaluations.normalized.jsonl` | one shared-schema row per (span, evaluator) |
| `artifacts/summary.json` | summary dataset: distributions, agreement, latency, usage, cost |
| `artifacts/run-meta.json`, `artifacts/eval-meta.json` | provenance for the trace run and the evaluation run |
| `artifacts/repeatability/` | the repeatability study: two replays per model per stored span, plus its report |
| `docs/REPORT.md` | generated experiment report |

These artifacts are committed so that the report can be checked and the UI works straight
after `git clone` without spending anything. Scratch runs (`artifacts/archive/`,
`artifacts/run-1/`) are gitignored.

**Rerunning overwrites the committed results in place.** `trace:live`, `eval:live`,
`summarize` and `report` write to the fixed paths above, so your run replaces the baseline
in your working tree (`git diff` shows the change; `git checkout artifacts/` restores it).
The repeatability files are not touched by those steps, which makes them stale: they replay
the 31 spans of the original baseline, and `verify.mjs` checks for exactly that count. After
a new baseline, either rerun the three repeatability scripts below or treat
`artifacts/repeatability/` as belonging to the original run only. To keep the original
intact, copy it aside first: `cp -r artifacts artifacts/archive/baseline-$(date -u +%Y%m%d)`.

### Repeatability study

`artifacts/repeatability/scripts/` holds the three scripts behind
`artifacts/repeatability/REPORT.md`. Only `repeat.mjs` makes model calls, and only with
`LIVE_RUN_APPROVED=yes`:

```bash
node artifacts/repeatability/scripts/verify.mjs    # read-only preflight and post-check
node artifacts/repeatability/scripts/repeat.mjs    # two fresh calls per model per stored span
node artifacts/repeatability/scripts/analyze.mjs   # writes repeat-analysis.json
```

The charts in `docs/images/` are rendered from the saved analysis by
`tools/render_repeatability_chart.py` and `tools/render_repeatability_paths.py`. They need
Python 3 with Pillow (`pip install pillow`) and read fonts from
`/System/Library/Fonts/Supplemental`, so out of the box they run on macOS only.

## Safety

- All scenario data is fabricated. Credential-shaped values are fake.
- Destructive commands are scoped to a throwaway directory inside the sandbox workspace.
- Every outbound request in the fixtures targets `127.0.0.1:1`, so nothing leaves the sandbox.
- Captured output is redacted (credential patterns, sensitive key names) and byte-capped
  before it is written to any artifact or sent to any evaluator.
- `--secret-env-vars=COPILOT_PROVIDER_API_KEY` strips the Gateway key from the agent's own
  shell environment and redacts it from its output.
- The sandbox is disposable and is stopped after artifact retrieval. To clean up local
  output: `rm -rf artifacts/` (and remove `.env` if you are done).

## Rubric

`rubric/rubric.v2.json` + `rubric/shared-prompt.v2.txt`, versioned and hashed. The shared
prompt is a sanitized derivative of the supplied classification prompt, retaining the
evidence definitions, the decision rules and the four risk-level definitions; the sections
describing fields this experiment does not request (`risk_score`, `primary_category`,
`execution_outcome`, `impact_status`, `is_irreversible`) were removed, and the removals are
recorded in the rubric's `sanitization` block.

Span evidence is appended in one fixed form:

```
Command: {{command}}  Raw tool input: {{tool_input}}  Tool output, if present: {{tool_output}}  Error type, if present: {{error_type}}  Classify this span.
```

See `docs/METHODOLOGY.md` and `docs/LIMITATIONS.md`.

## License

MIT, under the [repository license](../../LICENSE).
