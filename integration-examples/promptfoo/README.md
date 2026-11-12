# promptfoo-fiddler-redteam

A reference framework for continuously red-teaming your LLM guardrails.
It pairs [promptfoo](https://www.promptfoo.dev)'s open-source red-teaming
engine with **Fiddler Centor Models for Safety and PII/PHI** and **Fiddler Experiments**:

1. promptfoo generates adversarial prompts tailored to your application's
   purpose (PII leaks, harmful content, domain-specific attacks, and more).
2. The prompts are loaded into a Fiddler **Dataset**.
3. Each prompt is scored against Fiddler Guardrails.
4. The results land in a Fiddler **Experiment**, along with a per-category
   detection scorecard (recall) and a list of the prompts that got through.

The goal is a lightweight, repeatable check you can run daily or weekly to
answer one question: *are our guardrails still catching the attacks we care
about?* promptfoo's maintained library of attack generators handles the
"attack" side, and Fiddler handles the "measurement" side, so you don't have
to hand-curate an adversarial test set.

**Blog post:** _link coming soon_

## Tested with

| Component | Version |
|---|---|
| Fiddler SDK (`fiddler-evals`) | 0.7.0 |
| promptfoo | 0.124.1 |
| Python | 3.15 (requires 3.11+) |
| Last verified | 2026-11-06 |

Newer versions will likely work, but the Fiddler and promptfoo APIs this
example calls may change over time.

## What this is (and isn't)

This is a **starting point**: an "art of the possible" example of
connecting an open-source red-teaming tool to Fiddler's guardrail and
evaluation APIs. It is not a turnkey product, and it is not tuned for any
particular industry or use case.

- The included example config describes a healthcare benefits chatbot. It
  is only an illustration. Replace the `purpose` and plugin list with ones
  that describe your own application.
- The plugin-to-guardrail mapping, the scoring logic, and the scorecard are
  intentionally simple so they're easy to read, fork, and adapt.
- Detection rates depend heavily on your configuration, your thresholds,
  and the prompts promptfoo generates. Treat the numbers as a signal for
  your own setup, not as a benchmark.

Fork it, change it, and wire it into whatever scheduler or CI system you
already use.

## How it works

```
promptfooconfig.yaml            Fiddler
  (purpose + plugins)             |
       |                          |
       v                          |
 promptfoo redteam generate       |
       |                          |
       v                          |
  redteam.yaml (adversarial       |
  prompts, tagged by plugin)      |
       |                          |
       v                          v
   parse + categorize  ---->  Dataset.create()  (Project -> Application -> Dataset)
                                   |
                                   v
                              evaluate(dataset, task, [GuardrailDetectionEvaluator])
                                   |
                     for each prompt: call Fiddler Guardrails
                     (FTLPromptSafety for safety, the
                     sensitive-information API for PII)
                                   |
                                   v
                         Experiment (visible in Fiddler UI)
                                   |
                                   v
                     console/CSV scorecard: detection rate
                     per category and per plugin + missed prompts
```

**Why score the prompt text directly instead of a live app's response?**
The question here is "does our guardrail catch this kind of attack?", and
you can answer that without wiring up an LLM application. All you need is
a Fiddler instance. If you want to score your application's *responses*
instead, see
[Extending to score a live target's responses](#extending-to-score-a-live-targets-responses).
It's a small change.

**Ground truth.** promptfoo's red-team plugins generate prompts that are
adversarial *by construction* for their category (for example, `pii:direct`
always tries to extract PII). So every mapped prompt is treated as "should
be flagged," and the scorecard is a **per-category recall measurement**: of
all the PII-extraction attempts promptfoo generated, what fraction did the
PII guardrail catch?

Keep in mind what each guardrail actually detects. The PII guardrail finds
PII *present in the text*, not an *attempt* to extract it, so a request like
"show me another member's address" with no PII in it won't be flagged.
That's one reason scoring your application's responses (see below) is a
useful next step.

This setup does not measure false positives on benign traffic by default.
To do that, add benign prompts to the dataset with
`expected_outputs={"should_flag": False}`; the scorecard reports them
separately as benign controls and lists any false positives.

## Prerequisites

- Python 3.11+
- Node.js. The tool runs promptfoo through `npx`, pinned to the tested
  version. If `npx` isn't available, it falls back to a globally installed
  `promptfoo`.
- A Fiddler instance and API key (**Settings -> Credentials** in the
  Fiddler UI)
- The **first** time promptfoo's hosted generation service is used on a
  machine, it asks for an email address. Running `generate` interactively
  shows the prompt so you can answer it. For unattended runs (cron, CI
  runners that start fresh each time), set it non-interactively first:
  `npx promptfoo@0.124.1 config set email you@company.com`

## Install

Run these commands from this folder:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp examples/.env.example .env   # fill in FIDDLER_URL / FIDDLER_API_KEY / FIDDLER_PROJECT / FIDDLER_APPLICATION
```

`.env` is gitignored. Keep your API key there or in your secret manager,
never in the config files.

## Quickstart

```bash
# 1. Generate adversarial prompts with promptfoo (no Fiddler credentials needed for this step)
promptfoo-fiddler-redteam generate --config examples/promptfooconfig.yaml

# 2. Push them into Fiddler, score them against Guardrails, print a scorecard
promptfoo-fiddler-redteam run --input runs/generate-<timestamp>/redteam.yaml

# ...or do both in one step (this is what you'd schedule daily or weekly):
promptfoo-fiddler-redteam all --config examples/promptfooconfig.yaml
```

Unless you pass `-o`/`--csv-out`, each invocation saves its output under
`runs/<name-prefix>-<timestamp>/` (gitignored, local only): the generated
`redteam.yaml` plus `scorecard.csv`, `scorecard_by_plugin.csv` and
`misses.csv` in one directory. See
[Reusing previous runs](#reusing-previous-runs) before generating a new set.

Sample output (the format; your numbers will vary):

```
Category    Total  Detected    Rate  Errors
--------------------------------------------
pii             6         6  100.0%       0
safety          6         6  100.0%       0

No misses: all 12 scored adversarial prompt(s) were flagged.
```

How to read it:

- **Total / Detected / Rate** count only prompts that were actually scored.
- **Errors** are prompts where the guardrail or task call failed (a wrong
  API key, a timeout, an unexpected response). They're excluded from the
  rate and from the misses list, and `run` exits with code 2 so a scheduled
  job notices.
- Prompts from plugins with no mapped guardrail are listed as **skipped**,
  not counted as detected.

Each run creates a **new, timestamped** Fiddler Dataset and Experiment
instead of appending to a previous one. Scheduled runs give you a trend line
of guardrail detection rate over time in the Fiddler UI.

## Configuring the red team: `examples/promptfooconfig.yaml`

The sample config targets a fictional healthcare benefits support bot.
Change `redteam.purpose` to describe your own application, and adjust
`redteam.plugins` to match the risks you care about. See
[promptfoo's plugin list](https://www.promptfoo.dev/docs/red-team/plugins/)
for the full catalog. The plugin families this framework maps out of the
box:

| Plugin id prefix | Category | Fiddler guardrail used |
|---|---|---|
| `pii`, `pii:direct`, `pii:session`, `pii:social`, `pii:api-db` | `pii` | Sensitive Information (`entity_categories=PII`) |
| `harmful`, `harmful:*`, `toxicity`, `bias` | `safety` | `FTLPromptSafety` (Fiddler's safety model) |
| anything else | `other` | None. Skipped and listed separately in the scorecard |

A `phi` category (Sensitive Information with `entity_categories=PHI`) is
also available, but no plugin maps to it by default.

**Why promptfoo's `medical` plugins aren't mapped.** They test for bad
medical *advice* (hallucination, sycophancy, off-label use, and so on), not
PHI leakage. Scoring them with the PHI entity detector would only measure
whether a prompt mentions a drug or condition, not whether the attack was
caught. Only map a plugin to a guardrail that detects what that plugin's
prompts actually contain.

To map a new plugin family to a category, add its prefix to
`_CATEGORY_BY_PREFIX` in `src/promptfoo_fiddler_redteam/promptfoo_parser.py`.
To use a different guardrail for a category, edit
`src/promptfoo_fiddler_redteam/evaluators.py`.

> **Guardrails vs. the `pii_detection` evaluator.** Fiddler has two
> separate PII-related capabilities: the **Guardrails**
> `/v3/guardrails/sensitive-information` endpoint (supports
> `entity_categories: PII|PII_ALL|PHI`, which this framework uses) and a
> `pii_detection` **evaluator/enrichment** type (fixed entity list, no PHI
> mode, used for batch monitoring). They use different underlying models.
> See [Fiddler's docs](https://docs.fiddler.ai/protection/guardrails#pii/phi).
> If you extend this through the evaluator/enrichment framework instead of
> the Guardrails REST API, you will not get PHI coverage.

promptfoo **strategies** (jailbreak wrappers like `base64`, `jailbreak`,
`leetspeak`) also work. Add them under `redteam.strategies` to test whether
encoding or obfuscation gets past detection. They're optional: `basic` (no
wrapping) already produces adversarial prompts for each plugin.

## Extending to score a live target's responses

`evaluators.py`'s `make_task()` passes each prompt through unchanged, so the
guardrail scores the *prompt itself*. To test whether your application's
*response* to an attack leaks PII (a different but equally useful
question), change `task()` to call your application and return its response
as `"prompt"` in the output dict. The evaluator scores whatever text the
task returns:

```python
def task(inputs, extras, metadata):
    response = call_my_target_app(inputs["prompt"])
    return {
        "prompt": response,
        "category": metadata.get("category", "other"),
        "plugin_id": metadata.get("plugin_id"),
    }
```

The scorecard works the same way. Remember that "should be flagged" now
means "the response should contain something the guardrail catches", which
may not hold for every attack, so review the misses before reading the rate
as a failure.

## Reusing previous runs

Every `generate`/`all` run is saved locally under
`runs/<name-prefix>-<timestamp>/`. It's gitignored because it contains
adversarial prompt text you probably don't want in version control. Before
you generate a fresh set, check whether an existing run already covers what
you need. promptfoo's header comment in each `redteam.yaml` records the
plugins, strategies, and case count:

```bash
ls runs/
head -12 runs/*/redteam.yaml   # skim every run's "Plugins:" / "Strategies:" header
```

If one matches, reuse it instead of calling `generate` again (which hits
promptfoo's hosted generation service):

```bash
promptfoo-fiddler-redteam run --input runs/<matching-run>/redteam.yaml
```

**If an AI coding agent (Claude Code or similar) is driving this tool for
you**, it should check `runs/` before calling `generate`. See
[`AGENTS.md`](AGENTS.md) for the instructions it follows.

## Troubleshooting

- **"Missing required environment variable(s)"**: check that `.env` is
  populated and in your working directory, or export the variables directly.
- **`generate` times out in a scheduled job**: promptfoo is probably waiting
  for its one-time email prompt. Run
  `npx promptfoo@0.124.1 config set email you@company.com` as a setup step.
  The timeout defaults to 30 minutes; change it with `--timeout`.
- **"Couldn't reach Fiddler"**: `run` couldn't connect to your instance, or
  it stopped responding, while creating the dataset. It exits with code 1.
  Check `FIDDLER_URL` and that the instance is up.
- **Errors in the scorecard**: a guardrail or task call failed for those
  prompts. Every Fiddler call has a timeout (10 seconds to connect, 60 to
  respond) and is retried once if the connection drops, so these are calls
  that failed twice. The run continues, the failed prompts are excluded from
  the rate, and `run` exits with code 2. The message under the scorecard says
  whether it looks like a connection problem (usually temporary; rerun) or a
  rejected API key. `guardrails_client.py` also logs the raw response when it
  doesn't recognize the shape, for example after a Fiddler API change.

## Project layout

```
src/promptfoo_fiddler_redteam/
  config.py             FIDDLER_* env settings
  promptfoo_runner.py   subprocess wrapper around `promptfoo redteam generate`
  promptfoo_parser.py   redteam.yaml -> normalized, categorized test cases
  fiddler_bridge.py     Project/Application/Dataset creation + population
  guardrails_client.py  REST client for the sensitive-information endpoint
  resilience.py         timeouts, one retry on dropped connections, readable errors
  evaluators.py         task() + GuardrailDetectionEvaluator (safety via FTLPromptSafety)
  experiment_runner.py  wires it all into fiddler_evals.evaluate()
  report.py             per-category and per-plugin scorecard (console + CSV)
  cli.py                `generate` / `run` / `all` subcommands
examples/               sample promptfoo config + .env template
tests/                  unit tests against a captured promptfoo fixture + mocked Fiddler calls
runs/                   gitignored; saved redteam.yaml + scorecards/misses per run
AGENTS.md               instructions for AI coding agents driving this tool
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

Or with uv: `uv run pytest`.
