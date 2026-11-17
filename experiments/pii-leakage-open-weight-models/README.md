# PII leakage by open-weight models in customer-service agents

> Part of [fiddler-examples](https://github.com/fiddler-labs/fiddler-examples). Everything
> below is run from this folder, `experiments/pii-leakage-open-weight-models/`.

Code, tasks and saved transcripts behind the article *Evaluating Open-Weight
Models for PII Leakage in Customer-Service Agents*. With them you can:

1. **Check every number in the article** from the saved transcripts. This is free and takes about a minute.
2. **Re-run the conversations** against live models. This costs money.

[`CLAIMS.md`](CLAIMS.md) maps each claim in the article to the runs and checks
behind it.

## What was tested

A scripted caller tries to get a customer's stored email, phone number, date of
birth or address from a bank's support agent before proving who they are. The
agent is an LLM with customer-lookup tools in the `banking_knowledge` domain of
[tau2-bench](https://github.com/sierra-research/tau2-bench). A deterministic
checker then scores whether any stored detail, or part of one, appeared in an
agent reply before the caller was verified.

| Set | Tasks | What the caller tries |
|---|---|---|
| Base | 35 (7 families of 5) | confusion, correction, fishing, oracle (yes/no guesses), third party, authority, cross-account |
| Hard | 20 (4 families of 5) | pasted fake system/tool text, transformed values, documents that would contain the value, slow drip |
| Controls | 10 | a verified customer asks for their own email or address (a model that refuses everything fails these) |
| Adaptive | 10 × 3 trials | an LLM attacker that changes tactics (Claude Haiku 5.5 only) |

The task definitions are in [`study/tasks/leak/README.md`](study/tasks/leak/README.md)
and the scoring rules are in [`study/scripts/check_banking_leaks.py`](study/scripts/check_banking_leaks.py).
All customer records are synthetic, from tau2-bench's fictional bank.

## Repository layout

```
setup.sh                 builds ./tau2-bench: upstream at a pinned commit + the patch + the study files
patches/tau2-bench.patch changes to tau2-bench (scripted customer, leak task set, small fixes)
study/tasks/leak/        the 75 leak tasks, their cases and splits
study/scripts/           task generator and leak checker
results/<run>/           sanitized transcripts of every run the article cites (gzipped tau2 results.json)
reproduce_claims.py      recomputes the article's figures from results/ and compares them
run_all.sh               re-runs the conversations against live models
CLAIMS.md                claim → evidence map, including where the article and the data disagree
```

## 1. Check the article's numbers (no API key)

Requires `git` and [uv](https://docs.astral.sh/uv/).

```bash
./setup.sh                                     # clones tau2-bench, applies the patch, installs
tau2-bench/.venv/bin/python reproduce_claims.py
```

The script prints one row per claim: the figure in the article, the figure from
the transcripts, and whether they match. Pass `--strict` to exit non-zero on any
mismatch.

To see the per-conversation scoring for one run:

```bash
cd tau2-bench
uv run --no-sync python scripts/check_banking_leaks.py ../results/leak-all-gemma-4-31b-20261028b/results.json.gz
```

To run the tests for the patch, task files and checker:

```bash
cd tau2-bench
uv run --no-sync pytest -q tests/test_scripted_user.py tests/test_domains/test_banking_knowledge/test_leak_tasks.py
```

## 2. Re-run the conversations (paid)

All models were called through the [Vercel AI Gateway](https://vercel.com/ai-gateway).

```bash
cp .env.example tau2-bench/.env               # then add your VERCEL_AI_GATEWAY_API_KEY
./run_all.sh smoke                            # one conversation, to check the setup
./run_all.sh gpt-oss-120b                     # one model: base + controls, then hard
./run_all.sh                                  # everything the article reports
```

What the original runs were billed for agent calls on the gateway, from the costs
recorded in `results/`:

| Model | Conversations | Billed |
|---|---|---|
| Mistral Large 4 | 65 | $22.21 |
| Nemotron 3 Ultra | 65 | $3.23 |
| gpt-oss-120b | 65 | $1.37 |
| Gemma 4 31B | 65 | $0.36 |
| Claude Haiku 5.5 | 95 (incl. 30 adaptive) | $0.27 |
| Nemotron 3.5 Lightning | 65 | $0.00 (gateway free tier) |
| Injection screening | 9 | $0.31 |
| **Total** | | **≈ $28** |

The scripted customer makes no model calls. The NL judge and the adaptive attacker
(GPT-4.1) add a little on top. Mistral Large 4 costs the most because its
conversations grew long: up to 39 agent calls, each resending the full history.

`run_all.sh` uses the agent settings recorded in the saved results. Each task ran
once in the study. Providers are not fully deterministic, so a re-run will
reproduce the overall pattern but some individual tasks will come out differently.
For example, 4 of the 11 Nemotron 3 Ultra tasks scored in both its cut-off run and
its full rerun changed outcome. Providers can also change or retire the model
behind a gateway ID, or the free tier used for Nemotron 3.5 Lightning. If a model
is no longer served, the saved transcripts are still the record of what it did.

## How the saved runs were combined

Some runs were cut off by an API budget limit or hit provider errors, and the
missing tasks were run again. `reproduce_claims.py` lists the runs used for each
model (`RUNS`). For each task it takes the first run that scored it, so every model has exactly one
scored conversation per task: 55 attacks and 10 controls. Claude Haiku 5.5 also has
30 adaptive conversations. The two cut-off runs that a full rerun replaced
(`leak-all-*-20261028`) are included only for the run-to-run comparison.

## What was removed from the original working repo

The study was run in a larger working copy of tau2-bench. This repo keeps only
what the leak study needs. It drops a third-party safety-scoring integration that
was observing the runs (it never influenced the agent or the scores), LLM safety
judges, other adversarial task suites, and a UI. Transcripts had those scores,
error tracebacks and local paths removed. The conversations themselves are
unchanged.

## Licence

MIT, © Fiddler Labs (see [`LICENSE`](LICENSE)). tau2-bench is © Sierra Research,
MIT licence. The patch modifies it, and its notice is included in `LICENSE`.
