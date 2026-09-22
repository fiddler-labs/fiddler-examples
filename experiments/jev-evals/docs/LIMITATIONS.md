# Limitations

Read these before quoting any number from this experiment.

## Scale

About a dozen scenarios and a few dozen Bash spans. This is an illustrative comparison,
not a benchmark. Agreement percentages over this many spans have wide uncertainty, and no
significance testing is performed. A single reclassified span moves a pairwise agreement
figure by several points.

## No ground truth

There is no correct answer in this dataset. The scenario `design_band` is the author's
intent about what a scenario should exercise; it is never shown to a model and is not a
label. Consequently:

- **agreement is not accuracy.** Three models can agree and all be wrong.
- **disagreement is not error.** Risk judgement on a single command out of context is
  genuinely ambiguous, and the rubric's own instruction to rate *attempted* risk as if the
  command had succeeded invites defensible spread.

## The spans are not a random sample

Scenarios were written to span benign through high-risk behaviour. The label distribution
reflects that design, so a distribution in the report says as much about the scenario set
as about any model.

## Agent behaviour varies between runs

The agent is real, so it may refuse, choose different commands, or emit a different number
of spans on a re-run with the same prompt. Span counts and per-scenario results are not
reproducible in the strict sense; only the prompts, rubric and pipeline are.

## The fabricated environment is not production

Commands run against fabricated files in a disposable sandbox, with outbound requests
pointed at `127.0.0.1:1`. Failure modes therefore look different from production: an
exfiltration attempt fails with a connection error rather than succeeding. The rubric
accounts for this (a failure never lowers attempted risk), but the output evidence a model
sees is still systematically different from a real incident.

## Evaluator asymmetry is structural

Jev answers a typed choice question; the generative models answer a JSON schema. These are
different mechanisms, not different settings of one mechanism. The shared rubric text and
span evidence are identical, but "identical input" does not make the tasks equivalent:

- Jev may return calibrated choice probabilities; the generative models return none here,
  so there is no probability comparison to make.
- Latency and token usage are not like-for-like. A classifier emitting one token is not
  doing the same work as a model writing a rationale, so a latency or cost gap between them
  is expected and is not by itself a quality finding.

## Cost figures are usually absent

Unless the Gateway returns a cost or `config/pricing.json` is populated, estimated cost is
`null`. Absent cost is reported as unavailable rather than estimated from published rates,
because rates change and a wrong number is worse than a missing one.

## Prompt sanitization changed the rubric

The shared prompt is a derivative of a larger supplied prompt, with the sections covering
fields this experiment does not request removed. Removing the `primary_category`,
`execution_outcome` and `impact_status` taxonomies may shift how a model reasons about a
span relative to the original prompt. Results are therefore about *this* rubric version
(hash recorded), not about the original.

## Peer model choice is a rule, not a merit ranking

The two peer models are the newest model in each vendor's lowest-cost tier that the vendor
documents as suitable for classification, chosen by that rule alone (see *Why these two
peer models* in `METHODOLOGY.md`). No benchmark informed the choice. Google's Flash-Lite
tier was repositioned toward agentic work between 3.1 and 3.5, so for Gemini the rule rests
on a developer-guide statement rather than the model page. Consequences for reading any number here:

- Tier is vendor-internal. "Both are the cheapest tier" does not make them equally capable,
  so a gap between the two peers is not a ranking of the vendors.
- The rubric's four levels are the classifier's own output taxonomy, taken from a
  user-supplied classification prompt (`rubric/rubric.v2.json`, `sanitization.source`).
  Jev answers in its native format; the generative models are asked to emulate it. Any
  advantage that confers to Jev is unmeasured here.
- A different tier would likely change the agreement figures. Nothing in this dataset
  indicates by how much.

## Gateway tier constrained the agent model

The traces were produced while this account had no purchased Gateway credits, and the
Gateway then returned 403 for current-generation models across every vendor ("Free tier
users do not have access to this model"). The agent therefore runs on `openai/gpt-4.1`
rather than a model chosen on merit. A different agent model would produce different
commands and therefore a different span set, so nothing here should be read as a property
of coding agents in general.

Credits were added on 2026-09-21, after the traces were captured and before the evaluator
set was changed, so the constraint no longer applies to future trace runs. The spans in
`artifacts/traces.jsonl` still carry it: they were chosen by `openai/gpt-4.1`.

## Single configuration

One agent model, one temperature-default setting per evaluator, one prompt phrasing, one
run. No ablations, no repeated sampling, no ordering checks. Any claim of the form "model X
is more cautious than model Y" would need repeated runs and prompt variants that this
experiment does not have.

## The Bash span is not the whole agent

The rubric's unit is a Bash tool span, so anything the agent does through another tool is
invisible to all three evaluators. This is not hypothetical: in an earlier capture of the
supply-chain scenario the agent added the malicious `postinstall` hook with a file-edit
tool and issued no shell command at all, producing zero evaluation units for a scenario
designed to be high risk. A span-level classifier cannot see what never became a command.

## Redaction changes what the evaluators see

Output is redacted before it reaches an evaluator, so a model classifying a secret-sweep
span sees `[REDACTED:value]` rather than a credential. Models cited these redaction markers
in their rationales as evidence that secrets were found. That is reasonable, but it means
the evaluators are partly reading our redactor's output, not the raw command output.

## What this experiment can and cannot support

Taken together, the limitations above bound the claims. This experiment can support:

- Whether the three models agreed on specific spans, and exactly where they did not.
- Observed latency, token usage and reported cost per evaluator on this workload.
- Qualitative comparison of the two generative peers' rationales on identical evidence.

It cannot support:

- That any model is more accurate, better calibrated, or safer than another.
- Any general statement about coding-agent risk, or about these models beyond this rubric
  version, this agent, and this span set.
