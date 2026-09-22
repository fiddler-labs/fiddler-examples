# Methodology self-review

A deliberate pass over this experiment looking for the four failure modes named in the
brief: bias, inconsistent prompts, cherry-picking, and invalid comparisons. Each finding
records what was done about it. Findings that remain open are listed as open.

## 1. Inconsistent prompts between evaluators

**Risk:** Jev is called through a typed evaluation API and the generative models through a
JSON-schema call. It would be easy for the two paths to drift into different wording, which
would make any disagreement an artifact of the harness.

**Resolved.** All three paths build their text from one file, `rubric/shared-prompt.v2.txt`,
and one span template in the rubric JSON. Jev receives the shared prompt as its question
`instructions` and the span block as its `state`; the generative models receive the same
shared prompt as `system` and the same span block as `prompt`.

This is checked rather than assumed: each stored result carries
`submitted.submitted_prompt_hash` over the concatenated text, `summary.json` reports
`identical_submitted_prompt_hashes.all_identical`, and a unit test asserts the three hashes
are equal for a span. If the paths ever diverge, the summary says so.

**Residual:** identical text is not identical framing — a `choice` question and a JSON
schema are different mechanisms. Recorded in LIMITATIONS.

## 2. Asking Jev for something it does not do

**Risk:** requiring a rationale from all three would either force Jev out of classifier mode
or make it look deficient.

**Resolved.** Jev is asked exactly one choice question and never asked for free text. Its
`rationale` is `null` with `rationale_availability: "unsupported_by_evaluator"`, which is
distinct from the failure markers `unavailable_error` and `unavailable_empty`. The report
and UI render it as "not requested (classifier)", not as a gap. Explanation quality is
compared only between the two generative peers, which are the two that write one.

## 3. Design band treated as ground truth

**Risk:** each scenario has an author-assigned band (benign / suspicious / high_risk). Scoring
models against it would silently measure agreement with the author, and the temptation to do
so is strong because it produces a satisfying accuracy number.

**Resolved.** The band is never included in the evaluation state, never sent to a model, and
never used to compute a score. It appears in the UI and report only alongside the explicit
note that it is author intent and not ground truth. No accuracy metric exists anywhere in
the codebase.

## 4. Cherry-picking disagreements

**Risk:** a report that lists a handful of disagreements chosen by the author is an argument,
not evidence.

**Resolved.** The report states the total count first, lists disagreements in span order
rather than by any interestingness criterion, states explicitly when the list is truncated,
and points at `summary.json` for the complete set. `summary.spans` contains every span with
all three answers, so any claim in the report can be recomputed.

## 5. Invented metrics

**Risk:** filling in a plausible cost or confidence is easy and makes the comparison look
more complete than it is.

**Resolved.** Cost is reported only when the Gateway returns one or `config/pricing.json`
supplies rates; otherwise it is `null` with an `unavailable_reason` string. No self-reported
confidence is requested from the generative models, so there is no pseudo-probability to
compare against Jev's real choice probabilities. The UI renders every missing field as
*unavailable* rather than blank or zero.

## 6. Derived aggregates passed off as model output

**Risk:** a scenario-level risk level looks like a model verdict but no model ever saw a
whole scenario.

**Resolved.** Scenario rollups are computed locally, carry
`derived_note: "derived_aggregate_not_a_model_response"`, and are labelled as derived in
both the UI and the report.

## 7. Simulated data mistaken for live evidence

**Risk:** the offline stub evaluators produce a complete, plausible-looking dataset.

**Resolved.** Dry artifacts are written to a separate directory, every row carries
`simulated: true`, and the report opens with a block quote stating it is not evidence about
model behaviour. The UI shows a simulated banner.

## 8. Order effects

**Risk:** evaluators are always called in the same order (jev, gemini, openai).

**Not a confound.** The three calls are independent, stateless, and share no context; order
cannot influence an answer. Noted for completeness rather than fixed.

## Open issues

- **Sample size.** A few dozen spans. No confidence intervals, no significance testing.
  Agreement figures are descriptive only.
- **Single run, single configuration.** No repeated sampling, so within-model variance is
  unmeasured and any between-model gap is confounded with it. This is the single biggest
  barrier to claiming that one model is more or less cautious than another.
- **Agent model forced by tier.** Anthropic models are unavailable on this Gateway tier, so
  the agent runs on `openai/gpt-4.1`. The span set is a property of that agent.
- **Span distribution is authored.** Scenarios were written to span a risk range, so label
  distributions describe the scenario set as much as the models.

## Claims this experiment can support

- Whether the three models agreed on specific spans, and exactly where they did not.
- Observed latency, token usage and reported cost per evaluator on this workload.
- Qualitative comparison of the two generative peers' rationales on identical evidence.

## Claims it cannot support

- That any model is more accurate, better calibrated, or safer than another.
- Any general statement about coding-agent risk, or about these models beyond this rubric
  version, this agent, and this span set.
