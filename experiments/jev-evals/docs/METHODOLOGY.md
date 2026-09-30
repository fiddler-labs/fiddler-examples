# Methodology

## Question

Given one shell command a coding agent chose to run, do three models — a dedicated
classifier and two small generative models — assign the same risk level, and where do they
diverge?

## Unit of analysis

**One Bash tool span.** Not a whole scenario, not a whole session. Each evaluator sees one
command in isolation, exactly as the shared rubric states ("you cannot see other spans").
Scenario- and run-level numbers in the report and UI are derived locally by aggregating
span answers and are labelled `derived_aggregate_not_a_model_response`.

## Trace generation

Traces come from the real GitHub Copilot CLI (pinned 1.0.86) running inside a disposable
Vercel Sandbox, one sandbox per scenario. The harness supplies only a natural-language
prompt. The agent selects its own tools and commands; we capture what it did from its own
`--output-format json` event stream, pairing `tool.execution_start` with
`tool.execution_complete` on `toolCallId`.

Only `bash` tool calls become evaluation units. Other tool calls (`view`, `apply_patch`,
`rg`, …) are preserved in the trace timeline but are not classified, because the rubric's
unit is a Bash span.

Model routing uses Copilot's BYOK path (`COPILOT_PROVIDER_BASE_URL` →
`https://ai-gateway.vercel.sh/v1`), which per `copilot help providers` does not require
GitHub authentication. This keeps the agent and the evaluators on one credential and
removes any dependence on a host GitHub session.

### Fabricated-but-real

Scenarios are fabricated and deterministic in their *prompt*, not in their execution. The
agent's response to a prompt is genuine and may vary between runs: it may refuse, take a
different route, or be blocked. That variability is data, not noise to be suppressed. The
`expected_coverage` field records what a scenario is meant to exercise; it is a coverage
note, never a grading key.

### Separating claim from evidence

Three things are recorded separately and never merged:

- **the agent's claim** — its final message, self-reported and possibly wrong;
- **the guardrail decision** — whether a call was denied before execution;
- **the execution evidence** — exit code, output, duration from the tool result.

A denied call carries `error_type: "denied"`, `executed: false`, and a null `tool_output`,
so a block is never read as a safe outcome and never as a successful one.

## Evaluation

All three evaluators receive:

- the **same shared prompt text** (one file, hashed), and
- the **same span evidence block** rendered from one fixed template.

Equality is auditable, not asserted: every stored result carries
`submitted.submitted_prompt_hash`, and `summary.json` reports whether all three hashes
matched for every span. A mismatch would show up as
`identical_submitted_prompt_hashes.all_identical: false`.

| | Jev | Gemini 3.5 Flash Lite | GPT-5.4 nano |
|---|---|---|---|
| call | `experimental_evaluate`, one typed `choice` question | `generateObject` | `generateObject` |
| returns | `risk_level` | `risk_level`, `rationale` | `risk_level`, `rationale` |
| probabilities | when the provider returns them | not returned | not returned |

Jev is deliberately used as a classifier. Asking it for free text would change what it is,
so its `rationale` is always `null` with `rationale_availability:
"unsupported_by_evaluator"`. That marker distinguishes "not requested" from "requested and
missing" (`unavailable_error`, `unavailable_empty`).

### Why these two peer models

The selection rule is **the newest model in each vendor's lowest-cost tier that the vendor
documents as suitable for classification** — the cost point where guardrail-volume traffic
actually runs. It is not a merit selection, and no benchmark was consulted.

The tier clause is load-bearing. Without it the rule would admit any current model, since
every capable model can classify, and would stop selecting anything.

Both vendors name this task in their own documentation [1][2][3]:

- OpenAI's model page lists GPT-5.4 nano's targets as "classification, data extraction,
  ranking, and sub-agents", and calls it the "cheapest GPT-5.4-class model for simple
  high-volume tasks" [1].
- Google describes Gemini 3.5 Flash-Lite as "a low-latency, cost-effective multimodal model
  optimized for high-throughput, low-cost execution for subagent tasks and document
  parsing", named for "simple data extraction" and "high-volume agentic workflows" [2].
  Google's own developer guide is the source that names this task directly: "For
  high-volume extraction, routing, or classification: leave `thinking_level` at `minimal`
  (default) for maximum throughput" [3].

The two vendors document it with different force, which is why the rule says *suitable*
rather than *designated*. OpenAI names classification on the model page itself. For Google
it appears in the developer guide covering the 3.x Flash-Lite line, while the 3.5 model
page speaks of subagent work and document parsing — the tier was repositioned between
generations, from the high-volume workhorse framing that 3.1 Flash-Lite still carries
("workhorse model for high-volume use cases … RAG snippet ranking, translation, data
extraction") toward agentic use. Suitability is the claim the evidence supports for both;
designation would only be true of nano.

Both sit on the bottom rung of their vendor's ladder — nano < mini < full, Flash-Lite <
Flash < Pro — and in the same price bracket, which is what makes them peers of each other:

| | input / 1M | output / 1M |
|---|---|---|
| `google/gemini-3.5-flash-lite` | $0.30 | $2.50 |
| `openai/gpt-5.4-nano` | $0.20 | $1.25 |

Each peer is also pinned to the cheapest reasoning setting its vendor recommends for
classification — `thinkingLevel: minimal` for Flash-Lite [3], `reasoningEffort: none` for
nano, whose documented default is already `none` [1] — and to a single serving provider, so
neither the amount of thinking nor the provider behind the slug varies between them or
between runs. Both pins state the documented default explicitly rather than change it; the
point is that the setting is recorded in the run, not inherited silently.

### References

Accessed 2026-09-21. Prices are list rates from the vendors, not the rates the Gateway bills.

1. GPT-5.4 nano — model page, OpenAI. <https://developers.openai.com/api/docs/models/gpt-5.4-nano>
2. Gemini 3.5 Flash-Lite — model page, Google AI for Developers.
   <https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite>
3. "Gemini 3.6 Flash & 3.5 Flash-Lite: Developer guide", Patrick Loeber for Google AI.
   <https://dev.to/googleai/gemini-36-flash-35-flash-lite-developer-guide-268i>
4. Introducing GPT-5.4 mini and nano, OpenAI. <https://openai.com/index/introducing-gpt-5-4-mini-and-nano/>
5. Gemini Developer API pricing, Google. <https://ai.google.dev/gemini-api/docs/pricing>
6. Gemini 3.5 Flash-Lite, Google DeepMind — "Best for low-latency and high throughput
   agentic tasks". <https://deepmind.google/models/gemini/flash-lite/>
7. "Gemini 3.6 Flash and Gemini 3.5 Flash-Lite are now available on AI Gateway", Vercel.
   <https://vercel.com/changelog/gemini-3-6-flash-3-5-flash-lite-on-ai-gateway>

Jev belongs to the comparison by **function, not tier**: a purpose-built classifier against
the two generalists each vendor recommends when the job is classification. The question is
"specialist versus the generalist a team would otherwise reach for", not "three peers of a
kind".

Three things this rule does not claim:

- **Not equal capability.** Tier is defined inside each vendor's own lineup. Nothing makes
  Google's cheapest model equivalent to OpenAI's cheapest, so a disagreement between the
  two is not evidence that either tier is behind.
- **Not equal configuration.** Both peers ran at provider defaults, and the defaults
  differ: the nano tier reasons before answering, Flash-Lite largely does not. In the first
  run, on provider defaults, nano emitted 42,678 output tokens against Flash-Lite's 1,799 —
  a configuration gap as much as a model gap. Both are now pinned to their vendor's minimal
  reasoning setting, which closed it: 4,321 against 2,713 over twice as many calls.
- **Not the newest.** Model IDs and prices are those in force on the run date recorded in
  `run-meta.json`. Both vendors have since shipped newer models in the same tier.

## Metrics

Recorded per call: `risk_level`, `rationale`, `probabilities`, `latency_ms`, token `usage`,
estimated `cost`, `warnings`, `error`.

Every one of these is nullable. Cost is reported only when the provider returns one or when
`config/pricing.json` supplies rates; otherwise it is `null` with an explicit
`unavailable_reason`. No metric is ever back-filled, inferred from another model, or
estimated from a guessed rate.

## Comparisons that are made

- **Classification agreement** — exact-match rate per pair, and unanimous / split /
  three-way splits per span.
- **Latency and token usage** — per evaluator, where reported.
- **Explanation quality** — **only between Gemini and nano**, the two that write one.

## Comparisons that are deliberately not made

- **Jev vs the others on explanation quality.** Jev was never asked for one.
- **Accuracy against the design band.** The author's `design_band` is never shown to a
  model and is not ground truth; it describes what a scenario was built to exercise.
  Treating it as a label would measure agreement with the author, not correctness.
- **Typed probabilities vs generative confidence.** Only genuine typed probabilities are
  stored. No self-reported confidence is requested from the generative models, so there is
  nothing to falsely equate.

## Reproducibility

- One run marker per run, stamped on every trace, span and result.
- Rubric and shared prompt hashed; both hashes stored beside every result.
- Raw agent event streams, raw OTel export, and raw evaluator responses all preserved.
- Dry-run mode reproduces the whole pipeline offline with deterministic stubs, so the
  plumbing can be verified independently of any model's behaviour.
