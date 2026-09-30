#!/usr/bin/env node
import path from 'node:path';
import fs from 'node:fs';
import { ARTIFACTS, EVALUATORS, PATHS } from '../config.js';
import { readJson, readJsonl } from '../util.js';

const dir = ARTIFACTS;
const summary = readJson(path.join(dir, 'summary.json'));
const traces = readJsonl(path.join(dir, 'traces.jsonl'));
const results = readJsonl(path.join(dir, 'evaluations.normalized.jsonl'));
// The evaluators the loaded run used, not the ones config.js names today.
const evalMetaFile = path.join(dir, 'eval-meta.json');
const evalMeta = fs.existsSync(evalMetaFile) ? readJson(evalMetaFile) : null;

const f = (v, unit = '') => (v === null || v === undefined ? '_unavailable_' : `${v}${unit}`);
const pct = (v) => (v === null || v === undefined ? '_unavailable_' : `${(v * 100).toFixed(1)}%`);

const lines = [];
const evaluatorNames = (evalMeta?.evaluators ?? EVALUATORS).map((e) => e.model ?? e.key);
lines.push(
  `# Bash-span risk classification: ${evaluatorNames.length ? evaluatorNames.join(' vs ') : 'three evaluators'}`,
);
lines.push('');
lines.push(`Generated ${summary.generated_at} · run marker \`${summary.run_marker}\``);
lines.push('');
lines.push(`## Setup`);
lines.push('');
lines.push(`- Rubric: \`${summary.rubric.rubric_id}\` v${summary.rubric.rubric_version}, hash \`${summary.rubric.rubric_hash?.slice(0, 16)}\``);
lines.push(`- Shared prompt hash: \`${summary.rubric.shared_prompt_hash?.slice(0, 16)}\``);
lines.push(`- Unit of evaluation: one Bash tool span.`);
lines.push(`- Scenarios: ${summary.totals.scenarios}; Bash spans captured: ${summary.totals.spans}; evaluation calls: ${summary.totals.calls} (${summary.totals.spans} × ${summary.totals.evaluators}).`);
lines.push(`- Errored calls: ${summary.totals.errors}.`);
lines.push(`- Identical submitted input across all three evaluators: **${summary.totals.identical_submitted_prompt_hashes.all_identical ? 'verified' : 'NOT VERIFIED'}** (${summary.totals.identical_submitted_prompt_hashes.spans_checked} spans checked).`);
const agentMeta = traces.find((t) => t.execution?.agent)?.execution?.agent;
if (agentMeta) {
  lines.push(`- Agent under trace: \`${agentMeta.cli}\` routed to \`${agentMeta.provider_base_url}\` (wire model \`${agentMeta.wire_model}\`), GitHub auth used: ${agentMeta.github_auth_used ? 'yes' : 'no'}.`);
}
const zeroSpan = traces.filter((t) => t.spans.length === 0);
if (zeroSpan.length) {
  lines.push('');
  lines.push(`### Scenarios contributing no evaluation units`);
  lines.push('');
  lines.push(`The rubric's unit is a Bash tool span. ${zeroSpan.length} scenario(s) produced none, because the agent completed the task with other tools (file edit, view, search). These scenarios are captured as traces but contribute zero evaluation calls:`);
  lines.push('');
  for (const t of zeroSpan) {
    lines.push(`- \`${t.scenario_id}\` — agent claim: ${t.agent_claim ? `"${t.agent_claim.slice(0, 180).replace(/\n/g, ' ')}…"` : '_none captured_'}`);
  }
  lines.push('');
  lines.push(`This is a real coverage limit: a task can be carried out without a single shell command, and a Bash-span classifier sees nothing in that case.`);
}
lines.push('');
lines.push(`## Per-evaluator results`);
lines.push('');
lines.push(`| evaluator | model | low | medium | high | critical | latency p50 | latency p95 | total tokens | est. cost | probabilities | rationale |`);
lines.push(`|---|---|---|---|---|---|---|---|---|---|---|---|`);
for (const e of EVALUATORS) {
  const p = summary.per_evaluator[e.key];
  const d = p.distribution ?? {};
  lines.push(
    `| ${e.key} | \`${p.model}\` | ${d.low ?? 0} | ${d.medium ?? 0} | ${d.high ?? 0} | ${d.critical ?? 0} | ` +
      `${f(p.latency_ms?.p50, ' ms')} | ${f(p.latency_ms?.p95, ' ms')} | ${f(p.usage_totals?.total_tokens)} | ` +
      `${p.estimated_cost_usd === null ? `_unavailable_ (${p.cost_unavailable_reason})` : `$${p.estimated_cost_usd.toFixed(6)}`} | ` +
      `${p.probabilities_available ? 'returned' : '_not returned_'} | ` +
      `${p.rationale_capability === 'not_requested_classifier_only' ? '_not requested (classifier)_' : `${p.rationale_count} returned, mean ${p.rationale_length_chars?.mean ?? '?'} chars`} |`,
  );
}
lines.push('');
lines.push(`## Agreement`);
lines.push('');
lines.push(`Span-level agreement across all three: ` + Object.entries(summary.agreement_distribution).map(([k, v]) => `**${k}** ${v}`).join(', ') + '.');
lines.push('');
lines.push(`| pair | exact agreement | spans compared |`);
lines.push(`|---|---|---|`);
for (const [pair, v] of Object.entries(summary.pairwise_agreement)) {
  lines.push(`| ${pair.replace('|', ' vs ')} | ${pct(v.exact_agreement)} | ${v.compared_spans} |`);
}
lines.push('');
lines.push(`## Disagreements`);
lines.push('');
const disagreements = summary.spans.filter((s) => s.agreement !== 'unanimous');
lines.push(`${disagreements.length} of ${summary.spans.length} spans drew different labels. ` +
  `Listed below in span order${disagreements.length > 20 ? ', truncated to the first 20' : ''} — not selected by outcome. ` +
  `The complete set is in \`summary.json\`.`);
lines.push('');
if (!disagreements.length) lines.push('_No span drew different labels._');
for (const s of disagreements.slice(0, 20)) {
  lines.push(`- \`${s.span_id}\` (${s.agreement}) — ` + EVALUATORS.map((e) => `${e.key}: **${s.levels[e.key] ?? "n/a"}**`).join(', '));
  lines.push(`  - command: \`${(s.command ?? '').slice(0, 160)}\``);
  const gen = results.filter((r) => r.span_id === s.span_id && r.rationale);
  for (const r of gen) lines.push(`  - ${r.evaluator} rationale: ${r.rationale}`);
}
lines.push('');
lines.push(`## Per-scenario derived view`);
lines.push('');
lines.push(`Highest span risk per scenario. **Derived locally from span-level answers — no evaluator was shown a whole scenario.**`);
lines.push('');
lines.push(`| scenario | design band | spans | executed | denied | refusal detected | ` + EVALUATORS.map((e) => e.key).join(' | ') + ' |');
lines.push(`|---|---|---|---|---|---|` + EVALUATORS.map(() => '---|').join(''));
for (const s of summary.scenarios) {
  lines.push(
    `| ${s.scenario_id} | ${s.design_band} | ${s.spans} | ${s.outcome?.spans_executed ?? '?'} | ${s.outcome?.spans_denied_by_guardrail ?? '?'} | ` +
      `${s.outcome?.agent_refused ? 'yes' : 'no'} | ` +
      EVALUATORS.map((e) => s.derived_highest_span_risk[e.key] ?? 'n/a').join(' | ') + ' |',
  );
}
lines.push('');
lines.push(`## Reading guidance`);
lines.push('');
lines.push(`- The scenario design band is the author's intent. It is never shown to an evaluator and is **not** ground truth, so it does not measure accuracy.`);
lines.push(`- Jev is used as a classifier and was never asked for a rationale. Its blank rationale is a contract, not a failure.`);
const generativeNames = (evalMeta?.evaluators ?? EVALUATORS)
  .filter((e) => e.kind !== 'typed')
  .map((e) => e.model ?? e.key);
lines.push(`- Explanation quality is comparable only between ${generativeNames.join(' and ')}.`);
lines.push(`- Cost and token counts are reported only where the provider returned them or a price table is configured.`);
lines.push('');
lines.push(`See \`docs/METHODOLOGY.md\` and \`docs/LIMITATIONS.md\`.`);

const out = PATHS.report;
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, lines.join('\n') + '\n');
console.log(`[report] wrote ${out}`);
