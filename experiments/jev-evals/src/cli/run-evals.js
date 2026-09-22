#!/usr/bin/env node
import path from 'node:path';
import { ARTIFACTS, EVALUATORS, PATHS, assertLiveApproved, loadDotEnv } from '../config.js';
import { loadRubric } from '../rubric.js';
import { readJsonl, readJson, writeJsonl, writeJson } from '../util.js';
import { evaluateSpan } from '../evaluators/index.js';

loadDotEnv();
assertLiveApproved('the AI Gateway evaluations');

const dir = ARTIFACTS;
const tracesFile = path.join(dir, 'traces.jsonl');
const traces = readJsonl(tracesFile);
if (!traces.length) {
  console.error(`[evals] no traces at ${tracesFile}. Run the trace phase first.`);
  process.exit(1);
}
const runMeta = readJson(path.join(dir, 'run-meta.json'));
const rubric = loadRubric();

// The evaluation run marker defaults to the trace run's marker, which is right
// while one trace run is evaluated once. Evaluating the same traces a second
// time - a new evaluator set, say - would then reuse the first run's marker and
// mint colliding result_ids, so a re-evaluation passes its own marker.
const runMarker = process.env.EVAL_RUN_MARKER?.trim() || runMeta.run_marker;
// Rows carry the evaluation marker; the trace marker stays readable beside it.
const evalRunMeta = { ...runMeta, run_marker: runMarker, trace_run_marker: runMeta.run_marker };

const spans = traces.flatMap((t) => t.spans.map((s) => ({ span: s, trace: t })));
if (runMarker !== runMeta.run_marker) {
  console.log(`[evals] marker=${runMarker} (traces: ${runMeta.run_marker})`);
}
console.log(
  `[evals] rubric=${rubric.rubric_id}@${rubric.rubric_version} hash=${rubric.rubric_hash.slice(0, 12)}`,
);
console.log(`[evals] spans=${spans.length} evaluators=${EVALUATORS.length} calls=${spans.length * EVALUATORS.length}`);

const normalized = [];
const raw = [];
let done = 0;
for (const { span, trace } of spans) {
  // Evaluators run sequentially per span so one provider outage cannot be
  // mistaken for a disagreement in the results.
  for (const evaluator of EVALUATORS) {
    const out = await evaluateSpan({ rubric, span, trace, runMeta: evalRunMeta, evaluator });
    normalized.push(out.normalized);
    raw.push(out.raw);
    done += 1;
    if (out.normalized.error) {
      console.warn(`[evals] ${span.span_id} ${evaluator.key}: ERROR ${out.normalized.error.message}`);
    }
  }
  if (spans.length > 4) process.stdout.write(`\r[evals] ${done}/${spans.length * EVALUATORS.length} calls`);
}
process.stdout.write('\n');

const normFile = PATHS.evalNorm;
const rawFile = PATHS.evalRaw;
writeJsonl(normFile, normalized);
writeJsonl(rawFile, raw);
writeJson(path.join(dir, 'eval-meta.json'), {
  mode: 'live',
  simulated: false,
  run_marker: runMarker,
  trace_run_marker: runMeta.run_marker,
  rubric_id: rubric.rubric_id,
  rubric_version: rubric.rubric_version,
  rubric_hash: rubric.rubric_hash,
  shared_prompt_hash: rubric.shared_prompt_hash,
  evaluators: EVALUATORS,
  spans: spans.length,
  calls: normalized.length,
  errors: normalized.filter((r) => r.error).length,
  created_at: new Date().toISOString(),
});

console.log(`[evals] wrote ${normalized.length} normalized + ${raw.length} raw rows`);
console.log(`[evals] errors=${normalized.filter((r) => r.error).length}`);
