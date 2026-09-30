import crypto from 'node:crypto';
import fs from 'node:fs';
const R = process.cwd(); // run from the repo root
const { loadDotEnv, assertLiveApproved, EVALUATORS } = await import(`${R}/src/config.js`);
const { loadRubric, buildJevQuestions, renderSubmittedPrompt } = await import(`${R}/src/rubric.js`);
const { readJsonl, readJson, writeJsonl, writeJson, sha256 } = await import(`${R}/src/util.js`);
const { evaluateSpan } = await import(`${R}/src/evaluators/index.js`);

loadDotEnv();
assertLiveApproved('the three-model repeatability replicate evaluations');

const OUT = `${R}/artifacts/repeatability`;
fs.mkdirSync(OUT, { recursive: true });

const REPLICATES = 2;
const rubric = loadRubric();
const baselineRunMeta  = readJson(`${R}/artifacts/run-meta.json`);
const baselineEvalMeta = readJson(`${R}/artifacts/eval-meta.json`);
const baselineNorm = readJsonl(`${R}/artifacts/evaluations.normalized.jsonl`);
const baselineRaw  = readJsonl(`${R}/artifacts/evaluations.raw.jsonl`);
const traces = readJsonl(`${R}/artifacts/traces.jsonl`);
const spans = traces.flatMap(t => t.spans.map(s => ({ span: s, trace: t })));

const marker = `jevrepeat-${new Date().toISOString().replace(/[-:T]/g,'').slice(0,14)}-${crypto.randomBytes(3).toString('hex')}`;
// normalizeResult derives result_id from run_marker; a distinct marker keeps
// these rows from ever colliding with the untouched baseline rows.
// The baseline this study replays is an evaluation run, so its marker comes
// from eval-meta.json; run-meta.json names the trace run underneath it.
const baselineMarker = baselineEvalMeta.run_marker ?? baselineRunMeta.run_marker;
const runMeta = { ...baselineRunMeta, run_marker: marker };

const startedAt = new Date().toISOString();
console.log(`[repeat] marker=${marker}`);
console.log(`[repeat] spans=${spans.length} evaluators=${EVALUATORS.length} replicates=${REPLICATES} calls=${spans.length*EVALUATORS.length*REPLICATES}`);
console.log(`[repeat] rubric=${rubric.rubric_id}@${rubric.rubric_version} hash=${rubric.rubric_hash.slice(0,12)}`);

const normalized = [], raw = [];
let done = 0, errors = 0;
const total = spans.length * EVALUATORS.length * REPLICATES;

for (const { span, trace } of spans) {
  for (let rep = 1; rep <= REPLICATES; rep++) {
    for (const evaluator of EVALUATORS) {
      const base    = baselineNorm.find(r => r.span_id === span.span_id && r.evaluator === evaluator.key) ?? null;
      const baseRaw = baselineRaw .find(r => r.span_id === span.span_id && r.evaluator === evaluator.key) ?? null;
      const requestedAt = new Date().toISOString();
      // Identical code path to the baseline run: same evaluateSpan dispatch,
      // same rubric object, same evidence. Nothing is re-rendered or re-worded.
      const out = await evaluateSpan({ rubric, span, trace, runMeta, evaluator });
      const completedAt = new Date().toISOString();
      if (out.normalized.error) errors++;

      const submitted = renderSubmittedPrompt(rubric, span);
      const provenance = {
        study: 'three-model-repeatability',
        replicate: rep,
        replicates_planned: REPLICATES,
        evaluator: evaluator.key,
        model: evaluator.model,
        evaluator_kind: evaluator.kind,
        repeat_run_marker: marker,
        baseline_run_marker: baselineMarker,
        baseline_result_id: base?.result_id ?? null,
        baseline_risk_level: base?.risk_level ?? null,
        baseline_probabilities: base?.probabilities ?? null,
        baseline_created_at: base?.created_at ?? null,
        baseline_model_id: baseRaw?.raw_response?.response?.modelId ?? null,
        // Re-derived and compared, so a silent prompt drift cannot pass unnoticed.
        submitted_prompt_hash: sha256(submitted),
        submitted_prompt_hash_matches_baseline: base ? sha256(submitted) === base.submitted.submitted_prompt_hash : null,
        span_evidence_hash_matches_baseline:  base ? span.span_evidence_hash === base.submitted.span_evidence_hash : null,
        rubric_hash_matches_baseline: rubric.rubric_hash === (base?.rubric?.rubric_hash ?? null),
        questions_payload_hash: evaluator.kind === 'typed' ? sha256(JSON.stringify(buildJevQuestions(rubric))) : null,
        requested_at: requestedAt,
        completed_at: completedAt,
        source_traces_file: 'artifacts/traces.jsonl',
        reran_agent: false,
        executed_commands: false,
      };

      const n = { ...out.normalized, result_id: `${marker}:${span.span_id}:${evaluator.key}:r${rep}`, replicate: rep, provenance };
      normalized.push(n);
      raw.push({ ...out.raw, result_id: n.result_id, replicate: rep, provenance });

      done++;
      process.stdout.write(`\r[repeat] ${done}/${total} calls (errors=${errors})`);
    }
  }
}
process.stdout.write('\n');
const finishedAt = new Date().toISOString();

writeJsonl(`${OUT}/repeat.normalized.jsonl`, normalized);
writeJsonl(`${OUT}/repeat.raw.jsonl`, raw);

const perModel = {};
for (const e of EVALUATORS) {
  const rows = normalized.filter(r => r.evaluator === e.key);
  const rr   = raw.filter(r => r.evaluator === e.key);
  perModel[e.key] = {
    model: e.model,
    kind: e.kind,
    calls: rows.length,
    errors: rows.filter(r => r.error).length,
    observed_model_ids: [...new Set(rr.map(r => r.raw_response?.response?.modelId).filter(Boolean))],
    baseline_model_ids: [...new Set(baselineRaw.filter(r => r.evaluator === e.key).map(r => r.raw_response?.response?.modelId).filter(Boolean))],
    prompt_hash_matches_baseline: rows.filter(r => r.provenance.submitted_prompt_hash_matches_baseline === true).length,
  };
}

writeJson(`${OUT}/repeat-meta.json`, {
  study: 'three-model-repeatability',
  mode: 'live', simulated: false,
  repeat_run_marker: marker,
  baseline_run_marker: baselineMarker,
  baseline_eval_created_at: baselineEvalMeta.created_at,
  evaluators: EVALUATORS,
  replicates_per_span_per_model: REPLICATES,
  spans: spans.length,
  calls: normalized.length,
  errors,
  per_model: perModel,
  rubric_id: rubric.rubric_id,
  rubric_version: rubric.rubric_version,
  rubric_hash: rubric.rubric_hash,
  shared_prompt_hash: rubric.shared_prompt_hash,
  rubric_hash_matches_baseline: rubric.rubric_hash === baselineEvalMeta.rubric_hash,
  shared_prompt_hash_matches_baseline: rubric.shared_prompt_hash === baselineEvalMeta.shared_prompt_hash,
  config_notes: {
    call_path: 'src/evaluators/index.js evaluateSpan - unchanged, the same dispatch the baseline used',
    jev_request_options: 'experimental_evaluate called with model/state/questions only; no temperature, top_p, seed or version pin is available or set',
    generative_request_options: 'generateObject called with model/schema/system/prompt plus the providerOptions declared in src/config.js; no temperature, top_p or seed is set. Each peer is pinned to its vendor-recommended minimal reasoning setting (Gemini thinkingBudget 0, gpt-5.4-nano reasoningEffort none) and to one serving provider (gateway.only), identically in baseline and repeats',
    rate_limit_handling: 'a rate-limited call waits for the window and is retried rather than being recorded as an error; per-evaluator pacing is available via minIntervalMs in src/config.js and is not needed on the current plan. Waiting changes when a call is made, not what is sent',
    version_pinning: 'the Gateway returns only the model slug for all three models; no version or build identifier is exposed, so identical model builds between baseline and repeats cannot be proven',
    no_agent_rerun: 'Copilot was not rerun; no captured shell command was executed; no sandbox was created',
  },
  started_at: startedAt, finished_at: finishedAt, created_at: finishedAt,
});

console.log(`[repeat] wrote ${normalized.length} normalized + ${raw.length} raw rows to artifacts/repeatability/`);
console.log(`[repeat] errors=${errors}`);
console.log(JSON.stringify(perModel, null, 2));
