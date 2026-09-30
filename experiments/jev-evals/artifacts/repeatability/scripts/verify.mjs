/**
 * Read-only verification for the three-model repeatability study.
 * Makes NO network calls and writes nothing. Run from the repo root:
 *   node artifacts/repeatability/scripts/verify.mjs
 *
 * Part A runs against the baseline alone and is the preflight that was run
 * before any replicate call was made. Part B checks the completed replay.
 */
const R = process.cwd(); // run from the repo root
const { EVALUATORS } = await import(`${R}/src/config.js`);
const { loadRubric, renderSubmittedPrompt, renderSpanBlock, buildJevQuestions } = await import(`${R}/src/rubric.js`);
const { readJsonl, readJson, sha256 } = await import(`${R}/src/util.js`);

let failures = 0;
const ok = (label, cond) => { if (!cond) failures++; console.log(`${cond ? 'PASS' : 'FAIL'}  ${label}`); };

const rubric    = loadRubric();
const evalMeta  = readJson(`${R}/artifacts/eval-meta.json`);
const traces    = readJsonl(`${R}/artifacts/traces.jsonl`);
const baseNorm  = readJsonl(`${R}/artifacts/evaluations.normalized.jsonl`);
const baseRaw   = readJsonl(`${R}/artifacts/evaluations.raw.jsonl`);
const spans     = traces.flatMap((t) => t.spans);

console.log('== A. baseline preflight (no calls made) ==');
ok('rubric_hash matches eval-meta',        rubric.rubric_hash === evalMeta.rubric_hash);
ok('shared_prompt_hash matches eval-meta', rubric.shared_prompt_hash === evalMeta.shared_prompt_hash);
ok('31 spans in traces',                   spans.length === 31);
ok('93 baseline rows',                     baseNorm.length === 93);

for (const e of EVALUATORS) {
  const rows = baseNorm.filter((r) => r.evaluator === e.key);
  // Re-derive the submitted prompt from the stored span and compare to what the
  // baseline recorded, so a silent prompt drift cannot pass unnoticed.
  const identical = spans.filter((s) => {
    const row = rows.find((r) => r.span_id === s.span_id);
    return row
      && sha256(renderSubmittedPrompt(rubric, s)) === row.submitted.submitted_prompt_hash
      && renderSpanBlock(rubric, s) === row.submitted.span_block
      && s.span_evidence_hash === row.submitted.span_evidence_hash;
  }).length;
  ok(`${e.key}: 31 baseline rows byte-identical to re-render`, identical === 31 && rows.length === 31);
  ok(`${e.key}: 0 baseline errors`, baseRaw.filter((r) => r.evaluator === e.key && r.error).length === 0);
}

console.log('\n== B. completed replay ==');
const meta = readJson(`${R}/artifacts/repeatability/repeat-meta.json`);
const norm = readJsonl(`${R}/artifacts/repeatability/repeat.normalized.jsonl`);
const raw  = readJsonl(`${R}/artifacts/repeatability/repeat.raw.jsonl`);
const an   = readJson(`${R}/artifacts/repeatability/repeat-analysis.json`);

ok('normalized rows = 186',        norm.length === 186);
ok('raw rows = 186',               raw.length === 186);
ok('result_ids unique',            new Set(norm.map((r) => r.result_id)).size === 186);
ok('raw/normalized ids aligned',   norm.every((r, i) => r.result_id === raw[i].result_id));
ok('31 spans x 3 models x 2 reps', new Set(norm.map((r) => r.span_id)).size === 31
                                   && new Set(norm.map((r) => r.evaluator)).size === 3
                                   && new Set(norm.map((r) => r.replicate)).size === 2);
ok('every raw row preserved',      raw.every((r) => r.raw_response !== null));
ok('0 replay errors',              norm.filter((r) => r.error).length === 0 && meta.errors === 0);
ok('usage on every row',           norm.every((r) => r.usage?.total_tokens !== null));
ok('cost on every row',            norm.every((r) => typeof r.cost?.estimated_usd === 'number'));
ok('timestamps on every row',      norm.every((r) => r.provenance?.requested_at && r.provenance?.completed_at));
ok('repeat marker distinct from baseline',
   norm.every((r) => r.run_marker === meta.repeat_run_marker && r.run_marker !== meta.baseline_run_marker));
ok('submitted_prompt_hash matched baseline on all 186',
   norm.filter((r) => r.provenance.submitted_prompt_hash_matches_baseline === true).length === 186);
ok('rubric_hash matched baseline on all 186',
   norm.filter((r) => r.provenance.rubric_hash_matches_baseline === true).length === 186);
ok('no agent rerun, no command execution',
   norm.every((r) => r.provenance.reran_agent === false && r.provenance.executed_commands === false));

// The Gateway returns a slug only. Equal slugs do NOT prove equal model builds.
for (const e of EVALUATORS) {
  const c = an.per_model[e.key].replay_checks;
  ok(`${e.key}: routed slug equals baseline slug (${c.observed_model_ids.join(',') || 'none'})`,
     JSON.stringify(c.observed_model_ids) === JSON.stringify(c.baseline_model_ids));
}
console.log('NOTE  the Gateway exposes model slugs only, never an immutable version or build id,');
console.log('NOTE  so identical model builds between the baseline and this replay CANNOT be proven.');

console.log('\n== C. headline figures quoted in REPORT.md ==');
ok('jev 31 stable / 0 changed',    an.per_model.jev.stable_all_three === 31 && an.per_model.jev.changed_all_three === 0);
ok('gemini 28 stable / 3 changed', an.per_model.gemini.stable_all_three === 28 && an.per_model.gemini.changed_all_three === 3);
ok('openai 28 stable / 3 changed', an.per_model.openai.stable_all_three === 28 && an.per_model.openai.changed_all_three === 3);
ok('cohorts 20 unanimous / 11 disputed',
   an.cohort_rollup.unanimous.spans === 20 && an.cohort_rollup.disputed.spans === 11);
ok('jev probability medians 0.93 / 0.74',
   an.cohort_rollup.unanimous.jev_probabilities.chosen_prob_median_of_medians === 0.93
   && an.cohort_rollup.disputed.jev_probabilities.chosen_prob_median_of_medians === 0.74);
ok('jev max probability spread 0.06', an.jev_probabilities_overall.chosen_prob_spread_max === 0.06);
ok('6 spans changed for >=1 model', an.unstable_spans.length === 6);
ok('jev in no unstable span',         an.unstable_spans.every((s) => !s.models.jev));
ok('costs: jev $0 / gemini 0.0251527 / openai 0.01632445',
   an.per_model.jev.cost.total_usd === 0 && an.per_model.gemini.cost.total_usd === 0.0251527
   && an.per_model.openai.cost.total_usd === 0.01632445);

console.log(`\n${failures === 0 ? 'ALL CHECKS PASSED' : failures + ' CHECK(S) FAILED'}`);
process.exit(failures === 0 ? 0 : 1);
