import fs from 'node:fs';
const R = process.cwd(); // run from the repo root
const { EVALUATORS } = await import(`${R}/src/config.js`);
const { readJsonl, readJson, writeJson } = await import(`${R}/src/util.js`);

const OUT = `${R}/artifacts/repeatability`;
const RANK = { low: 0, medium: 1, high: 2, critical: 3 };
const med = (a) => { if (!a.length) return null; const s=[...a].sort((x,y)=>x-y); const m=s.length>>1;
  return s.length%2 ? s[m] : (s[m-1]+s[m])/2; };
const r4 = (v) => v == null ? null : Number(v.toFixed(4));

const baseNorm = readJsonl(`${R}/artifacts/evaluations.normalized.jsonl`);
const rep      = readJsonl(`${OUT}/repeat.normalized.jsonl`);
const repMeta  = readJson(`${OUT}/repeat-meta.json`);
const summary  = readJson(`${R}/artifacts/summary.json`);
const traces   = readJsonl(`${R}/artifacts/traces.jsonl`);
const spans    = traces.flatMap(t => t.spans.map(s => ({ ...s, trace_id: t.trace_id })));

// Baseline agreement cohort per span, taken from the existing summary dataset.
const spanRows = summary.spans ?? summary.span_rows ?? [];
const cohortOf = Object.fromEntries(spanRows.map(s => [s.span_id, s.agreement]));

const perSpan = [];
for (const span of spans) {
  const row = { span_id: span.span_id, scenario_id: span.scenario_id, command: span.command,
    baseline_cohort: cohortOf[span.span_id] ?? 'unknown', models: {} };
  for (const e of EVALUATORS) {
    const b  = baseNorm.find(r => r.span_id === span.span_id && r.evaluator === e.key);
    const r1 = rep.find(r => r.span_id === span.span_id && r.evaluator === e.key && r.replicate === 1);
    const r2 = rep.find(r => r.span_id === span.span_id && r.evaluator === e.key && r.replicate === 2);
    const labels = { baseline: b?.risk_level ?? null, r1: r1?.risk_level ?? null, r2: r2?.risk_level ?? null };
    const answered = Object.values(labels).filter(Boolean);
    const distinct = [...new Set(answered)];
    row.models[e.key] = {
      labels,
      errors: [b,r1,r2].filter(x => x?.error).length,
      repeats_agree: (labels.r1 && labels.r2) ? labels.r1 === labels.r2 : null,
      all_three_stable: answered.length === 3 ? distinct.length === 1 : null,
      repeats_match_baseline: (labels.r1 && labels.r2 && labels.baseline)
        ? (labels.r1 === labels.baseline && labels.r2 === labels.baseline) : null,
      level_spread: answered.length > 1
        ? Math.max(...answered.map(l => RANK[l])) - Math.min(...answered.map(l => RANK[l])) : null,
      // What the model said, per observation. Jev returns no prose; its
      // probabilities below carry the same job.
      rationales: { baseline: b?.rationale ?? null, r1: r1?.rationale ?? null, r2: r2?.rationale ?? null },
      rationale_availability: b?.rationale_availability ?? r1?.rationale_availability ?? null,
      // Jev only: calibrated choice probabilities.
      probabilities: e.key === 'jev'
        ? { baseline: b?.probabilities ?? null, r1: r1?.probabilities ?? null, r2: r2?.probabilities ?? null }
        : null,
    };
  }
  // Jev chosen-label probability across the three observations.
  const j = row.models.jev;
  if (j?.probabilities) {
    const pts = ['baseline','r1','r2'].map(k => {
      const lvl = j.labels[k], p = j.probabilities[k];
      return (lvl && p && p[lvl] != null) ? p[lvl] : null;
    }).filter(v => v != null);
    j.chosen_prob = { points: pts, min: pts.length?Math.min(...pts):null, max: pts.length?Math.max(...pts):null,
      spread: pts.length?r4(Math.max(...pts)-Math.min(...pts)):null, median: r4(med(pts)) };
    // Per-level max absolute movement across the three observations.
    const lv = ['low','medium','high','critical'];
    const moves = lv.map(l => { const v = ['baseline','r1','r2'].map(k => j.probabilities[k]?.[l])
      .filter(x => x != null); return v.length>1 ? Math.max(...v)-Math.min(...v) : null; }).filter(x=>x!=null);
    j.max_level_prob_movement = moves.length ? r4(Math.max(...moves)) : null;
  }
  perSpan.push(row);
}

const perModel = {};
for (const e of EVALUATORS) {
  const rows = rep.filter(r => r.evaluator === e.key);
  const baseRows = baseNorm.filter(r => r.evaluator === e.key);
  const m = perSpan.map(s => s.models[e.key]);
  const usable = m.filter(x => x.all_three_stable !== null);
  const tokens = rows.reduce((a,r) => a + (r.usage?.total_tokens ?? 0), 0);
  const inTok  = rows.reduce((a,r) => a + (r.usage?.input_tokens ?? 0), 0);
  const outTok = rows.reduce((a,r) => a + (r.usage?.output_tokens ?? 0), 0);
  const costs  = rows.map(r => r.cost?.estimated_usd).filter(v => typeof v === 'number');
  const bases  = [...new Set(rows.map(r => r.cost?.basis).filter(Boolean))];
  const lat    = rows.map(r => r.latency_ms).filter(v => typeof v === 'number');
  perModel[e.key] = {
    model: e.model, kind: e.kind,
    repeat_calls: rows.length,
    errors: rows.filter(r => r.error).length,
    spans_scored: usable.length,
    stable_all_three: usable.filter(x => x.all_three_stable).length,
    changed_all_three: usable.filter(x => x.all_three_stable === false).length,
    repeats_agree_with_each_other: m.filter(x => x.repeats_agree === true).length,
    repeats_both_match_baseline: m.filter(x => x.repeats_match_baseline === true).length,
    max_level_spread_observed: Math.max(0, ...m.map(x => x.level_spread ?? 0)),
    spans_with_spread_ge_2: m.filter(x => (x.level_spread ?? 0) >= 2).length,
    usage: { input_tokens: inTok, output_tokens: outTok, total_tokens: tokens },
    cost: { total_usd: costs.length ? Number(costs.reduce((a,b)=>a+b,0).toFixed(8)) : null,
            basis: bases, calls_with_cost: costs.length },
    latency_ms: { median: med(lat), min: lat.length?Math.min(...lat):null, max: lat.length?Math.max(...lat):null },
    baseline_reference: {
      calls: baseRows.length,
      errors: baseRows.filter(r => r.error).length,
      total_tokens: baseRows.reduce((a,r) => a + (r.usage?.total_tokens ?? 0), 0),
      cost_usd: (() => { const c = baseRows.map(r => r.cost?.estimated_usd).filter(v=>typeof v==='number');
        return c.length ? Number(c.reduce((a,b)=>a+b,0).toFixed(8)) : null; })(),
    },
    replay_checks: {
      prompt_hash_matches_baseline: rows.filter(r => r.provenance.submitted_prompt_hash_matches_baseline === true).length,
      rubric_hash_matches_baseline: rows.filter(r => r.provenance.rubric_hash_matches_baseline === true).length,
      observed_model_ids: repMeta.per_model?.[e.key]?.observed_model_ids ?? [],
      baseline_model_ids: repMeta.per_model?.[e.key]?.baseline_model_ids ?? [],
    },
  };
}

// Jev probability behaviour, overall and by baseline cohort.
const jevProb = (subset) => {
  const rows = subset.map(s => s.models.jev).filter(x => x?.chosen_prob?.points?.length);
  const spreads = rows.map(x => x.chosen_prob.spread).filter(v => v != null);
  const medians = rows.map(x => x.chosen_prob.median).filter(v => v != null);
  const moves   = rows.map(x => x.max_level_prob_movement).filter(v => v != null);
  return { spans: rows.length,
    chosen_prob_median_of_medians: r4(med(medians)),
    chosen_prob_spread_median: r4(med(spreads)),
    chosen_prob_spread_max: spreads.length ? r4(Math.max(...spreads)) : null,
    spans_with_zero_spread: spreads.filter(v => v === 0).length,
    max_level_prob_movement_median: r4(med(moves)),
    max_level_prob_movement_max: moves.length ? r4(Math.max(...moves)) : null };
};

const cohorts = {};
for (const c of ['unanimous','split','three_way','incomplete','unknown']) {
  const subset = perSpan.filter(s => s.baseline_cohort === c);
  if (!subset.length) continue;
  cohorts[c] = { spans: subset.length,
    per_model: Object.fromEntries(EVALUATORS.map(e => {
      const m = subset.map(s => s.models[e.key]).filter(x => x.all_three_stable !== null);
      return [e.key, { spans_scored: m.length, stable: m.filter(x => x.all_three_stable).length,
                       changed: m.filter(x => x.all_three_stable === false).length }];
    })),
    jev_probabilities: jevProb(subset) };
}
// The blog's cohort language groups every non-unanimous span as disputed.
const disputed = perSpan.filter(s => ['split','three_way'].includes(s.baseline_cohort));
const unanimous = perSpan.filter(s => s.baseline_cohort === 'unanimous');

const analysis = {
  study: 'three-model-repeatability',
  repeat_run_marker: repMeta.repeat_run_marker,
  baseline_run_marker: repMeta.baseline_run_marker,
  spans: perSpan.length,
  replicates_per_span_per_model: repMeta.replicates_per_span_per_model,
  observations_per_span_per_model: 3,
  total_repeat_calls: rep.length,
  per_model: perModel,
  baseline_cohorts: cohorts,
  cohort_rollup: {
    unanimous: { spans: unanimous.length, jev_probabilities: jevProb(unanimous),
      per_model: Object.fromEntries(EVALUATORS.map(e => [e.key,
        { stable: unanimous.filter(s => s.models[e.key].all_three_stable).length,
          changed: unanimous.filter(s => s.models[e.key].all_three_stable === false).length }])) },
    disputed: { spans: disputed.length, jev_probabilities: jevProb(disputed),
      per_model: Object.fromEntries(EVALUATORS.map(e => [e.key,
        { stable: disputed.filter(s => s.models[e.key].all_three_stable).length,
          changed: disputed.filter(s => s.models[e.key].all_three_stable === false).length }])) },
  },
  jev_probabilities_overall: jevProb(perSpan),
  unstable_spans: perSpan.filter(s => EVALUATORS.some(e => s.models[e.key].all_three_stable === false))
    .map(s => ({ span_id: s.span_id, scenario_id: s.scenario_id, baseline_cohort: s.baseline_cohort,
      command: s.command.slice(0, 120),
      models: Object.fromEntries(EVALUATORS.filter(e => s.models[e.key].all_three_stable === false)
        .map(e => [e.key, s.models[e.key].labels])) })),
  config_notes: repMeta.config_notes,
  created_at: new Date().toISOString(),
};

writeJson(`${OUT}/repeat-analysis.json`, analysis);
writeJson(`${OUT}/repeat-per-span.json`, perSpan);
console.log(JSON.stringify({ per_model: perModel, cohort_rollup: analysis.cohort_rollup,
  jev_overall: analysis.jev_probabilities_overall, unstable: analysis.unstable_spans }, null, 2));
