import { EVALUATORS } from './config.js';
import { RISK_LEVELS } from './rubric.js';

const RANK = Object.fromEntries(RISK_LEVELS.map((l, i) => [l, i]));

/**
 * Build the summary dataset from preserved artifacts only.
 *
 * Everything here is a local computation over stored model answers. Derived
 * values (for example a scenario's highest span risk) are labelled as derived
 * and are never presented as a model response.
 */
export function summarize({ traces, results, evalMeta }) {
  const bySpan = new Map();
  for (const r of results) {
    if (!bySpan.has(r.span_id)) bySpan.set(r.span_id, {});
    bySpan.get(r.span_id)[r.evaluator] = r;
  }

  const spanRows = [];
  for (const trace of traces) {
    for (const span of trace.spans) {
      const per = bySpan.get(span.span_id) ?? {};
      const levels = EVALUATORS.map((e) => per[e.key]?.risk_level ?? null);
      const answered = levels.filter(Boolean);
      const distinct = [...new Set(answered)];
      spanRows.push({
        span_id: span.span_id,
        trace_id: trace.trace_id,
        scenario_id: trace.scenario_id,
        step_id: span.step_id,
        command: span.command,
        error_type: span.error_type,
        executed: span.executed,
        exit_code: span.exit_code,
        // Author's design band, for reference only - not ground truth.
        scenario_design_band: trace.design_band,
        levels: Object.fromEntries(EVALUATORS.map((e) => [e.key, per[e.key]?.risk_level ?? null])),
        agreement:
          answered.length < EVALUATORS.length
            ? 'incomplete'
            : distinct.length === 1
              ? 'unanimous'
              : distinct.length === 2
                ? 'split'
                : 'three_way',
        max_level_spread:
          answered.length > 1
            ? Math.max(...answered.map((l) => RANK[l])) - Math.min(...answered.map((l) => RANK[l]))
            : null,
        result_ids: Object.fromEntries(EVALUATORS.map((e) => [e.key, per[e.key]?.result_id ?? null])),
      });
    }
  }

  const perEvaluator = {};
  for (const e of EVALUATORS) {
    const rows = results.filter((r) => r.evaluator === e.key);
    const ok = rows.filter((r) => !r.error);
    const latencies = ok.map((r) => r.latency_ms).filter((v) => typeof v === 'number').sort((a, b) => a - b);
    const costs = ok.map((r) => r.cost?.estimated_usd).filter((v) => typeof v === 'number');
    const rationales = ok.filter((r) => r.rationale_availability === 'provided');
    perEvaluator[e.key] = {
      model: e.model,
      kind: e.kind,
      calls: rows.length,
      errors: rows.filter((r) => r.error).length,
      distribution: countBy(ok.map((r) => r.risk_level)),
      latency_ms: latencies.length
        ? { p50: quantile(latencies, 0.5), p95: quantile(latencies, 0.95), mean: mean(latencies) }
        : null,
      usage_totals: sumUsage(ok),
      estimated_cost_usd: costs.length ? Number(costs.reduce((a, b) => a + b, 0).toFixed(8)) : null,
      cost_unavailable_reason: costs.length
        ? null
        : (ok.find((r) => r.cost?.unavailable_reason)?.cost.unavailable_reason ?? 'not reported'),
      probabilities_available: ok.some((r) => r.probabilities),
      rationale_capability:
        e.kind === 'typed' ? 'not_requested_classifier_only' : 'requested_and_returned',
      rationale_count: rationales.length,
      rationale_length_chars: rationales.length
        ? {
            mean: Math.round(mean(rationales.map((r) => r.rationale.length))),
            min: Math.min(...rationales.map((r) => r.rationale.length)),
            max: Math.max(...rationales.map((r) => r.rationale.length)),
          }
        : null,
    };
  }

  const pairwise = {};
  for (let i = 0; i < EVALUATORS.length; i++) {
    for (let j = i + 1; j < EVALUATORS.length; j++) {
      const a = EVALUATORS[i].key;
      const b = EVALUATORS[j].key;
      const both = spanRows.filter((s) => s.levels[a] && s.levels[b]);
      const same = both.filter((s) => s.levels[a] === s.levels[b]).length;
      pairwise[`${a}|${b}`] = {
        compared_spans: both.length,
        exact_agreement: both.length ? Number((same / both.length).toFixed(4)) : null,
        disagreement_spans: both.filter((s) => s.levels[a] !== s.levels[b]).map((s) => s.span_id),
      };
    }
  }

  const scenarioRows = traces.map((trace) => {
    const rows = spanRows.filter((s) => s.scenario_id === trace.scenario_id);
    const derivedMax = {};
    for (const e of EVALUATORS) {
      const levels = rows.map((r) => r.levels[e.key]).filter(Boolean);
      derivedMax[e.key] = levels.length
        ? RISK_LEVELS[Math.max(...levels.map((l) => RANK[l]))]
        : null;
    }
    return {
      scenario_id: trace.scenario_id,
      trace_id: trace.trace_id,
      title: trace.title,
      design_band: trace.design_band,
      spans: rows.length,
      outcome: trace.outcome,
      // Derived locally by taking the highest span risk level. Not a model
      // response: no evaluator was ever shown the whole scenario.
      derived_highest_span_risk: derivedMax,
      derived_note: 'derived_aggregate_not_a_model_response',
      unanimous_spans: rows.filter((r) => r.agreement === 'unanimous').length,
    };
  });

  return {
    generated_at: new Date().toISOString(),
    run_marker: evalMeta?.run_marker ?? null,
    mode: evalMeta?.mode ?? null,
    simulated: evalMeta?.simulated ?? null,
    rubric: {
      rubric_id: evalMeta?.rubric_id ?? null,
      rubric_version: evalMeta?.rubric_version ?? null,
      rubric_hash: evalMeta?.rubric_hash ?? null,
      shared_prompt_hash: evalMeta?.shared_prompt_hash ?? null,
    },
    totals: {
      scenarios: traces.length,
      spans: spanRows.length,
      evaluators: EVALUATORS.length,
      calls: results.length,
      errors: results.filter((r) => r.error).length,
      identical_submitted_prompt_hashes: identicalSubmissions(results),
    },
    per_evaluator: perEvaluator,
    pairwise_agreement: pairwise,
    agreement_distribution: countBy(spanRows.map((s) => s.agreement)),
    spans: spanRows,
    scenarios: scenarioRows,
  };
}

/** Audit check: all three evaluators must have received identical input text. */
function identicalSubmissions(results) {
  const bySpan = new Map();
  for (const r of results) {
    const set = bySpan.get(r.span_id) ?? new Set();
    set.add(r.submitted?.submitted_prompt_hash ?? 'missing');
    bySpan.set(r.span_id, set);
  }
  const mismatched = [...bySpan.entries()].filter(([, set]) => set.size > 1).map(([id]) => id);
  return { spans_checked: bySpan.size, mismatched_spans: mismatched, all_identical: mismatched.length === 0 };
}

function countBy(values) {
  const out = {};
  for (const v of values) out[v ?? 'null'] = (out[v ?? 'null'] ?? 0) + 1;
  return out;
}
function mean(xs) {
  return xs.length ? Number((xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(2)) : null;
}
function quantile(sorted, q) {
  if (!sorted.length) return null;
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}
function sumUsage(rows) {
  const keys = ['input_tokens', 'output_tokens', 'total_tokens'];
  const out = {};
  for (const k of keys) {
    const vals = rows.map((r) => r.usage?.[k]).filter((v) => typeof v === 'number');
    out[k] = vals.length ? vals.reduce((a, b) => a + b, 0) : null;
  }
  return out;
}
