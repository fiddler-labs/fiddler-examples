import { sha256 } from '../util.js';

export const RESULT_SCHEMA_VERSION = '2.0.0';

/**
 * One shared normalized result row per (span, evaluator).
 *
 * `risk_level` is the only field every evaluator produces. `rationale` is
 * nullable and carries an explicit availability marker so a missing rationale is
 * never read as a failure and never fabricated.
 */
export function normalizeResult({
  runMeta,
  span,
  trace,
  evaluator,
  rubric,
  submittedPrompt,
  spanBlock,
  riskLevel,
  rationale,
  rationaleAvailability,
  probabilities,
  latencyMs,
  usage,
  cost,
  warnings,
  error,
}) {
  return {
    schema_version: RESULT_SCHEMA_VERSION,
    result_id: `${runMeta.run_marker}:${span.span_id}:${evaluator.key}`,
    run_marker: runMeta.run_marker,
    trace_id: trace.trace_id,
    scenario_id: trace.scenario_id,
    span_id: span.span_id,
    step_id: span.step_id,
    evaluator: evaluator.key,
    model: evaluator.model,
    evaluator_kind: evaluator.kind,
    rubric: {
      rubric_id: rubric.rubric_id,
      rubric_version: rubric.rubric_version,
      rubric_hash: rubric.rubric_hash,
      shared_prompt_hash: rubric.shared_prompt_hash,
    },
    submitted: {
      span_evidence_hash: span.span_evidence_hash,
      submitted_prompt_hash: sha256(submittedPrompt),
      span_block: spanBlock,
    },
    risk_level: riskLevel ?? null,
    rationale: rationale ?? null,
    rationale_availability: rationaleAvailability,
    // Typed choice probabilities, only when the evaluator actually returns them.
    probabilities: probabilities ?? null,
    latency_ms: latencyMs ?? null,
    usage: usage ?? { input_tokens: null, output_tokens: null, total_tokens: null },
    cost: cost ?? { estimated_usd: null, basis: null },
    warnings: warnings ?? null,
    error: error ?? null,
    created_at: new Date().toISOString(),
  };
}
