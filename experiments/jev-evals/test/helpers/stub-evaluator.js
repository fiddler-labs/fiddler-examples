import { renderSpanBlock, renderSubmittedPrompt, RISK_LEVELS } from '../../src/rubric.js';
import { normalizeResult } from '../../src/evaluators/schema.js';
import { sha256 } from '../../src/util.js';

/**
 * Deterministic offline stand-in, used only by the test suite. It exercises the
 * exact same normalization path as the live evaluators without making any
 * network call.
 *
 * It lives under test/ on purpose: nothing in src/ can reach it, so no code
 * path in the product can produce a fabricated rating. Every row it produces is
 * still marked `simulated: true`.
 */
export async function evaluateSpanStub({ rubric, span, trace, runMeta, evaluator }) {
  const spanBlock = renderSpanBlock(rubric, span);
  const submittedPrompt = renderSubmittedPrompt(rubric, span);
  // Stable pseudo-random choice derived from the span and evaluator, so dry runs
  // are reproducible and still produce agreement and disagreement to render.
  const digest = sha256(`${evaluator.key}:${span.span_evidence_hash}`);
  const level = RISK_LEVELS[parseInt(digest.slice(0, 8), 16) % RISK_LEVELS.length];
  const isTyped = evaluator.kind === 'typed';

  const usage = { input_tokens: 500, output_tokens: isTyped ? 1 : 40, total_tokens: isTyped ? 501 : 540 };
  const normalized = normalizeResult({
    runMeta,
    span,
    trace,
    evaluator,
    rubric,
    submittedPrompt,
    spanBlock,
    riskLevel: level,
    rationale: isTyped ? null : `SIMULATED rationale for ${span.span_id}: derived offline from the span evidence hash, not a model response.`,
    rationaleAvailability: isTyped ? 'unsupported_by_evaluator' : 'provided',
    probabilities: isTyped ? Object.fromEntries(RISK_LEVELS.map((l) => [l, l === level ? 0.7 : 0.1])) : null,
    latencyMs: 1,
    usage,
    cost: { estimated_usd: null, basis: null, unavailable_reason: 'dry run: no provider call was made' },
    warnings: null,
    error: null,
  });
  normalized.simulated = true;

  return {
    normalized,
    raw: {
      result_id: normalized.result_id,
      run_marker: runMeta.run_marker,
      trace_id: trace.trace_id,
      span_id: span.span_id,
      evaluator: evaluator.key,
      model: evaluator.model,
      rubric_version: rubric.rubric_version,
      rubric_hash: rubric.rubric_hash,
      simulated: true,
      raw_response: { note: 'dry run stub, no provider call', level },
      error: null,
      created_at: new Date().toISOString(),
    },
  };
}
