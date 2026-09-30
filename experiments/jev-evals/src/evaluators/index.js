import { EVALUATORS } from '../config.js';
import { renderSpanBlock, renderSubmittedPrompt } from '../rubric.js';
import { evaluateWithJev } from './jev.js';
import { evaluateWithGenerative } from './generative.js';
import { normalizeResult } from './schema.js';

export { EVALUATORS };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Per-evaluator pacing. A provider plan can cap a model at a few requests per
 * minute, and a refused call is a hole in the dataset rather than a finding, so
 * the caller waits instead. Only evaluators that declare `minIntervalMs` wait.
 */
const lastCallAt = new Map();
async function paced(evaluator, fn) {
  const gap = evaluator.minIntervalMs ?? 0;
  if (gap) {
    const wait = gap - (Date.now() - (lastCallAt.get(evaluator.key) ?? 0));
    if (wait > 0) await sleep(wait);
  }
  try {
    return await fn();
  } finally {
    if (gap) lastCallAt.set(evaluator.key, Date.now());
  }
}

/** A rate-limited call is retried once the window has passed, not recorded as an error. */
const isRateLimit = (err) =>
  err?.name === 'GatewayRateLimitError' || /rate limit/i.test(err?.message ?? '');

/** Evaluate one span with one evaluator, normalizing both success and failure. */
export async function evaluateSpan({ rubric, span, trace, runMeta, evaluator }) {
  const call = () =>
    evaluator.kind === 'typed'
      ? evaluateWithJev({ rubric, span })
      : evaluateWithGenerative({ rubric, span, evaluator });
  try {
    let out;
    for (let attempt = 1; ; attempt++) {
      try {
        out = await paced(evaluator, call);
        break;
      } catch (err) {
        if (!isRateLimit(err) || attempt > 3) throw err;
        await sleep(60_000);
      }
    }
    return {
      normalized: normalizeResult({ runMeta, span, trace, evaluator, rubric, ...out }),
      raw: rawRow({ runMeta, span, trace, evaluator, rubric, raw: out.raw, error: null }),
    };
  } catch (err) {
    const spanBlock = renderSpanBlock(rubric, span);
    const submittedPrompt = renderSubmittedPrompt(rubric, span);
    const error = { name: err?.name ?? 'Error', message: err?.message ?? String(err) };
    return {
      normalized: normalizeResult({
        runMeta,
        span,
        trace,
        evaluator,
        rubric,
        submittedPrompt,
        spanBlock,
        riskLevel: null,
        rationale: null,
        rationaleAvailability:
          evaluator.kind === 'typed' ? 'unsupported_by_evaluator' : 'unavailable_error',
        probabilities: null,
        latencyMs: null,
        usage: null,
        cost: null,
        warnings: null,
        error,
      }),
      raw: rawRow({ runMeta, span, trace, evaluator, rubric, raw: null, error }),
    };
  }
}

function rawRow({ runMeta, span, trace, evaluator, rubric, raw, error }) {
  return {
    result_id: `${runMeta.run_marker}:${span.span_id}:${evaluator.key}`,
    run_marker: runMeta.run_marker,
    trace_id: trace.trace_id,
    span_id: span.span_id,
    evaluator: evaluator.key,
    model: evaluator.model,
    rubric_version: rubric.rubric_version,
    rubric_hash: rubric.rubric_hash,
    raw_response: raw,
    error,
    created_at: new Date().toISOString(),
  };
}
