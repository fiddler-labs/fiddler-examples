import { generateObject } from 'ai';
import { gateway } from '@ai-sdk/gateway';
import { z } from 'zod';
import { RISK_LEVELS, renderSpanBlock, renderSubmittedPrompt } from '../rubric.js';
import { estimateCost, extractProviderCost } from './pricing.js';

/**
 * Both generative peers receive the identical shared rubric and identical span
 * evidence, and return the common `risk_level` plus a short evidence-grounded
 * rationale. Nothing outside those two fields is requested.
 */
export function buildResponseSchema(rubric) {
  return z.object({
    risk_level: z.enum(RISK_LEVELS).describe('Exactly one risk level from the shared rubric.'),
    rationale: z.string().describe(rubric.outputs.rationale_schema_instruction),
  });
}

export async function evaluateWithGenerative({ rubric, span, evaluator }) {
  const spanBlock = renderSpanBlock(rubric, span);
  const submittedPrompt = renderSubmittedPrompt(rubric, span);

  const started = Date.now();
  const result = await generateObject({
    model: gateway(evaluator.model),
    schema: buildResponseSchema(rubric),
    system: rubric.shared_prompt,
    prompt: spanBlock,
    // Set per evaluator in config.js; absent for an evaluator that declares none,
    // so the call path is otherwise unchanged from the first run.
    ...(evaluator.providerOptions ? { providerOptions: evaluator.providerOptions } : {}),
  });
  const latencyMs = Date.now() - started;

  const usage = {
    input_tokens: result.usage?.inputTokens ?? null,
    output_tokens: result.usage?.outputTokens ?? null,
    total_tokens: result.usage?.totalTokens ?? null,
  };

  return {
    spanBlock,
    submittedPrompt,
    riskLevel: result.object?.risk_level ?? null,
    rationale: result.object?.rationale ?? null,
    rationaleAvailability: result.object?.rationale ? 'provided' : 'unavailable_empty',
    // Generative models return no calibrated choice probabilities here.
    probabilities: null,
    latencyMs,
    usage,
    cost: estimateCost({
      model: evaluator.model,
      usage,
      providerCostUsd: extractProviderCost(result.providerMetadata),
    }),
    warnings: result.warnings ?? null,
    raw: {
      object: result.object ?? null,
      finishReason: result.finishReason ?? null,
      usage: result.usage ?? null,
      warnings: result.warnings ?? null,
      providerMetadata: result.providerMetadata ?? null,
      response: result.response
        ? {
            id: result.response.id ?? null,
            modelId: result.response.modelId ?? null,
            timestamp: result.response.timestamp ? new Date(result.response.timestamp).toISOString() : null,
          }
        : null,
    },
  };
}
