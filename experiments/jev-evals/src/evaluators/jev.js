import { experimental_evaluate as evaluate } from 'ai';
import { gateway } from '@ai-sdk/gateway';
import { buildJevQuestions, renderSpanBlock, renderSubmittedPrompt } from '../rubric.js';
import { estimateCost, extractProviderCost } from './pricing.js';

/**
 * Jev is used strictly as a classifier: one typed choice question over the four
 * risk levels, and no rationale is requested. Its typed answer - including the
 * choice probabilities when it returns them - is preserved as Jev data.
 */
export async function evaluateWithJev({ rubric, span }) {
  const spanBlock = renderSpanBlock(rubric, span);
  const submittedPrompt = renderSubmittedPrompt(rubric, span);
  const questions = buildJevQuestions(rubric);

  const started = Date.now();
  const result = await evaluate({
    model: gateway.evaluationModel('typesafe-ai/jev'),
    state: spanBlock,
    questions,
  });
  const latencyMs = Date.now() - started;

  const answer = result.answers.risk_level;
  const usage = {
    input_tokens: result.usage?.inputTokens ?? null,
    output_tokens: result.usage?.outputTokens ?? null,
    total_tokens: result.usage?.totalTokens ?? null,
  };

  return {
    spanBlock,
    submittedPrompt,
    riskLevel: answer?.choice ?? null,
    // Jev does not produce free text under this contract; never fabricate one.
    rationale: null,
    rationaleAvailability: 'unsupported_by_evaluator',
    probabilities: answer?.probabilities ?? null,
    latencyMs,
    usage,
    cost: estimateCost({
      model: 'typesafe-ai/jev',
      usage,
      providerCostUsd: extractProviderCost(result.providerMetadata),
    }),
    warnings: result.warnings ?? null,
    raw: {
      answers: result.answers,
      usage: result.usage ?? null,
      warnings: result.warnings ?? null,
      rounding: result.rounding ?? null,
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
