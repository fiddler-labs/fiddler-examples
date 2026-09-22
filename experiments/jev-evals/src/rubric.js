import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from './config.js';
import { readJson, sha256, stableStringify } from './util.js';

export const RUBRIC_DIR = path.join(ROOT, 'rubric');
export const RUBRIC_FILE = path.join(RUBRIC_DIR, 'rubric.v2.json');

export const RISK_LEVELS = ['low', 'medium', 'high', 'critical'];

/**
 * The rubric is loaded once and hashed, including the shared prompt text, so
 * every stored result can be checked against the exact rubric that produced it.
 */
export function loadRubric(file = RUBRIC_FILE) {
  const rubric = readJson(file);
  const sharedPrompt = fs.readFileSync(path.join(path.dirname(file), rubric.shared_prompt_file), 'utf8').trimEnd();
  const withPrompt = { ...rubric, shared_prompt: sharedPrompt };
  return {
    ...withPrompt,
    shared_prompt_hash: sha256(sharedPrompt),
    rubric_hash: sha256(stableStringify(withPrompt)),
  };
}

/**
 * Render the span evidence block. The wording and spacing of the template are
 * fixed by the rubric so all three evaluators see byte-identical evidence.
 */
export function renderSpanBlock(rubric, span) {
  const value = (v) => (v === null || v === undefined || v === '' ? 'none' : typeof v === 'string' ? v : JSON.stringify(v));
  return rubric.span_template
    .replace('{{command}}', value(span.command))
    .replace('{{tool_input}}', value(span.tool_input))
    .replace('{{tool_output}}', value(span.tool_output))
    .replace('{{error_type}}', value(span.error_type));
}

/** The full text submitted to every evaluator, used for auditing equality. */
export function renderSubmittedPrompt(rubric, span) {
  return `${rubric.shared_prompt}\n\n${renderSpanBlock(rubric, span)}`;
}

/** Jev: a single typed choice question over the four risk levels. */
export function buildJevQuestions(rubric) {
  return {
    risk_level: {
      type: 'choice',
      instructions: rubric.shared_prompt,
      criteria: Object.fromEntries(
        RISK_LEVELS.map((level) => [level, rubric.risk_level_definitions[level]]),
      ),
    },
  };
}
