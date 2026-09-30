import test from 'node:test';
import assert from 'node:assert/strict';
import { loadRubric, renderSpanBlock, renderSubmittedPrompt, buildJevQuestions, RISK_LEVELS } from '../src/rubric.js';
import { redactString, redactEnvSnapshot } from '../src/redact.js';
import { buildSpan, classifyOutcome } from '../src/trace.js';
import { evaluateSpanStub } from './helpers/stub-evaluator.js';
import { buildResponseSchema } from '../src/evaluators/generative.js';
import { summarize } from '../src/summarize.js';
import { EVALUATORS } from '../src/config.js';

const rubric = loadRubric();
const scenario = { id: 'sc-test', title: 'T', design_band: 'benign', requested_action: 'do a thing', prompt: 'p', fixture_agent_claim: 'I did the thing.' };
const step = { step_id: 'call_1', command: 'cat .env', description: 'read config', guardrail: 'allow' };
const span = buildSpan({
  scenario,
  step,
  index: 0,
  execution: { executed: true, error_type: null, exit_code: 0, stdout: 'API_TOKEN=sk-abcdefghijklmnopqrst', stderr: '', started_at: '2026-09-21T00:00:00.000Z', ended_at: '2026-09-21T00:00:01.000Z', duration_ms: 1000 },
});

test('rubric exposes exactly the four risk levels and no forbidden fields', () => {
  assert.deepEqual(rubric.risk_levels, ['low', 'medium', 'high', 'critical']);
  assert.deepEqual(rubric.outputs.common, ['risk_level']);
  assert.deepEqual(rubric.outputs.generative_only, ['rationale']);
  const text = JSON.stringify(rubric.shared_prompt);
  for (const forbidden of ['risk_score', 'primary_category', 'execution_outcome', 'impact_status', 'is_irreversible']) {
    assert.ok(!text.includes(forbidden), `shared prompt must not mention ${forbidden}`);
  }
});

test('jev is asked one choice question and never asked for a rationale', () => {
  const q = buildJevQuestions(rubric);
  assert.deepEqual(Object.keys(q), ['risk_level']);
  assert.equal(q.risk_level.type, 'choice');
  assert.deepEqual(Object.keys(q.risk_level.criteria), RISK_LEVELS);
  assert.equal(q.risk_level.instructions, rubric.shared_prompt);
});

test('generative schema returns exactly risk_level and rationale', () => {
  const shape = buildResponseSchema(rubric).shape;
  assert.deepEqual(Object.keys(shape).sort(), ['rationale', 'risk_level']);
});

test('the span evidence block follows the fixed template', () => {
  const block = renderSpanBlock(rubric, span);
  assert.ok(block.startsWith('Command: cat .env'));
  assert.ok(block.includes('Raw tool input:'));
  assert.ok(block.includes('Tool output, if present:'));
  assert.ok(block.includes('Error type, if present: none'));
  assert.ok(block.endsWith('Classify this span.'));
});

test('credentials are redacted from captured output', () => {
  assert.ok(!span.tool_output.includes('sk-abcdefghijklmnopqrst'));
  assert.match(span.tool_output, /REDACTED/);
  assert.equal(redactString('AKIAIOSFODNN7EXAMPLE'), '[REDACTED:aws-access-key-id]');
  assert.equal(redactEnvSnapshot({ MY_TOKEN: 'abc', PATH: '/bin' }).MY_TOKEN, '[REDACTED:sensitive-key]');
});

test('refusal, guardrail denial and execution are distinguished', () => {
  const denied = buildSpan({
    scenario,
    step: { ...step, step_id: 'call_2', guardrail: 'deny' },
    index: 1,
    execution: { executed: false, error_type: 'denied', exit_code: null, stdout: '', stderr: '' },
  });
  assert.equal(denied.tool_output, null);
  assert.equal(denied.error_type, 'denied');
  const outcome = classifyOutcome('I will not disable that check.', [span, denied]);
  assert.equal(outcome.spans_executed, 1);
  assert.equal(outcome.spans_denied_by_guardrail, 1);
  assert.equal(outcome.agent_refused, true);
  assert.equal(classifyOutcome('I did it.', [span]).agent_refused, false);
});

test('all three evaluators receive byte-identical submitted text', async () => {
  const trace = { trace_id: 't1', scenario_id: 'sc-test', design_band: 'benign', title: 'T', outcome: {}, spans: [span] };
  const runMeta = { run_marker: 'rm1' };
  const results = [];
  for (const evaluator of EVALUATORS) {
    const out = await evaluateSpanStub({ rubric, span, trace, runMeta, evaluator });
    results.push(out.normalized);
  }
  const hashes = new Set(results.map((r) => r.submitted.submitted_prompt_hash));
  assert.equal(hashes.size, 1);
  assert.equal(results[0].submitted.span_block, renderSpanBlock(rubric, span));
  assert.ok(renderSubmittedPrompt(rubric, span).startsWith(rubric.shared_prompt));
});

test('jev normalizes to a null rationale with an unavailable marker', async () => {
  const trace = { trace_id: 't1', scenario_id: 'sc-test', design_band: 'benign', title: 'T', outcome: {}, spans: [span] };
  const runMeta = { run_marker: 'rm1' };
  const jev = await evaluateSpanStub({ rubric, span, trace, runMeta, evaluator: EVALUATORS[0] });
  assert.equal(jev.normalized.evaluator, 'jev');
  assert.equal(jev.normalized.rationale, null);
  assert.equal(jev.normalized.rationale_availability, 'unsupported_by_evaluator');
  assert.ok(RISK_LEVELS.includes(jev.normalized.risk_level));

  const gemini = await evaluateSpanStub({ rubric, span, trace, runMeta, evaluator: EVALUATORS[1] });
  assert.equal(gemini.normalized.rationale_availability, 'provided');
  assert.equal(typeof gemini.normalized.rationale, 'string');
});

test('unavailable metrics stay null rather than being inferred', async () => {
  const trace = { trace_id: 't1', scenario_id: 'sc-test', design_band: 'benign', title: 'T', outcome: {}, spans: [span] };
  const out = await evaluateSpanStub({ rubric, span, trace, runMeta: { run_marker: 'rm1' }, evaluator: EVALUATORS[2] });
  assert.equal(out.normalized.cost.estimated_usd, null);
  assert.equal(out.normalized.probabilities, null);
  assert.equal(out.normalized.simulated, true);
});

test('scenario-level risk is derived locally and labelled as not a model response', async () => {
  const trace = { trace_id: 't1', scenario_id: 'sc-test', design_band: 'benign', title: 'T', outcome: { spans_total: 1 }, spans: [span] };
  const runMeta = { run_marker: 'rm1' };
  const results = [];
  for (const evaluator of EVALUATORS) {
    results.push((await evaluateSpanStub({ rubric, span, trace, runMeta, evaluator })).normalized);
  }
  const summary = summarize({ traces: [trace], results, evalMeta: { run_marker: 'rm1', mode: 'dry' } });
  assert.equal(summary.scenarios[0].derived_note, 'derived_aggregate_not_a_model_response');
  assert.equal(summary.totals.identical_submitted_prompt_hashes.all_identical, true);
  assert.ok(['unanimous', 'split', 'three_way'].includes(summary.spans[0].agreement));
});

test('sandbox credentials are all-or-nothing and fall through to OIDC when unset', async () => {
  const { sandboxCredentials } = await import('../src/executors/copilot-sandbox.js');
  assert.deepEqual(sandboxCredentials({}), {});
  assert.deepEqual(
    sandboxCredentials({ VERCEL_TOKEN: 't', VERCEL_TEAM_ID: 'team', VERCEL_PROJECT_ID: 'proj' }),
    { token: 't', teamId: 'team', projectId: 'proj' },
  );
  assert.throws(() => sandboxCredentials({ VERCEL_TOKEN: 't' }), /must be set together/);
});

test('static UI export writes the same dataset the local server serves', async () => {
  const fs = await import('node:fs');
  const os = await import('node:os');
  const path = await import('node:path');
  const { buildStaticUi } = await import('../ui/build.js');
  const { loadDataset } = await import('../ui/dataset.js');
  const out = fs.mkdtempSync(path.join(os.tmpdir(), 'jev-ui-'));
  const r = buildStaticUi(out);
  for (const f of ['index.html', 'app.js', 'styles.css', 'data.json']) {
    assert.ok(fs.existsSync(path.join(out, f)), `${f} exported`);
  }
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(out, 'data.json'), 'utf8')), loadDataset());
  assert.equal(r.traces, loadDataset().traces.length);
  fs.rmSync(out, { recursive: true, force: true });
});
