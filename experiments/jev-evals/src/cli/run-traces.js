#!/usr/bin/env node
import path from 'node:path';
import fs from 'node:fs';
import { ARTIFACTS, PATHS, loadDotEnv } from '../config.js';
import { makeRunMarker, writeJsonl, writeJson } from '../util.js';
import { SCENARIOS, SPAN_COUNT } from '../scenarios.js';

loadDotEnv();

const runMarker = process.env.RUN_MARKER || makeRunMarker('jevlive');

const tracesFile = PATHS.traces;
const metaFile = PATHS.runMeta;

console.log(`[traces] one disposable Vercel Sandbox per scenario, real Copilot CLI as the agent (BYOK via AI Gateway, no GitHub auth).`);
console.log(`[traces] The agent chooses its own commands; span counts are whatever it actually did.`);
console.log(`[traces] run_marker=${runMarker} scenarios=${SCENARIOS.length} fixture_spans=${SPAN_COUNT}`);

const { runCopilotSandbox } = await import('../executors/copilot-sandbox.js');
const result = await runCopilotSandbox({
  runMarker,
  onProgress: (p) => console.log(`[traces] ${p.phase} ${p.scenario_id ?? ''} ${p.sandbox_id ?? p.error ?? ''}`.trimEnd()),
});
// Persist the agent's own raw JSONL exactly as retrieved from each sandbox.
const rawDir = path.join(ARTIFACTS, 'raw');
fs.mkdirSync(rawDir, { recursive: true });
for (const trace of result.traces) {
  const raw = trace._raw ?? {};
  delete trace._raw;
  if (raw.agent_events) fs.writeFileSync(path.join(rawDir, `${trace.scenario_id}.agent-events.jsonl`), raw.agent_events);
  if (raw.otel) fs.writeFileSync(path.join(rawDir, `${trace.scenario_id}.copilot-otel.jsonl`), raw.otel);
  if (raw.agent_stderr?.trim()) fs.writeFileSync(path.join(rawDir, `${trace.scenario_id}.agent-stderr.log`), raw.agent_stderr);
}
console.log(`[traces] raw agent artifacts written to ${path.relative(process.cwd(), rawDir)}`);

writeJsonl(tracesFile, result.traces);
writeJson(metaFile, {
  ...result.runMeta,
  scenarios: SCENARIOS.length,
  spans: result.traces.reduce((n, t) => n + t.spans.length, 0),
  traces_file: path.relative(process.cwd(), tracesFile),
});

const spans = result.traces.reduce((n, t) => n + t.spans.length, 0);
const denied = result.traces.reduce((n, t) => n + t.outcome.spans_denied_by_guardrail, 0);
const executed = result.traces.reduce((n, t) => n + t.outcome.spans_executed, 0);
console.log(`[traces] wrote ${result.traces.length} traces / ${spans} spans to ${tracesFile}`);
console.log(`[traces] executed=${executed} denied_by_guardrail=${denied} expected_eval_calls=${spans * 3}`);
