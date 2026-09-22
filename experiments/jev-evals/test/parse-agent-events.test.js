import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { FIXTURES } from '../src/config.js';
import { parseAgentEvents } from '../src/executors/copilot-sandbox.js';

const jsonl = fs.readFileSync(path.join(FIXTURES, 'copilot-events.sample.jsonl'), 'utf8');
const parsed = parseAgentEvents(jsonl);

test('only bash tool calls become evaluation units', () => {
  assert.equal(parsed.bashCalls.length, 3);
  assert.ok(!parsed.bashCalls.some((c) => c.step.command === ''));
});

test('command and agent-written description come from the agent', () => {
  assert.equal(parsed.bashCalls[0].step.command, 'cat config/.env.fabricated');
  assert.equal(parsed.bashCalls[0].step.description, 'Read the configuration file');
});

test('exit codes and durations are taken from execution evidence', () => {
  assert.equal(parsed.bashCalls[0].execution.exit_code, 0);
  assert.equal(parsed.bashCalls[1].execution.exit_code, 7);
  assert.equal(parsed.bashCalls[1].execution.error_type, 'command_failed');
  assert.equal(parsed.bashCalls[0].execution.duration_ms, 108);
});

test('denied tool calls are recorded as denied and not as executed', () => {
  const denied = parsed.bashCalls[2];
  assert.equal(denied.execution.executed, false);
  assert.equal(denied.execution.error_type, 'denied');
  assert.equal(denied.step.guardrail, 'deny');
  assert.equal(denied.execution.exit_code, null);
});

test("the agent's final message is captured as its self-reported claim", () => {
  assert.match(parsed.finalMessage, /attempted the upload/);
});

test('session metadata is captured from the result event', () => {
  assert.equal(parsed.session.session_id, 'sess-123');
  assert.equal(parsed.session.exit_code, 0);
});

test('ephemeral events are excluded from the timeline', () => {
  assert.ok(!parsed.events.some((e) => e.type === 'session.mcp_server_status_changed'));
  assert.ok(parsed.events.some((e) => e.type === 'tool.execution_start'));
});
