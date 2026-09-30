import { Sandbox } from '@vercel/sandbox';
import { assertLiveApproved } from '../config.js';
import { SCENARIOS, SETUP_COMMANDS, WORKSPACE } from '../scenarios.js';
import { buildSpan, buildTrace } from '../trace.js';

/**
 * LIVE EXPERIMENT EXECUTOR.
 *
 * For each scenario this provisions a disposable Vercel Sandbox, installs the
 * real GitHub Copilot CLI inside it, points the CLI at Vercel AI Gateway with
 * BYOK provider routing (no GitHub authentication), hands Copilot the scenario
 * prompt, and lets Copilot decide and execute its own tool calls.
 *
 * The harness never chooses a command here. Every captured Bash span is one the
 * agent genuinely issued.
 *
 * Verified against Copilot CLI 1.0.86 (`copilot help providers`):
 *   COPILOT_PROVIDER_BASE_URL activates BYOK and GitHub auth is not required.
 */

export const ARTIFACT_DIR = '/vercel/sandbox/artifacts';
export const COPILOT_CLI_VERSION = process.env.COPILOT_CLI_VERSION ?? '1.0.86';
export const GATEWAY_BASE_URL = process.env.AI_GATEWAY_BASE_URL ?? 'https://ai-gateway.vercel.sh/v1';

/**
 * Agent model, routed through the Gateway. Kept separate from the evaluators.
 *
 * Verified against this account's Gateway tier: Anthropic models return 403
 * ("Free tier users do not have access to this model"), so the default is an
 * OpenAI model that the tier does allow. Override with AGENT_WIRE_MODEL.
 */
export const AGENT_WIRE_MODEL = process.env.AGENT_WIRE_MODEL ?? 'openai/gpt-4.1';
export const AGENT_MODEL_ID = process.env.AGENT_MODEL_ID ?? 'gpt-4.1';

const SANDBOX_TIMEOUT_MS = Number(process.env.SANDBOX_TIMEOUT_MS ?? 10 * 60 * 1000);

/**
 * Vercel Sandbox credentials. Three ways in, checked in this order by the SDK:
 *   1. explicit VERCEL_TOKEN + VERCEL_TEAM_ID + VERCEL_PROJECT_ID (non-interactive; CI),
 *   2. VERCEL_OIDC_TOKEN, written to .env.local by `vercel env pull` (expires in ~12h),
 *   3. an interactive browser sign-in when neither is set and stdin is a TTY.
 * The SDK rejects a partial set of the three explicit variables, so fail early with
 * a clearer message than it gives.
 */
export function sandboxCredentials(env = process.env) {
  const token = env.VERCEL_TOKEN;
  const teamId = env.VERCEL_TEAM_ID;
  const projectId = env.VERCEL_PROJECT_ID;
  const set = [token, teamId, projectId].filter(Boolean).length;
  if (set === 0) return {};
  if (set < 3) {
    throw new Error(
      'VERCEL_TOKEN, VERCEL_TEAM_ID and VERCEL_PROJECT_ID must be set together ' +
        '(or none of them, to use VERCEL_OIDC_TOKEN from `vercel env pull`).',
    );
  }
  return { token, teamId, projectId };
}
const COPILOT_TIMEOUT_MS = Number(process.env.COPILOT_TIMEOUT_MS ?? 5 * 60 * 1000);

export async function runCopilotSandbox({ runMarker, scenarios = SCENARIOS, onProgress = () => {} }) {
  assertLiveApproved('the live Copilot-in-Sandbox experiment');
  const gatewayKey = requireGatewayKey();

  const traces = [];
  const failures = [];
  const startedAt = new Date().toISOString();

  for (const scenario of scenarios) {
    onProgress({ phase: 'scenario.start', scenario_id: scenario.id });
    try {
      const trace = await runOneScenario({ scenario, runMarker, gatewayKey, onProgress });
      traces.push(trace);
      onProgress({
        phase: 'scenario.done',
        scenario_id: scenario.id,
        bash_spans: trace.spans.length,
      });
    } catch (err) {
      // One scenario failing must not discard the scenarios already captured.
      failures.push({ scenario_id: scenario.id, error: err?.message ?? String(err) });
      onProgress({ phase: 'scenario.failed', scenario_id: scenario.id, error: err?.message });
    }
  }

  return {
    traces,
    runMeta: {
      run_marker: runMarker,
      mode: 'live',
      executor: 'copilot-cli-in-vercel-sandbox',
      simulated: false,
      agent: {
        cli: `@github/copilot@${COPILOT_CLI_VERSION}`,
        provider_type: 'openai',
        provider_base_url: GATEWAY_BASE_URL,
        wire_model: AGENT_WIRE_MODEL,
        model_id: AGENT_MODEL_ID,
        github_auth_used: false,
      },
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      scenarios_attempted: scenarios.length,
      scenarios_captured: traces.length,
      failures,
    },
  };
}

async function runOneScenario({ scenario, runMarker, gatewayKey, onProgress }) {
  let sandbox = null;
  const sandboxMeta = { kind: 'vercel-sandbox' };
  const retrieved = { agent_events: null, agent_stderr: null, otel: null, run_marker: null };
  const startedAt = new Date().toISOString();

  try {
    sandbox = await Sandbox.create({
      ...sandboxCredentials(),
      timeout: SANDBOX_TIMEOUT_MS,
      runtime: 'node22',
      tags: { experiment: 'jev-evals', scenario: scenario.id.slice(0, 30) },
    });
    // The SDK identifies a sandbox by `name` (there is no id field); the
    // session id is what ties this run to Vercel's own logs.
    Object.assign(sandboxMeta, {
      sandbox_name: sandbox.name ?? null,
      session_id: sandbox.sandbox?.currentSessionId ?? null,
      // The Vercel project id is an account identifier, not evidence about the
      // agent. Record only that one was present so the artifact can be published.
      project_id: sandbox.projectId ? '[REDACTED:vercel-project-id]' : null,
      region: sandbox.region ?? null,
      runtime: sandbox.runtime ?? null,
      vcpus: sandbox.vcpus ?? null,
      memory: sandbox.memory ?? null,
      timeout_ms: sandbox.timeout ?? null,
      created_at: sandbox.createdAt ? new Date(sandbox.createdAt).toISOString() : null,
    });

    onProgress({ phase: 'sandbox.created', scenario_id: scenario.id, sandbox_id: sandboxMeta.sandbox_name });

    // 1. Fabricated workspace.
    await sh(sandbox, `mkdir -p ${ARTIFACT_DIR} ${WORKSPACE}`);
    for (const cmd of SETUP_COMMANDS) await sh(sandbox, cmd);
    await sh(sandbox, `printf '%s\\n' '${runMarker}' > ${ARTIFACT_DIR}/RUN_MARKER`);

    // 2. Real Copilot CLI, pinned.
    onProgress({ phase: 'copilot.install', scenario_id: scenario.id });
    const install = await sh(sandbox, `npm install -g @github/copilot@${COPILOT_CLI_VERSION} 2>&1 | tail -3`);
    if (install.exitCode !== 0) throw new Error(`Copilot CLI install failed: ${install.stdout}${install.stderr}`);
    const version = await sh(sandbox, 'copilot --version');
    sandboxMeta.copilot_version = version.stdout.trim().split('\n')[0] ?? null;

    // 3. Hand the prompt to the agent. Copilot chooses its own tool calls.
    onProgress({ phase: 'copilot.run', scenario_id: scenario.id });
    const copilotStarted = Date.now();
    const run = await sh(
      sandbox,
      copilotCommand(scenario),
      copilotEnv({ gatewayKey, runMarker, scenario }),
      COPILOT_TIMEOUT_MS,
    );
    const copilotMs = Date.now() - copilotStarted;
    sandboxMeta.copilot_exit_code = run.exitCode;
    sandboxMeta.copilot_duration_ms = copilotMs;
  } finally {
    if (sandbox) {
      // 4. ALWAYS retrieve artifacts before teardown, success or failure.
      retrieved.agent_events = await readText(sandbox, `${ARTIFACT_DIR}/agent-events.jsonl`);
      retrieved.agent_stderr = await readText(sandbox, `${ARTIFACT_DIR}/agent-stderr.log`);
      retrieved.otel = await readText(sandbox, `${ARTIFACT_DIR}/copilot-otel.jsonl`);
      retrieved.run_marker = await readText(sandbox, `${ARTIFACT_DIR}/RUN_MARKER`);
      sandboxMeta.retrieved_run_marker = retrieved.run_marker?.trim() ?? null;
      sandboxMeta.artifacts_retrieved_before_stop = true;

      // 5. Only then destroy the sandbox.
      try {
        await sandbox.stop();
        sandboxMeta.stopped = true;
      } catch (err) {
        sandboxMeta.stopped = false;
        sandboxMeta.stop_error = err?.message ?? String(err);
      }
    }
  }

  const parsed = parseAgentEvents(retrieved.agent_events ?? '');
  const runMeta = {
    run_marker: runMarker,
    mode: 'live',
    executor: 'copilot-cli-in-vercel-sandbox',
    agent: {
      cli: `@github/copilot@${COPILOT_CLI_VERSION}`,
      provider_type: 'openai',
      provider_base_url: GATEWAY_BASE_URL,
      wire_model: AGENT_WIRE_MODEL,
      model_id: AGENT_MODEL_ID,
      github_auth_used: false,
    },
    sandbox: sandboxMeta,
    started_at: startedAt,
    finished_at: new Date().toISOString(),
  };

  const spans = parsed.bashCalls.map((call, index) =>
    buildSpan({ scenario, step: call.step, index, execution: call.execution }),
  );

  const trace = buildTrace({
    scenario,
    spans,
    runMeta,
    events: parsed.events,
    agentClaim: parsed.finalMessage,
    agentSession: parsed.session,
  });
  trace.raw_agent_artifacts = {
    agent_events_jsonl_lines: retrieved.agent_events ? retrieved.agent_events.split('\n').filter(Boolean).length : 0,
    agent_stderr_present: Boolean(retrieved.agent_stderr?.trim()),
    otel_jsonl_lines: retrieved.otel ? retrieved.otel.split('\n').filter(Boolean).length : 0,
  };
  trace._raw = retrieved;
  return trace;
}

/** Non-interactive Copilot invocation, adapted from the reference project. */
function copilotCommand(scenario) {
  const prompt = shellQuote(scenario.prompt);
  return [
    `cd ${WORKSPACE} &&`,
    `copilot -p ${prompt}`,
    `--allow-all-tools`,
    `--no-ask-user`,
    `--no-remote-export`,
    `--no-custom-instructions`,
    `--disable-builtin-mcps`,
    `--no-auto-update`,
    `--secret-env-vars=COPILOT_PROVIDER_API_KEY`,
    `--output-format json`,
    `--log-level none`,
    `> ${ARTIFACT_DIR}/agent-events.jsonl 2> ${ARTIFACT_DIR}/agent-stderr.log`,
    `; echo "copilot_exit=$?"`,
  ].join(' ');
}

/**
 * BYOK routing to Vercel AI Gateway. No GH_TOKEN, GITHUB_TOKEN or
 * COPILOT_GITHUB_TOKEN is ever set here.
 */
function copilotEnv({ gatewayKey, runMarker, scenario }) {
  return {
    HOME: '/vercel/sandbox/copilot-home',
    COPILOT_HOME: '/vercel/sandbox/copilot-home/.copilot',
    COPILOT_CACHE_HOME: '/vercel/sandbox/copilot-cache',
    COPILOT_AUTO_UPDATE: 'false',
    // BYOK: activates custom provider routing, GitHub auth not required.
    COPILOT_PROVIDER_TYPE: 'openai',
    COPILOT_PROVIDER_BASE_URL: GATEWAY_BASE_URL,
    COPILOT_PROVIDER_API_KEY: gatewayKey,
    COPILOT_PROVIDER_WIRE_MODEL: AGENT_WIRE_MODEL,
    COPILOT_PROVIDER_MODEL_ID: AGENT_MODEL_ID,
    COPILOT_MODEL: AGENT_WIRE_MODEL,
    // Copilot's own OTel stream, exported to a file we retrieve before teardown.
    COPILOT_OTEL_ENABLED: 'true',
    COPILOT_OTEL_EXPORTER_TYPE: 'file',
    COPILOT_OTEL_FILE_EXPORTER_PATH: `${ARTIFACT_DIR}/copilot-otel.jsonl`,
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: 'true',
    OTEL_SERVICE_NAME: 'github-copilot-jev-evals',
    OTEL_RESOURCE_ATTRIBUTES: `demo.marker=${runMarker},scenario.id=${scenario.id}`,
  };
}

/**
 * Turn the agent's own JSONL event stream into normalized Bash spans.
 *
 * Only `bash` tool calls become evaluation units, because the rubric's unit is
 * one Bash tool span. Every other agent event is preserved in the timeline.
 */
export function parseAgentEvents(jsonl) {
  const events = [];
  const starts = new Map();
  const bashCalls = [];
  let finalMessage = null;
  let session = null;

  for (const line of jsonl.split('\n')) {
    if (!line.trim()) continue;
    let ev;
    try {
      ev = JSON.parse(line);
    } catch {
      continue;
    }

    if (ev.type === 'tool.execution_start') starts.set(ev.data?.toolCallId, ev);
    if (ev.type === 'tool.execution_complete') {
      const start = starts.get(ev.data?.toolCallId);
      const toolName = start?.data?.toolName ?? ev.data?.toolName ?? null;
      if (toolName === 'bash') bashCalls.push(toBashCall(start, ev));
    }
    if (ev.type === 'assistant.message' && ev.data?.content) finalMessage = ev.data.content;
    if (ev.type === 'result') {
      session = {
        session_id: ev.sessionId ?? null,
        exit_code: ev.exitCode ?? null,
        usage: ev.usage ?? null,
      };
    }

    // Keep a compact timeline; the full stream is preserved in the raw artifact.
    if (!ev.ephemeral) {
      events.push({
        at: ev.timestamp ?? null,
        type: ev.type,
        tool_call_id: ev.data?.toolCallId ?? null,
        tool_name: ev.data?.toolName ?? null,
        event_id: ev.id ?? null,
      });
    }
  }

  return { events, bashCalls, finalMessage, session };
}

function toBashCall(start, complete) {
  const args = start?.data?.arguments ?? {};
  const startedAt = start?.timestamp ?? null;
  const endedAt = complete?.timestamp ?? null;
  const exitCode = complete?.data?.shellExecution?.exitCode ?? null;
  const denied = isDenied(complete);

  return {
    step: {
      // The agent authored both of these.
      step_id: complete?.data?.toolCallId ?? start?.data?.toolCallId ?? null,
      command: args.command ?? '',
      description: args.description ?? '',
      guardrail: denied ? 'deny' : 'allow',
      tool_call_id: complete?.data?.toolCallId ?? null,
    },
    execution: {
      executed: !denied,
      error_type: denied ? 'denied' : exitCode === 0 ? null : 'command_failed',
      exit_code: denied ? null : exitCode,
      stdout: complete?.data?.result?.content ?? complete?.data?.result?.detailedContent ?? '',
      stderr: '',
      started_at: startedAt,
      ended_at: endedAt,
      duration_ms: startedAt && endedAt ? new Date(endedAt) - new Date(startedAt) : null,
    },
  };
}

function isDenied(complete) {
  const d = complete?.data ?? {};
  if (d.denied === true || d.permissionDenied === true) return true;
  const text = `${d.result?.content ?? ''}`;
  return /permission denied by user|tool call denied|not permitted/i.test(text);
}

function requireGatewayKey() {
  const key = process.env.AI_GATEWAY_API_KEY;
  if (!key) {
    throw new Error(
      'AI_GATEWAY_API_KEY is not set. The live run routes both the Copilot agent (BYOK) ' +
        'and the three evaluators through Vercel AI Gateway. Set it in .env - this project ' +
        'never creates a Gateway key on your behalf.',
    );
  }
  return key;
}

async function sh(sandbox, command, env, timeoutMs) {
  const finished = await sandbox.runCommand({
    cmd: 'bash',
    args: ['-lc', command],
    env,
    timeoutMs: timeoutMs ?? 120_000,
  });
  const [stdout, stderr] = await Promise.all([finished.stdout(), finished.stderr()]);
  return { exitCode: finished.exitCode, stdout, stderr };
}

async function readText(sandbox, path) {
  try {
    const buf = await sandbox.readFileToBuffer({ path });
    return buf ? buf.toString('utf8') : null;
  } catch {
    return null;
  }
}

function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}
