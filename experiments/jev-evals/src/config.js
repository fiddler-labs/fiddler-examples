import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const ARTIFACTS = path.join(ROOT, 'artifacts');
export const FIXTURES = path.join(ROOT, 'fixtures');

export const PATHS = {
  traces: path.join(ARTIFACTS, 'traces.jsonl'),
  evalRaw: path.join(ARTIFACTS, 'evaluations.raw.jsonl'),
  evalNorm: path.join(ARTIFACTS, 'evaluations.normalized.jsonl'),
  summary: path.join(ARTIFACTS, 'summary.json'),
  runMeta: path.join(ARTIFACTS, 'run-meta.json'),
  report: path.join(ROOT, 'docs', 'REPORT.md'),
};

/**
 * Each vendor's designated classification tier, plus Jev as the specialist.
 * See docs/METHODOLOGY.md "Why these two peer models" for the selection rule.
 *
 * `providerOptions` pins each peer to the cheapest reasoning setting its vendor
 * recommends for classification, so the two are comparable in operation and not
 * only in intent. Without it the defaults diverge: nano reasons before
 * answering, Flash-Lite largely does not.
 *
 * `gateway.only` pins the serving provider. The Gateway otherwise falls back
 * between providers mid-run - openai to azure, vertex to google - which both
 * trips a 5 rpm cap on the azure route and makes "the same model answered
 * twice" untrue in a study about repeatability.
 */
export const EVALUATORS = [
  { key: 'jev', model: 'typesafe-ai/jev', kind: 'typed' },
  {
    key: 'gemini',
    model: 'google/gemini-3.5-flash-lite',
    kind: 'generative',
    providerOptions: {
      google: { thinkingConfig: { thinkingLevel: 'minimal' } },
      gateway: { only: ['vertex'] },
    },
  },
  {
    key: 'openai',
    model: 'openai/gpt-5.4-nano',
    kind: 'generative',
    providerOptions: {
      openai: { reasoningEffort: 'none' },
      gateway: { only: ['openai'] },
    },
  },
];

// Output caps keep traces reviewable and content-safe.
export const OUTPUT_CAP_BYTES = 4096;
export const COMMAND_TIMEOUT_MS = 20_000;

/**
 * Load .env files into process.env without overwriting values already set.
 * `.env.local` is written by `vercel link` / `vercel env pull` and carries the
 * VERCEL_OIDC_TOKEN used to authenticate Vercel Sandbox.
 */
export function loadDotEnv(files = [path.join(ROOT, '.env'), path.join(ROOT, '.env.local')]) {
  let loaded = false;
  for (const f of [].concat(files)) loaded = loadOneEnvFile(f) || loaded;
  return loaded;
}

function loadOneEnvFile(file) {
  if (!fs.existsSync(file)) return false;
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    const m = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/.exec(line);
    if (!m || line.trim().startsWith('#')) continue;
    const v = m[2].replace(/^["']|["']$/g, '');
    if (process.env[m[1]] === undefined) process.env[m[1]] = v;
  }
  return true;
}

/** Fail closed: live work only runs with explicit approval. */
export function assertLiveApproved(what) {
  if (process.env.LIVE_RUN_APPROVED !== 'yes') {
    throw new Error(
      `Refusing to run ${what} live: LIVE_RUN_APPROVED is not "yes". ` +
        `Set LIVE_RUN_APPROVED=yes explicitly to authorize live calls.`,
    );
  }
}
