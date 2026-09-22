/**
 * Redaction is applied to every command, output and metadata value before an
 * artifact is written. It is deliberately conservative: it prefers redacting a
 * harmless value over leaking a real one.
 */

/**
 * Key names whose values are dropped. Deliberately narrower than "contains
 * session": a sandbox `session_id` is an audit identifier, not a credential,
 * and redacting it would cost traceability for no security benefit.
 */
const SENSITIVE_KEY =
  /(api[_-]?key|secret|password|passwd|credential|token|bearer|cookie|private[_-]?key|session[_-]?(token|secret|key)|auth(orization)?)/i;

const PATTERNS = [
  // Provider-shaped credentials.
  [/\bAKIA[0-9A-Z]{16}\b/g, '[REDACTED:aws-access-key-id]'],
  [/\bghp_[A-Za-z0-9]{20,}\b/g, '[REDACTED:github-token]'],
  [/\bsk-[A-Za-z0-9_-]{16,}\b/g, '[REDACTED:api-key]'],
  [/\bvck_[A-Za-z0-9_-]{16,}\b/g, '[REDACTED:vercel-key]'],
  [/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g, '[REDACTED:private-key]'],
  [/\bBearer\s+[A-Za-z0-9._~+/-]{12,}=*/g, 'Bearer [REDACTED:bearer-token]'],
  // KEY=value / KEY: value assignments whose key looks sensitive.
  [/\b([A-Za-z_][A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH))\s*[=:]\s*("[^"]*"|'[^']*'|\S+)/gi, '$1=[REDACTED:value]'],
  // Basic-auth style URLs.
  [/\b([a-z][a-z0-9+.-]*:\/\/)[^\s/@]+:[^\s/@]+@/gi, '$1[REDACTED:userinfo]@'],
];

export function redactString(input) {
  if (typeof input !== 'string') return input;
  let out = input;
  for (const [re, replacement] of PATTERNS) out = out.replace(re, replacement);
  return out;
}

/** Recursively redact a structure; keys that look sensitive lose their value. */
export function redactValue(value, keyHint = '') {
  if (typeof value === 'string') {
    if (keyHint && SENSITIVE_KEY.test(keyHint)) return '[REDACTED:sensitive-key]';
    return redactString(value);
  }
  if (Array.isArray(value)) return value.map((v) => redactValue(v, keyHint));
  if (value && typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = redactValue(v, k);
    return out;
  }
  return value;
}

/** Environment snapshots keep names, never values. */
export function redactEnvSnapshot(env) {
  const out = {};
  for (const [k, v] of Object.entries(env ?? {})) {
    out[k] = SENSITIVE_KEY.test(k) ? '[REDACTED:sensitive-key]' : redactString(String(v));
  }
  return out;
}
