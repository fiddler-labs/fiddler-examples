import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from '../config.js';

const PRICING_FILE = path.join(ROOT, 'config', 'pricing.json');

/**
 * Cost is only reported when a price table is actually configured, or when the
 * provider returns a cost itself. It is never estimated from guessed rates.
 */
export function loadPricing() {
  if (!fs.existsSync(PRICING_FILE)) return null;
  try {
    return JSON.parse(fs.readFileSync(PRICING_FILE, 'utf8'));
  } catch {
    return null;
  }
}

/**
 * AI Gateway reports a real cost on the response. Prefer it over any local
 * table; only fall back to a configured price table, and otherwise report the
 * cost as unavailable rather than estimating it.
 */
export function extractProviderCost(providerMetadata) {
  const candidates = [
    providerMetadata?.gateway?.cost,
    providerMetadata?.gateway?.usage?.cost,
    providerMetadata?.gateway?.gateway_cost,
    providerMetadata?.vercel?.cost,
  ];
  for (const c of candidates) {
    const n = typeof c === 'string' ? Number(c) : c;
    if (typeof n === 'number' && Number.isFinite(n)) return n;
  }
  return undefined;
}

export function estimateCost({ model, usage, providerCostUsd }) {
  if (typeof providerCostUsd === 'number') {
    return { estimated_usd: providerCostUsd, basis: 'provider_reported' };
  }
  const table = loadPricing();
  const rates = table?.[model];
  if (!rates || usage?.input_tokens == null || usage?.output_tokens == null) {
    return {
      estimated_usd: null,
      basis: null,
      unavailable_reason: !rates
        ? 'no price table entry for this model in config/pricing.json'
        : 'token usage not reported by the provider',
    };
  }
  const usd =
    (usage.input_tokens / 1e6) * rates.input_usd_per_mtok +
    (usage.output_tokens / 1e6) * rates.output_usd_per_mtok;
  return { estimated_usd: Number(usd.toFixed(8)), basis: 'config/pricing.json' };
}
