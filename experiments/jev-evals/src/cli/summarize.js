#!/usr/bin/env node
import path from 'node:path';
import fs from 'node:fs';
import { ARTIFACTS, PATHS } from '../config.js';
import { readJsonl, readJson, writeJson } from '../util.js';
import { summarize } from '../summarize.js';

const dir = ARTIFACTS;

const traces = readJsonl(path.join(dir, 'traces.jsonl'));
const results = readJsonl(path.join(dir, 'evaluations.normalized.jsonl'));
const evalMetaFile = path.join(dir, 'eval-meta.json');
const evalMeta = fs.existsSync(evalMetaFile) ? readJson(evalMetaFile) : null;

if (!traces.length || !results.length) {
  console.error(`[summary] missing traces or results in ${dir}`);
  process.exit(1);
}

const summary = summarize({ traces, results, evalMeta });
const out = PATHS.summary;
writeJson(out, summary);

console.log(`[summary] ${summary.totals.scenarios} scenarios, ${summary.totals.spans} spans, ${summary.totals.calls} calls`);
console.log(`[summary] identical submitted prompts across evaluators: ${summary.totals.identical_submitted_prompt_hashes.all_identical}`);
console.log(`[summary] agreement: ${JSON.stringify(summary.agreement_distribution)}`);
console.log(`[summary] wrote ${out}`);
