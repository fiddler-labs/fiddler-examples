/**
 * Assembles the read-only dataset the UI displays from the preserved artifacts.
 * Shared by the local server (ui/server.js) and the static export (ui/build.js)
 * so both surfaces show byte-identical data. It never computes model answers.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ARTIFACTS } from '../src/config.js';
import { readJsonl, readJson } from '../src/util.js';

export const UI_DIR = path.dirname(fileURLToPath(import.meta.url));
export const REPEAT_REPORT = path.join(ARTIFACTS, 'repeatability', 'REPORT.md');

export function loadDataset() {
  const dir = ARTIFACTS;
  const maybe = (f) => (fs.existsSync(path.join(dir, f)) ? readJson(path.join(dir, f)) : null);
  const rawDir = path.join(ARTIFACTS, 'raw');
  // The repeatability replay is a separate, read-only study over the same
  // stored spans. Its absence is normal - the UI degrades to hiding every
  // repeat affordance.
  const repeatDir = path.join(ARTIFACTS, 'repeatability');
  const repeatFile = (f) =>
    fs.existsSync(path.join(repeatDir, f)) ? readJson(path.join(repeatDir, f)) : null;
  return {
    available: fs.existsSync(path.join(dir, 'traces.jsonl')),
    ...loadRubric(maybe('eval-meta.json')),
    dir: path.relative(process.cwd(), dir),
    traces: readJsonl(path.join(dir, 'traces.jsonl')),
    results: readJsonl(path.join(dir, 'evaluations.normalized.jsonl')),
    raw_results: readJsonl(path.join(dir, 'evaluations.raw.jsonl')),
    summary: maybe('summary.json'),
    eval_meta: maybe('eval-meta.json'),
    run_meta: maybe('run-meta.json'),
    raw_agent_files: fs.existsSync(rawDir) ? fs.readdirSync(rawDir) : [],
    repeat_per_span: repeatFile('repeat-per-span.json'),
    repeat_analysis: repeatFile('repeat-analysis.json'),
    repeat_meta: repeatFile('repeat-meta.json'),
  };
}

/**
 * The rubric the evaluators were given, so the UI can show its definitions
 * next to the results. The version recorded in eval-meta picks the file; if
 * nothing matches, the newest rubric file is used and marked as such.
 */
export function loadRubric(evalMeta) {
  const rubricDir = path.join(UI_DIR, '..', 'rubric');
  if (!fs.existsSync(rubricDir)) return { rubric: null, shared_prompt: null };
  const files = fs.readdirSync(rubricDir).filter((f) => f.endsWith('.json')).sort();
  const all = files.map((f) => readJson(path.join(rubricDir, f)));
  const want = evalMeta?.rubric_version;
  const rubric = all.find((r) => r.rubric_version === want) ?? all.at(-1) ?? null;
  if (!rubric) return { rubric: null, shared_prompt: null };
  const promptFile = path.join(rubricDir, rubric.shared_prompt_file ?? '');
  return {
    rubric: { ...rubric, matches_run: rubric.rubric_version === want },
    shared_prompt: fs.existsSync(promptFile) ? fs.readFileSync(promptFile, 'utf8') : null,
  };
}
