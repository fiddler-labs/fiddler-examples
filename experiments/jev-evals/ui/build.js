#!/usr/bin/env node
/**
 * Static export of the UI for hosting on any file server (Vercel, Pages, S3).
 * Writes ui/dist/ with the three page files plus the same dataset the local
 * server assembles, precomputed as data.json, and the repeatability report as
 * plain text. Everything here is derived from committed files, so ui/dist/ is
 * gitignored and regenerated with `npm run ui:build`.
 */
import fs from 'node:fs';
import path from 'node:path';
import { loadDataset, REPEAT_REPORT, UI_DIR } from './dataset.js';

export function buildStaticUi(outDir = path.join(UI_DIR, 'dist')) {
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });
  for (const f of ['index.html', 'app.js', 'styles.css']) {
    fs.copyFileSync(path.join(UI_DIR, f), path.join(outDir, f));
  }
  const data = loadDataset();
  fs.writeFileSync(path.join(outDir, 'data.json'), JSON.stringify(data));
  if (fs.existsSync(REPEAT_REPORT)) {
    fs.copyFileSync(REPEAT_REPORT, path.join(outDir, 'repeat-report.txt'));
  }
  return { outDir, traces: data.traces.length, hasRepeatReport: fs.existsSync(REPEAT_REPORT) };
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  const r = buildStaticUi();
  console.log(`[ui:build] wrote ${path.relative(process.cwd(), r.outDir)}/ (${r.traces} traces, repeat report: ${r.hasRepeatReport ? 'yes' : 'no'})`);
}
