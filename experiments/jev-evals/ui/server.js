#!/usr/bin/env node
/**
 * Minimal local analysis server: Node's http module, no framework, no build
 * step. It serves the static UI and reads the preserved artifacts from disk.
 * It never computes model answers - it only exposes what is already stored.
 *
 * The two data routes use the same relative names that `npm run ui:build`
 * writes into ui/dist/, so the page code is identical locally and when hosted.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { loadDataset, REPEAT_REPORT, UI_DIR } from './dataset.js';

const PORT = Number(process.env.UI_PORT ?? 5177);
// Bind IPv4 loopback explicitly. Node's default listen() binds to :: , which on
// some hosts is IPv6-only and refuses connections to 127.0.0.1.
const HOST = process.env.UI_HOST ?? '127.0.0.1';
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  if (url.pathname === '/data.json') {
    res.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' });
    return res.end(JSON.stringify(loadDataset()));
  }
  // The repeatability report, read-only. The path is fixed in code and takes
  // nothing from the request, so there is no traversal surface.
  if (url.pathname === '/repeat-report.txt') {
    const report = REPEAT_REPORT;
    if (!fs.existsSync(report)) {
      res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
      return res.end('No repeatability report: artifacts/repeatability/REPORT.md is not present.');
    }
    // text/plain so the browser renders it inline instead of downloading it.
    res.writeHead(200, { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store' });
    return fs.createReadStream(report).pipe(res);
  }

  const file = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
  const full = path.join(UI_DIR, file);
  if (!full.startsWith(UI_DIR) || !fs.existsSync(full) || fs.statSync(full).isDirectory()) {
    res.writeHead(404, { 'content-type': 'text/plain' });
    return res.end('not found');
  }
  res.writeHead(200, { 'content-type': TYPES[path.extname(full)] ?? 'application/octet-stream' });
  fs.createReadStream(full).pipe(res);
});

server.listen(PORT, HOST, () => console.log(`[ui] listening on http://${HOST}:${PORT}`));
