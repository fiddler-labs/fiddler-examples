/**
 * Three-pane drill-down over the preserved artifacts: scenario, command, and
 * the three ratings of that command.
 *
 * The reading path is deliberately shallow: pick a scenario, pick a command,
 * compare three ratings. Operational detail - raw input, full output, hashes,
 * ids, tokens, cost, sandbox and run metadata - lives in one labelled evidence
 * disclosure so it never competes with the ratings.
 *
 * Everything shown comes from the stored records. A field that was never
 * captured is rendered as unavailable rather than filled in, and anything
 * derived locally says so.
 */
// Keys and initials are fixed by the pipeline; the display name is derived from
// whichever model the loaded run actually used, so swapping a model in
// src/config.js cannot leave the UI naming the previous one.
const EVALUATORS = [
  { key: 'jev', short: 'J', name: 'Jev' },
  { key: 'gemini', short: 'G', name: 'Gemini' },
  { key: 'openai', short: 'O', name: 'OpenAI' },
];

/**
 * "google/gemini-3.5-flash-lite" -> "Gemini 3.5 Flash Lite"
 * "openai/gpt-5.4-nano"          -> "GPT-5.4 nano"
 * The OpenAI families keep the vendor's own casing: GPT stays uppercase and the
 * size word stays lower, which is how the model pages write them.
 */
function modelDisplayName(modelId) {
  const slug = String(modelId).split('/').pop() ?? '';
  const parts = slug.split('-');
  if (parts[0] === 'gpt') return `GPT-${parts[1] ?? ''} ${parts.slice(2).join(' ')}`.trim();
  return parts
    .map((part) => (/^[0-9]/.test(part) ? part : part.charAt(0).toUpperCase() + part.slice(1)))
    .join(' ');
}

/** Name each evaluator after the model in the loaded dataset, once per load. */
function nameEvaluators(data) {
  const byKey = new Map((data?.eval_meta?.evaluators ?? []).map((e) => [e.key, e.model]));
  for (const e of EVALUATORS) {
    const model = byKey.get(e.key) ?? data?.results?.find((r) => r.evaluator === e.key)?.model;
    if (model && e.key !== 'jev') e.name = modelDisplayName(model);
  }
}
const LEVELS = ['low', 'medium', 'high', 'critical'];
const LEVEL_INITIAL = { low: 'L', medium: 'M', high: 'H', critical: 'C' };
const OUTPUT_PREVIEW_LINES = 8;

const state = {
  data: null,
  scenario: null,
  span: null,
  filter: '',
  onlyDisagree: false,
  rawView: false,
  view: 'ratings',
  repeatOnlyChanged: false,
  repeatExpanded: null,
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const na = (why) => `<span class="unavailable">${esc(why ?? 'unavailable')}</span>`;
const val = (v, fmt = esc, why) => (v === null || v === undefined || v === '' ? na(why) : fmt(v));

async function load() {
  state.data = await (await fetch('data.json')).json();
  nameEvaluators(state.data);
  state.scenario = state.data.traces[0]?.scenario_id ?? null;
  state.span = firstSpanOf(state.scenario);
  render();
}

const traceOf = (id) => state.data.traces.find((t) => t.scenario_id === id);

function visibleSpans(scenarioId) {
  const t = traceOf(scenarioId);
  if (!t) return [];
  return t.spans.filter((s) => !state.onlyDisagree || !['unanimous', 'incomplete'].includes(agreementOf(s.span_id)));
}
const firstSpanOf = (scenarioId) => visibleSpans(scenarioId)[0]?.span_id ?? null;

function resultsFor(spanId) {
  const out = {};
  for (const r of state.data.results) if (r.span_id === spanId) out[r.evaluator] = r;
  return out;
}

function agreementOf(spanId) {
  const per = resultsFor(spanId);
  const levels = EVALUATORS.map((e) => per[e.key]?.risk_level).filter(Boolean);
  if (levels.length < EVALUATORS.length) return 'incomplete';
  const n = new Set(levels).size;
  return n === 1 ? 'unanimous' : n === 2 ? 'split' : 'three_way';
}

/** A command whose result changes how you read it: failed, denied, or not run. */
const isFlagged = (span) => !span.executed || span.error_type !== null || span.exit_code !== 0;

/* ---------------- repeatability ----------------
 * A second study re-asked all three models the same stored commands twice more.
 * It lives behind its own header tab and shares nothing with the ratings path,
 * so the default reading path stays exactly as it was.
 */

const AGREEMENT_TEXT = {
  unanimous: 'all three agreed',
  split: 'two agreed, one differed',
  three_way: 'all three differed',
  incomplete: 'not all three answered',
};

/** The three observations, in the order they were made. */
const RUNS = [
  { key: 'baseline', label: 'original' },
  { key: 'r1', label: 'repeat 1' },
  { key: 'r2', label: 'repeat 2' },
];

/** The three observations in order, as plain text. */
const repeatLabels = (rep) => [rep?.labels?.baseline, rep?.labels?.r1, rep?.labels?.r2].filter(Boolean);

/* ---------------- render ---------------- */

function render() {
  const d = state.data;

  // The header carries no standing text; it speaks only when there is nothing to show.
  $('caveat').hidden = d.available;
  $('caveat').textContent = d.available ? '' : 'No artifacts yet. Run the pipeline first.';

  // The repeat study exists for the live run only. With no study to show, the
  // tab is withdrawn rather than left to open an empty panel.
  const hasRepeat = !!d.repeat_analysis;
  $('view-repeat').hidden = !hasRepeat;
  if (!hasRepeat && state.view === 'repeat') setView('ratings');

  renderSummary();
  renderScenarios();
  renderSpans();
  renderDetail();
  renderRepeat();
}

function renderSummary() {
  const s = state.data.summary;
  if (!s) {
    $('summary').innerHTML = `<p class="empty">No summary.json yet — run the summarize step.</p>`;
    return;
  }

  // Counts sit inside their own segment wherever one fits, so the bar is read
  // without translating "L18 M7 H6" back into levels.
  const dist = EVALUATORS.map((e) => {
    const p = s.per_evaluator[e.key] ?? {};
    const d = p.distribution ?? {};
    const total = Object.values(d).reduce((a, b) => a + b, 0) || 1;
    const bar = LEVELS.map((l) => {
      if (!d[l]) return '';
      const pct = (d[l] / total) * 100;
      return `<i class="${l}" style="width:${pct}%"
        title="${d[l]} ${l}">${pct >= 9 ? d[l] : ''}</i>`;
    }).join('');
    const spoken = LEVELS.filter((l) => d[l]).map((l) => `${d[l]} ${l}`).join(', ');
    return `<div class="sumrow"><span class="who" title="${esc(e.name)}">${esc(e.name)}</span>
      <span class="bar" role="img" aria-label="${esc(`${e.name}: ${spoken}`)}">${bar}</span></div>`;
  }).join('');

  // Speed, cost and explanation on one row per model: the three axes the page
  // title promises, side by side. Latency is a bar as well as a number, since
  // three medians an order of magnitude apart are a comparison.
  const medians = EVALUATORS.map((e) => s.per_evaluator[e.key]?.latency_ms?.p50).filter((v) => v != null);
  const slowest = medians.length ? Math.max(...medians) : 1;
  const money = (p) => {
    if (p?.estimated_cost_usd == null) return na(p?.cost_unavailable_reason ?? 'n/a');
    if (p.estimated_cost_usd === 0) return `<span title="no billed cost recorded for this run">$0</span>`;
    return `$${p.estimated_cost_usd.toFixed(3)}`;
  };
  const words = (p) => {
    const chars = p?.rationale_length_chars?.mean;
    if (chars == null) return na(p?.rationale_capability === 'not_requested_classifier_only' ? 'none' : 'n/a');
    return `~${Math.round(chars / 5.5)} words`;
  };
  const axes = EVALUATORS.map((e) => {
    const p = s.per_evaluator[e.key];
    const ms = p?.latency_ms?.p50;
    return `<div class="sumrow axes"><span class="who" title="${esc(e.name)}">${esc(e.name)}</span>
      <span class="lat" role="img" aria-label="${esc(`${e.name}: ${ms ?? 'unknown'} ms median`)}">
        ${ms != null ? `<i style="width:${(ms / slowest) * 100}%"></i>` : ''}</span>
      <span class="num">${ms != null ? `${ms} ms` : na('n/a')}</span>
      <span class="num cost">${money(p)}</span>
      <span class="num words">${words(p)}</span></div>`;
  }).join('');
  const axesHead = `<div class="cols axes"><span></span><span></span>
      <span class="tip" tabindex="0" data-tip="median latency per call; bars are scaled to the slowest model">median</span>
      <span class="tip" tabindex="0" data-tip="estimated total cost of this run's evaluation calls, from token usage">run cost</span>
      <span class="tip" tabindex="0" data-tip="mean length of the model's written rationale">explanation</span></div>`;

  const ag = s.agreement_distribution ?? {};
  const total = s.totals?.spans ?? 0;

  const spans = s.totals?.spans ?? 0;
  $('summary').innerHTML = `
    <h2 class="sum-head">Risk rating overview
      <span class="muted">${spans} commands · 3 models</span></h2>
    <div class="sumblock">
      <div class="lab">Risk levels assigned</div>
      ${dist}
      <ul class="level-key">${LEVELS.map((l) =>
        `<li><span class="sw ${l}"></span>${l}</li>`).join('')}</ul>
    </div>
    <div class="sumblock">
      <div class="lab">Speed, cost and explanation</div>
      ${axesHead}${axes}
    </div>
    <div class="sumblock landed">
      <div class="lab">Where they landed</div>
      ${landed(ag, total)}
      ${pairwise(s.pairwise_agreement)}
    </div>
    <div class="sumblock figures">
      <details>
        <summary>Exact figures <span class="hint">— the numbers behind the bars above, per model</span></summary>
        ${exactFigures(s)}
      </details>
      ${evaluatorDefs()}
    </div>`;
}

/**
 * Who the three evaluators are and what they were asked: model ids, what each
 * returns, the provider settings that were fixed for the run, the four risk
 * levels as the rubric defines them, and the shared prompt text itself.
 */
function evaluatorDefs() {
  const meta = state.data.eval_meta;
  const rubric = state.data.rubric;
  if (!meta && !rubric) return '';

  const settings = (ev) => {
    const out = [];
    for (const [vendor, opts] of Object.entries(ev.providerOptions ?? {})) {
      if (vendor === 'gateway') continue;
      for (const [k, v] of Object.entries(opts)) {
        if (v && typeof v === 'object') {
          for (const [k2, v2] of Object.entries(v)) out.push(`${k2}: ${v2}`);
        } else out.push(`${k}: ${v}`);
      }
    }
    return out.length ? out.join(', ') : 'defaults';
  };
  const returns = (key, ev) => {
    const caps = rubric?.capabilities?.[key];
    if (caps?.returns) return caps.returns.join(' + ');
    return ev.kind === 'typed' ? 'risk_level' : 'risk_level + rationale';
  };
  const evaluators = (meta?.evaluators ?? EVALUATORS).map((ev) => {
    const e = EVALUATORS.find((x) => x.key === ev.key) ?? ev;
    const typed = ev.kind === 'typed';
    const opts = settings(ev);
    return `<li class="ev-card">
      <div class="ev-name"><b>${esc(e.name)}</b>
        <span class="ev-kind">${typed ? 'typed classifier' : 'generative'}</span></div>
      <code class="id">${esc(ev.model ?? '')}</code>
      <dl>
        <dt>returns</dt><dd>${esc(returns(ev.key, ev))}</dd>
        <dt>how</dt><dd>${typed ? 'one choice question, calibrated probabilities'
                                : 'free text parsed into the fields'}</dd>
        <dt>settings</dt><dd>${opts === 'defaults' ? '<span class="muted">defaults</span>' : `<code>${esc(opts)}</code>`}</dd>
      </dl>
    </li>`;
  }).join('');

  // One card per level, definition quoted verbatim from the rubric file.
  const levels = rubric ? LEVELS.map((l) => `<li class="level-card ${l}">
      <div class="level-name"><span class="sw ${l}"></span><b class="${l}">${l}</b></div>
      <p>${esc(rubric.risk_level_definitions?.[l] ?? '')}</p>
    </li>`).join('') : '';

  const rubricLine = rubric
    ? `<code class="id">${esc(rubric.rubric_id)} v${esc(rubric.rubric_version)}</code>${
        rubric.matches_run === false ? ' <span class="high">(newest file; the run recorded a different version)</span>' : ''}`
    : na('rubric file not found');

  return `<details class="defs">
    <summary>Evaluators and rubric <span class="hint">— who rated, what they returned, and the
      definitions they were given</span></summary>
    <ul class="ev-cards">${evaluators}</ul>
    <div class="defs-col">
      ${levels ? `<div class="lab">Risk levels, as defined in rubric ${rubricLine}</div>
        <ul class="level-cards">${levels}</ul>` : ''}
      ${rubric?.outputs?.rationale_schema_instruction
        ? `<div class="lab">Rationale, where requested</div>
           <p class="note">${esc(rubric.outputs.rationale_schema_instruction)}</p>` : ''}
      ${state.data.shared_prompt
        ? `<details class="prompt-text">
            <summary>Shared prompt text, as sent <span class="hint">— ${state.data.shared_prompt.split('\n').length} lines</span></summary>
            <pre>${esc(state.data.shared_prompt)}</pre>
          </details>` : ''}
    </div>
  </details>`;
}

/**
 * Agreement as one segmented bar. Every segment carries its count as text and a
 * distinct fill pattern, and the key below repeats count and plain-text label,
 * so nothing here depends on colour alone.
 */
function landed(ag, total) {
  const bands = [
    { key: 'unanimous', n: ag.unanimous ?? 0, label: 'all three agreed' },
    { key: 'split', n: ag.split ?? 0, label: 'split two-to-one' },
    { key: 'three_way', n: ag.three_way ?? 0, label: 'all three differed' },
  ];
  const counted = bands.reduce((a, b) => a + b.n, 0) || 1;

  const segs = bands
    .filter((b) => b.n > 0)
    .map((b) => {
      const pct = (b.n / counted) * 100;
      // Below roughly a tenth of the width the inline number stops fitting.
      return `<i class="seg ${b.key}" style="width:${pct}%" title="${b.n} ${b.label}">${
        pct >= 10 ? `<b>${b.n}</b>` : ''
      }</i>`;
    })
    .join('');

  const key = bands
    .map((b) => `<li><span class="sw ${b.key}"></span><b>${b.n}</b> ${esc(b.label)}</li>`)
    .join('');

  return `<div class="landed-bar" role="img"
      aria-label="${bands.map((b) => `${b.n} ${b.label}`).join(', ')}, of ${total} commands">${segs}</div>
    <ul class="landed-key">${key}</ul>`;
}

/**
 * How often each pair of models chose the very same level. Folded by default:
 * the closed line carries the range, so the headline survives without the
 * three bars, and a reader who wants the pairs opens it.
 */
function pairwise(pa) {
  if (!pa) return '';
  // First word of the display name: "Gemini", "GPT-5.4"; long enough to tell
  // the three apart, short enough for the pair to fit the label cell.
  const name = (k) => (EVALUATORS.find((e) => e.key === k)?.name ?? k).split(' ')[0];
  const entries = Object.entries(pa).map(([k, v]) => {
    const [a, b] = k.split('|');
    return { label: `${name(a)} & ${name(b)}`, pct: Math.round((v.exact_agreement ?? 0) * 100) };
  });
  if (!entries.length) return '';
  const lo = Math.min(...entries.map((e) => e.pct));
  const hi = Math.max(...entries.map((e) => e.pct));
  const range = lo === hi ? `${lo}%` : `${lo}–${hi}%`;
  const rows = entries.map((e) => `<div class="sumrow pair"><span class="who">${esc(e.label)}</span>
      <span class="lat" role="img" aria-label="${e.pct}% exact agreement"><i style="width:${e.pct}%"></i></span>
      <span class="num"><b>${e.pct}%</b></span></div>`).join('');
  return `<details class="pairs">
      <summary><span class="sumlabel">Pair by pair</span>
        <span class="hint">— exact match ${range} of the time</span></summary>
      ${rows}
    </details>`;
}

/**
 * The numbers behind the bars. One row per evaluator, then the pair table.
 * Every figure column is right-aligned and tabular so a column can be scanned
 * down; the level counts sit on one line so a row stays one line tall.
 */
function exactFigures(s) {
  const rows = EVALUATORS.map((e) => {
    const p = s.per_evaluator[e.key] ?? {};
    const d = p.distribution ?? {};
    const levels = LEVELS.map((l) =>
      `<span class="lv"><i class="sw ${l}"></i>${d[l] ?? 0}</span>`).join('');
    return `<tr>
      <td><b>${esc(e.name)}</b><br><code class="id">${esc(p.model ?? '')}</code></td>
      <td class="levels">${levels}</td>
      <td class="fig">${val(p.latency_ms?.p50, (v) => `${v.toLocaleString()} ms`, 'not recorded')}</td>
      <td class="fig">${val(p.latency_ms?.p95, (v) => `${v.toLocaleString()} ms`, '—')}</td>
      <td class="fig">${val(p.usage_totals?.total_tokens, (v) => v.toLocaleString(), 'not reported')}</td>
      <td class="fig">${val(p.estimated_cost_usd, (v) => `$${v.toFixed(6)}`, p.cost_unavailable_reason)}</td>
      <td>${p.probabilities_available ? 'returned' : na('none')}</td>
      <td>${p.rationale_capability === 'not_requested_classifier_only'
        ? na('not requested')
        : `${p.rationale_count} · mean ${p.rationale_length_chars?.mean ?? '?'} chars`}</td>
      <td class="fig">${p.errors}</td></tr>`;
  }).join('');

  const nameOf = (key) => EVALUATORS.find((e) => e.key === key)?.name ?? key;
  const pairRows = Object.entries(s.pairwise_agreement ?? {}).map(([k, v]) => {
    const [a, b] = k.split('|');
    const pct = v.exact_agreement != null ? v.exact_agreement * 100 : null;
    return `<tr>
      <td>${esc(nameOf(a))} <span class="muted">vs</span> ${esc(nameOf(b))}</td>
      <td class="fig">${pct == null ? na() : `${pct.toFixed(1)}%`}</td>
      <td class="pairbar">${pct == null ? '' :
        `<span class="bar" role="img" aria-label="${pct.toFixed(1)} percent">
           <i class="pct" style="width:${pct}%"></i></span>`}</td>
      <td class="fig">${v.compared_spans}</td></tr>`;
  }).join('');

  return `<table class="figtable">
      <caption>Per evaluator, across all ${s.totals?.spans ?? 0} commands</caption>
      <tr><th>evaluator</th><th>levels</th><th class="fig">latency p50</th>
        <th class="fig">p95</th><th class="fig">tokens</th><th class="fig">cost</th>
        <th>probabilities</th><th>explanations</th><th class="fig">errors</th></tr>
      ${rows}</table>
    <table class="figtable">
      <caption>How often two evaluators chose the same level</caption>
      <tr><th>pair</th><th class="fig">same level</th><th></th>
        <th class="fig">commands</th></tr>
      ${pairRows}</table>`;
}

function renderScenarios() {
  const html = state.data.traces
    .filter((t) => `${t.scenario_id} ${t.title} ${t.design_band}`.toLowerCase().includes(state.filter))
    .map((t) => {
      const contested = t.spans.filter((s) => !['unanimous', 'incomplete'].includes(agreementOf(s.span_id))).length;
      // Two narrow numeric columns instead of a wrapped subtitle: the counts line
      // up down the list, and the pane heading says what they are. The numbers
      // read bare, so each carries its own spoken label.
      const n = t.spans.length;
      return `<div class="row scenario" role="option" aria-selected="${t.scenario_id === state.scenario}" data-id="${esc(t.scenario_id)}">
        <span class="name">${esc(t.title)}</span>
        <span class="n" role="img" aria-label="${n} command${n === 1 ? '' : 's'}">${n}</span>
        <span class="n split" role="img"
          aria-label="${contested ? `${contested} contested` : 'none contested'}">${contested || ''}</span>
      </div>`;
    })
    .join('');
  const shown = (html.match(/class="row scenario"/g) ?? []).length;
  const all = state.data.traces.length;
  const count = shown !== all ? `<p class="muted count">${shown} of ${all} scenarios</p>` : '';
  $('scenario-list').innerHTML = count + (html || '<p class="empty">No scenarios match.</p>');
  for (const el of $('scenario-list').querySelectorAll('.row')) {
    el.onclick = () => {
      state.scenario = el.dataset.id;
      state.span = firstSpanOf(state.scenario);
      state.rawView = false;
      render();
    };
  }
}

function renderSpans() {
  const t = traceOf(state.scenario);
  const spans = visibleSpans(state.scenario);
  const legend = $('span-legend');
  legend.textContent = t && spans.length ? EVALUATORS.map((e) => e.short).join(' ') : '';
  legend.title = EVALUATORS.map((e) => `${e.short} = ${e.name}`).join(', ');

  if (!t) return void ($('span-list').innerHTML = '<p class="empty">Select a scenario.</p>');
  if (!spans.length) {
    return void ($('span-list').innerHTML = `<p class="empty">${
      state.onlyDisagree
        ? 'No contested commands in this scenario.'
        : 'The agent ran no shell commands in this scenario.'
    }</p>`);
  }

  $('span-list').innerHTML = spans
    .map((s) => {
      const per = resultsFor(s.span_id);
      const glyph = EVALUATORS.map((e) => {
        const lv = per[e.key]?.risk_level;
        return `<b class="${lv ?? 'none'}" title="${esc(e.name)}: ${esc(lv ?? 'no answer')}">${esc(lv ? LEVEL_INITIAL[lv] : '·')}</b>`;
      }).join('');
      return `<div class="row span" role="option" aria-selected="${s.span_id === state.span}"
          data-id="${esc(s.span_id)}" data-agreement="${agreementOf(s.span_id)}">
        <span class="cmd">${esc(s.command)}</span>
        <span class="flag" title="${isFlagged(s) ? 'did not exit cleanly' : ''}">${isFlagged(s) ? '!' : ''}</span>
        <span class="glyph">${glyph}</span>
      </div>`;
    })
    .join('');

  for (const el of $('span-list').querySelectorAll('.row')) {
    el.onclick = () => {
      state.span = el.dataset.id;
      state.rawView = false;
      renderSpans();
      renderDetail();
    };
  }
}

function renderDetail() {
  const t = traceOf(state.scenario);
  const span = t?.spans.find((s) => s.span_id === state.span);
  if (!t) return void ($('detail').innerHTML = '<p class="empty">Select a scenario.</p>');
  if (!span) {
    return void ($('detail').innerHTML = `<p class="empty">Select a command to see the three ratings.</p>`);
  }

  const per = resultsFor(span.span_id);
  const agreement = agreementOf(span.span_id);
  const flagged = isFlagged(span);

  $('detail').innerHTML = `
    <div class="detail-head">
      <h2>Command ${t.spans.indexOf(span) + 1} of ${t.spans.length}</h2>
      <span class="pill ${agreement}">${agreement === 'unanimous' ? 'all three agreed'
        : agreement === 'split' ? 'two agreed, one differed'
        : agreement === 'three_way' ? 'all three differed' : 'not all answered'}</span>
      ${flagged ? `<span class="muted">${span.executed
        ? `exited ${span.exit_code}`
        : `<b class="high">not run — ${esc(span.error_type ?? 'blocked')}</b>`}</span>` : ''}
    </div>

    ${state.rawView
      ? `<button id="toggle-raw">back to ratings</button>
         <pre>${esc(JSON.stringify({ span, results: per, raw_results: state.data.raw_results.filter((r) => r.span_id === span.span_id) }, null, 2))}</pre>`
      : `
      <div class="task">
        <span class="lab">Task given to the agent</span>
        <span class="task-title">${esc(t.title)}</span>
        <p class="task-prompt">${esc(t.prompt ?? '')}</p>
      </div>

      <h3>Command</h3>
      <pre class="cmd">${esc(span.command)}</pre>

      ${outputBlock(span, flagged)}

      <h3>How the three rated it</h3>
      <div class="ratings">${EVALUATORS.map((e) => rating(e, per[e.key])).join('')}</div>

      <details class="evidence">
        <summary>Evidence and methodology</summary>
        ${evidence(t, span, per)}
      </details>`}
  `;

  const toggle = $('toggle-raw');
  if (toggle) toggle.onclick = () => { state.rawView = false; renderDetail(); };
  const rawLink = $('open-raw');
  if (rawLink) rawLink.onclick = () => { state.rawView = true; renderDetail(); };
}

/**
 * Output is previewed, not hidden: a rating often turns on one line of it. A
 * command that failed or was blocked opens expanded, because that is exactly
 * when the output changes how the command reads.
 */
function outputBlock(span, flagged) {
  if (span.tool_output === null) {
    return `<h3>Output</h3><pre>${esc(span.guardrail?.reason ?? 'Not executed — no output.')}</pre>`;
  }
  const lines = span.tool_output.split('\n');
  const truncated = lines.length > OUTPUT_PREVIEW_LINES;
  const preview = lines.slice(0, OUTPUT_PREVIEW_LINES).join('\n');
  if (!truncated || flagged) {
    return `<h3>Output</h3><pre>${esc(span.tool_output)}</pre>`;
  }
  return `<h3>Output</h3>
    <pre>${esc(preview)}</pre>
    <details><summary>full output (${lines.length} lines)</summary>
      <pre>${esc(span.tool_output)}</pre></details>`;
}

function rating(evaluator, r) {
  if (!r) return `<div class="rating"><p class="who">${esc(evaluator.name)}</p>${na('no result stored')}</div>`;
  const probs = r.probabilities
    ? LEVELS.map((l) => [l, r.probabilities[l] ?? 0]).filter(([, v]) => v > 0.005) : null;
  const probBar = probs ? `<div class="bar probbar" role="img"
      aria-label="${esc(probs.map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(', '))}">${
      probs.map(([k, v]) => `<i class="${k}" style="width:${v * 100}%" title="${k} ${(v * 100).toFixed(0)}%">${
        v >= 0.14 ? `${(v * 100).toFixed(0)}%` : ''}</i>`).join('')}</div>
    <p class="probs">${esc(probs.map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(' · '))}</p>` : '';

  return `<div class="rating">
    <p class="who">${esc(evaluator.name)}${r.simulated ? ' <span class="split">simulated</span>' : ''}
      <span class="lat">${val(r.latency_ms, (v) => `${v} ms`, 'latency not recorded')}</span></p>
    <p class="level ${esc(r.risk_level ?? '')}">${esc(r.risk_level ?? 'no answer')}</p>
    ${probBar}
    <p class="why">${r.rationale
      ? esc(r.rationale)
      : na(r.rationale_availability === 'unsupported_by_evaluator'
          ? 'classifier — no explanation requested'
          : r.rationale_availability)}</p>
    ${r.error ? `<p class="lat"><span class="critical">error: ${esc(r.error.message)}</span></p>` : ''}
  </div>`;
}

/** Everything operational, in one place, out of the reading path. */
function evidence(t, span, per) {
  const meta = t.execution ?? {};
  const sandbox = meta.sandbox ?? {};
  const agent = meta.agent ?? state.data.run_meta?.agent ?? null;
  const derived = state.data.summary?.scenarios?.find((x) => x.scenario_id === t.scenario_id);

  const perModel = EVALUATORS.map((e) => {
    const r = per[e.key];
    if (!r) return '';
    return `<tr><td><code class="id">${esc(e.key)}</code></td>
      <td>${val(r.usage?.total_tokens, (v) => `${v} (in ${r.usage.input_tokens ?? '?'} / out ${r.usage.output_tokens ?? '?'})`, 'not reported')}</td>
      <td>${val(r.cost?.estimated_usd, (v) => `$${v.toFixed(6)} (${esc(r.cost.basis)})`, r.cost?.unavailable_reason ?? 'not available')}</td>
      <td><code class="id">${esc(r.result_id)}</code></td>
      <td><code class="id">${esc(r.rubric.rubric_version)} / ${esc(r.rubric.rubric_hash.slice(0, 12))}</code></td>
      <td><code class="id">${esc(r.submitted.submitted_prompt_hash.slice(0, 12))}</code></td></tr>`;
  }).join('');

  return `
    <h3>What was submitted</h3>
    <div class="kv">
      <span class="muted">raw tool input</span><span><code class="id">${esc(span.tool_input)}</code></span>
      <span class="muted">span id</span><span><code class="id">${esc(span.span_id)}</code></span>
      <span class="muted">exit code</span><span>${span.exit_code === null ? na('not executed') : span.exit_code}</span>
      <span class="muted">duration</span><span>${val(span.duration_ms, (v) => `${v} ms`, 'not recorded')}</span>
      <span class="muted">error type</span><span>${span.error_type ?? 'none'}</span>
      <span class="muted">guardrail</span><span>${esc(span.guardrail?.decision ?? '—')}${
        span.guardrail?.reason ? ` — ${esc(span.guardrail.reason)}` : ''}</span>
      <span class="muted">started</span><span>${val(span.started_at, esc, 'not recorded')}</span>
    </div>

    <h3>Tokens, cost and identifiers</h3>
    <table><tr><th>model</th><th>tokens</th><th>cost</th><th>result id</th><th>rubric</th><th>input hash</th></tr>${perModel}</table>

    <h3>Scenario and run</h3>
    <table>
      <tr><th>trace id</th><td><code class="id">${esc(t.trace_id)}</code></td></tr>
      <tr><th>scenario id</th><td><code class="id">${esc(t.scenario_id)}</code></td></tr>
      <tr><th>scenario</th><td>${esc(t.title)}
        <span class="muted">— the task this command came from</span></td></tr>
      <tr><th>requested action</th><td>${esc(t.requested_action)}</td></tr>
      <tr><th>prompt given to agent</th><td>${val(t.prompt)}
        <span class="muted">(never shown to the evaluators)</span></td></tr>
      <tr><th>agent's stated intent</th><td>${val(span.description, esc, 'none given')}
        <span class="muted">— the agent's own words for this command; the evaluators saw it
        inside the raw tool input</span></td></tr>
      <tr><th>agent claim</th><td>${val(t.agent_claim, esc, 'no final message captured')}</td></tr>
      <tr><th>outcome</th><td>${t.outcome.spans_executed}/${t.outcome.spans_total} executed ·
        ${t.outcome.spans_denied_by_guardrail} denied · ${t.outcome.spans_failed} nonzero exit ·
        refusal detected: ${t.outcome.agent_refused ? 'yes' : 'no'}</td></tr>
      <tr><th>design band</th><td>${esc(t.design_band)}
        <span class="muted">(author intent, never shown to a model, not ground truth)</span></td></tr>
      <tr><th>execution</th><td>${esc(meta.executor ?? '')} · mode ${esc(meta.mode ?? '')} ·
        ${meta.simulated ? 'simulated' : 'live'}</td></tr>
      <tr><th>agent config</th><td>${agent
        ? `${esc(agent.cli)} → ${esc(agent.provider_base_url)} · wire model ${esc(agent.wire_model)} · github auth: ${agent.github_auth_used ? 'yes' : 'no'}`
        : na('not recorded')}</td></tr>
      <tr><th>sandbox</th><td>${sandbox.sandbox_name
        ? `<code class="id">${esc(sandbox.sandbox_name)}</code> · region ${esc(sandbox.region ?? '?')} ·
           ${esc(sandbox.copilot_version ?? '?')} · artifacts retrieved before stop:
           ${sandbox.artifacts_retrieved_before_stop ? 'yes' : 'no'} · stopped: ${sandbox.stopped ? 'yes' : 'no'}`
        : na('no sandbox recorded')}</td></tr>
      ${derived ? `<tr><th>derived highest risk</th><td>${EVALUATORS.map((e) =>
        `${e.key}: <span class="pill ${derived.derived_highest_span_risk[e.key] ?? ''}">${esc(derived.derived_highest_span_risk[e.key] ?? 'n/a')}</span>`).join(' ')}
        <span class="muted">— derived locally from per-command ratings; no model saw a whole scenario</span></td></tr>` : ''}
    </table>

    <details>
      <summary>agent events (${(t.events ?? []).length})</summary>
      <pre>${esc((t.events ?? []).map((e) => `${e.at ?? ''}  ${e.type}${e.tool_name ? `  [${e.tool_name}]` : ''}`).join('\n') || 'no events recorded')}</pre>
    </details>

    <button id="open-raw">raw JSON for this command</button>`;
}

/* ---------------- repeatability view ----------------
 * A separate reading path, opened from the header. It never renders into the
 * ratings view, and the ratings view never reads from it.
 */

const REPEAT_BARS = [
  { key: 'stable', label: 'same every time' },
  { key: 'changed', label: 'changed at least once' },
];

function renderRepeat() {
  const el = $('repeat-view');
  const a = state.data.repeat_analysis;

  if (!a) {
    el.innerHTML = `<div class="repeat-wrap"><h2>Repeatability</h2>
      <p class="empty">No repeat study stored. It is produced by
      <code class="id">artifacts/repeatability/</code> and exists for the live run only.</p></div>`;
    return;
  }

  const m = state.data.repeat_meta ?? {};
  const sum = (pick) => Object.values(a.per_model).reduce((n, p) => n + (pick(p) ?? 0), 0);
  const ratings = a.total_repeat_calls + sum((p) => p.baseline_reference?.calls);
  const errors = sum((p) => p.errors);
  el.innerHTML = `
    <section class="sumstrip" aria-label="Repeatability summary">
      <h2 class="sum-head">Repeatability overview
        <span class="muted">${a.spans} commands · 3 models · ${a.observations_per_span_per_model} runs each
          · ${ratings} ratings${errors ? ` · <b class="high">${errors} errors</b>` : ' · no errors'}</span></h2>
      <div class="sumblock">
        <div class="lab">Same rating every time?</div>
        ${repeatChart(a)}
      </div>
      <div class="sumblock rp-cohort">
        <div class="lab">By original agreement</div>
        ${cohortBlock(a)}
      </div>
      <div class="sumblock">
        <div class="lab">Jev's probability movement</div>
        ${jevBlock(a)}
      </div>
    </section>

    <div class="explore-head"><h2>Per-command repeats</h2></div>
    <div class="repeat-wrap">
      <label class="muted opt">
        <input type="checkbox" id="repeat-only-changed" ${state.repeatOnlyChanged ? 'checked' : ''}>
        only commands where a rating changed
      </label>
      ${repeatMatrix()}

      <p class="rp-report" id="repeat-report">
        <a href="repeat-report.txt" target="_blank" rel="noopener">Open the full repeatability report</a>
        <span class="muted">— <code class="id">artifacts/repeatability/REPORT.md</code>, opens in a new tab</span>
      </p>

      ${repeatLimits()}

      <details class="evidence">
        <summary>Run metadata and provenance</summary>
        ${repeatMeta(a, m)}
      </details>
    </div>`;

  const cb = $('repeat-only-changed');
  if (cb) cb.onchange = (e) => {
    state.repeatOnlyChanged = e.target.checked;
    state.repeatExpanded = null;
    renderRepeat();
    $('repeat-only-changed')?.focus();
  };

  for (const b of el.querySelectorAll('.rp-toggle')) {
    b.onclick = () => {
      const spanId = b.dataset.span;
      state.repeatExpanded = state.repeatExpanded === spanId ? null : spanId;
      renderRepeat();
      [...el.querySelectorAll('.rp-toggle')].find((next) => next.dataset.span === spanId)?.focus();
    };
  }
  for (const b of el.querySelectorAll('.rp-open')) {
    b.onclick = () => openInRatings(b.dataset.scenario, b.dataset.span);
  }
}

/**
 * The setup as figures rather than a sentence: what was re-rated, how many
 * times, and how much of it came back clean.
 */

/**
 * What the study cannot show. Four separate limits, so they are read as four
 * things rather than skimmed as one paragraph of hedging.
 */
function repeatLimits() {
  const limits = [
    ['Consistency, not accuracy', 'a model can repeat itself and still be wrong'],
    ['No ground truth', 'no independently established correct label exists for these commands'],
    ['Slugs, not versions', 'the same model slug came back every time, but the Gateway exposes no build id'],
    ['Nothing re-executed', 'no sandbox and no command run again; only the stored evidence was re-sent'],
  ];
  return `<details class="evidence">
    <summary>What this does and doesn't show</summary>
    <ul class="rp-limits">${limits.map(([lead, rest]) =>
      `<li><b>${lead}</b> — ${rest}</li>`).join('')}</ul>
  </details>`;
}

/**
 * One row per model. The bar is decorative: every number it encodes is also
 * printed beside it, and the row carries a full text equivalent.
 */
function repeatChart(a) {
  const rows = EVALUATORS.map((e) => {
    const p = a.per_model[e.key];
    if (!p) return '';
    const of = p.spans_scored || 1;
    const segs = REPEAT_BARS.map((b) => {
      const n = b.key === 'stable' ? p.stable_all_three : p.changed_all_three;
      if (!n) return '';
      const pct = (n / of) * 100;
      return `<i class="rp-${b.key}" style="width:${pct}%" title="${n} ${b.label}">${pct >= 9 ? n : ''}</i>`;
    }).join('');
    const text = `${e.name}: ${p.stable_all_three} of ${p.spans_scored} commands rated the same every time, ${p.changed_all_three} changed at least once`;
    return `<div class="sumrow"><span class="who" title="${esc(e.name)}">${esc(e.name)}</span>
      <span class="bar" role="img" aria-label="${esc(text)}">${segs}</span>
      <span class="num">${p.stable_all_three}/${p.spans_scored} same</span></div>`;
  }).join('');
  const key = REPEAT_BARS.map((b) => `<li><span class="sw rp-${b.key}"></span>${esc(b.label)}</li>`).join('');
  return `${rows}<ul class="level-key">${key}</ul>`;
}

function cohortBlock(a) {
  const c = a.cohort_rollup ?? {};
  const cell = (coh, key) => {
    const v = c[coh]?.per_model?.[key];
    const n = c[coh]?.spans;
    return v ? `<b>${v.stable}</b>/${n} same` : na('n/a');
  };
  const rows = EVALUATORS.map((e) => `<div class="sumrow">
      <span class="who" title="${esc(e.name)}">${esc(e.name)}</span>
      <span class="num">${cell('unanimous', e.key)}</span>
      <span class="num">${cell('disputed', e.key)}</span></div>`).join('');
  return `<div class="cols"><span></span>
      <span class="tip" tabindex="0" data-tip="commands where all three models originally gave the same rating">agreed (${c.unanimous?.spans ?? '?'})</span>
      <span class="tip" tabindex="0" data-tip="commands where the three models originally disagreed">disputed (${c.disputed?.spans ?? '?'})</span></div>
    ${rows}`;
}

/** Probabilities are a Jev-only field; the other two never returned any. */
function jevBlock(a) {
  const o = a.jev_probabilities_overall;
  const c = a.cohort_rollup ?? {};
  if (!o) return na('not available');
  const f = (v) => v == null ? na('none') : v;
  const row = (label, p) => p ? `<tr><td>${esc(label)}</td><td class="fig">${p.spans}</td>
      <td class="fig">${f(p.chosen_prob_median_of_medians)}</td>
      <td class="fig">${f(p.chosen_prob_spread_median)}</td>
      <td class="fig">${f(p.chosen_prob_spread_max)}</td></tr>` : '';
  return `<table class="figtable">
      <tr><th class="tip" tabindex="0" data-tip="whether the three models originally agreed on the command">originally</th>
          <th class="fig">n</th>
          <th class="fig tip" tabindex="0" data-tip="median probability Jev put on its chosen rating">median p</th>
          <th class="fig tip" tabindex="0" data-tip="median spread of that probability across the ${a.observations_per_span_per_model} runs">median move</th>
          <th class="fig tip" tabindex="0" data-tip="largest spread of that probability across the ${a.observations_per_span_per_model} runs">largest</th></tr>
      ${row('all', o)}
      ${row('agreed', c.unanimous?.jev_probabilities)}
      ${row('disputed', c.disputed?.jev_probabilities)}
    </table>`;
}

/** How the per-command movement is distributed, in one sentence. */

/**
 * The repeat study as a matrix: one command per row, three cells per model for
 * the original run and the two repeats.
 *
 * A cell's letter is the rating, so the grid reads without colour. A model whose
 * three cells are not all the same is outlined as a group, which replaces a
 * per-cell "same/changed" tag that would otherwise repeat 93 times. Each model
 * group carries one spoken label, and any row can be expanded for the full
 * command, the exact sequences, and a jump into the ratings view.
 */
function repeatMatrix() {
  const rows = (state.data.repeat_per_span ?? []).filter((r) =>
    !state.repeatOnlyChanged || EVALUATORS.some((e) => r.models?.[e.key]?.all_three_stable === false));

  if (!rows.length) {
    return `<p class="empty">${state.repeatOnlyChanged
      ? 'No command changed rating for any model.'
      : 'No per-command repeat records stored.'}</p>`;
  }

  const body = rows.map((r) => {
    const open = state.repeatExpanded === r.span_id;
    const cells = EVALUATORS.map((e) => {
      const rep = r.models?.[e.key];
      if (!rep) return `<td class="rp-group">${na('none')}</td>`;
      const seen = repeatLabels(rep);
      const changed = rep.all_three_stable === false;
      const boxes = seen
        .map((lv) => `<i class="rp-cell ${esc(lv)}" aria-hidden="true">${LEVEL_INITIAL[lv] ?? '?'}</i>`)
        .join('');
      // One spoken label per model group; the letters themselves are hidden
      // from assistive tech so the row is not read out as "L L L".
      const spoken = `${e.name}: ${seen.join(', then ')}${changed ? ' — changed' : ' — same every time'}`;
      return `<td class="rp-group${changed ? ' changed' : ''}">
        <span role="img" aria-label="${esc(spoken)}">${boxes}</span></td>`;
    }).join('');

    const short = r.command.length > 58 ? r.command.slice(0, 58) + '…' : r.command;
    return `<tr class="rp-mrow${open ? ' open' : ''}">
        <th scope="row">
          <button type="button" class="rp-toggle" data-span="${esc(r.span_id)}"
                  aria-expanded="${open}" aria-controls="exp-${esc(r.span_id)}">
            <span class="rp-caret" aria-hidden="true">${open ? '▾' : '▸'}</span>
            <code class="id">${esc(short)}</code>
          </button>
        </th>
        ${cells}
      </tr>
      <tr class="rp-exprow" id="exp-${esc(r.span_id)}" ${open ? '' : 'hidden'}>
        <td colspan="${EVALUATORS.length + 1}">${open ? expandedRow(r) : ''}</td>
      </tr>`;
  }).join('');

  const head = EVALUATORS.map((e) =>
    `<th scope="col" class="rp-group-head">${esc(e.name)}
      <span class="sub">orig · r1 · r2</span></th>`).join('');

  return `${matrixLegend()}<div class="rp-scroll">
    <table class="rp-matrix">
      <caption class="note">${rows.length} command${rows.length === 1 ? '' : 's'}.
        Select one to see what each model said.</caption>
      <thead><tr><th scope="col">command</th>${head}</tr></thead>
      <tbody>${body}</tbody>
    </table></div>`;
}

function matrixLegend() {
  const levels = LEVELS.map((l) =>
    `<li><i class="rp-cell ${l}" aria-hidden="true">${LEVEL_INITIAL[l]}</i> ${l}</li>`).join('');
  return `<ul class="rp-legend">
    ${levels}
    <li class="rp-legend-changed"><span class="rp-group changed" aria-hidden="true"><i class="rp-cell low">L</i></span>
      outlined: this model changed its rating</li>
  </ul>`;
}

/** The full detail for one command, shown only when its row is expanded. */
function expandedRow(r) {
  const cards = EVALUATORS.map((e) => {
    const rep = r.models?.[e.key];
    if (!rep) return `<section class="rp-card"><header><h4>${esc(e.name)}</h4></header>
      <div class="rp-card-body">${na('no answer')}</div></section>`;
    const changed = rep.all_three_stable === false;
    return `<section class="rp-card${changed ? ' changed' : ''}">
      <header>
        <h4>${esc(e.name)}</h4>
        <span class="rp-tag ${changed ? 'changed' : 'same'}">${changed ? 'changed' : 'same every time'}</span>
      </header>
      <div class="rp-card-body">
        ${RUNS.map((run) => runLine(rep, run)).join('')}
      </div>
      ${cardFoot(e.key, rep)}
    </section>`;
  }).join('');

  return `<div class="rp-exp">
    <pre class="cmd">${esc(r.command)}</pre>
    <div class="rp-cards">${cards}</div>
    <footer class="rp-exp-foot">
      <span><span class="muted">originally</span> ${esc(AGREEMENT_TEXT[r.baseline_cohort] ?? r.baseline_cohort)}</span>
      <span><span class="muted">span</span> <code class="id">${esc(r.span_id)}</code></span>
      <button type="button" class="rp-open" data-span="${esc(r.span_id)}" data-scenario="${esc(r.scenario_id)}">
        Open in ratings view →
      </button>
    </footer>
  </div>`;
}

/** The card's last line: Jev's probability movement, or why there is no prose. */
function cardFoot(key, rep) {
  const spread = key === 'jev' ? rep.chosen_prob?.spread : null;
  const parts = [];
  if (spread != null) parts.push(`<b>${spread.toFixed(2)}</b> widest gap between the three probabilities`);
  if (rep.rationale_availability === 'unsupported_by_evaluator')
    parts.push('Returns a level and calibrated probabilities, not prose.');
  return parts.length ? `<div class="rp-card-foot">${parts.join(' · ')}</div>` : '';
}

/** One observation: the level it gave, then why — or, for Jev, with what probability. */
function runLine(rep, run) {
  const lv = rep.labels?.[run.key];
  if (!lv) return `<div class="rp-run"><span class="rp-when">${run.label}</span>
    <span class="rp-what">${na('no answer')}</span></div>`;
  const p = rep.probabilities?.[run.key]?.[lv];
  const said = rep.rationales?.[run.key];
  return `<div class="rp-run">
    <span class="rp-when">${run.label}</span>
    <span class="rp-what">
      <i class="rp-cell ${esc(lv)}" aria-hidden="true">${LEVEL_INITIAL[lv] ?? '?'}</i>
      <b class="${esc(lv)}">${esc(lv)}</b>
      ${p != null ? `<span class="rp-prob">p ${p.toFixed(2)}</span>` : ''}
    </span>
    ${said ? `<p class="rp-why">${esc(said)}</p>` : ''}
  </div>`;
}

/** How far Jev's probability on its own answer travelled across the three runs. */

/** Everything operational, out of the reading path. */
function repeatMeta(a, m) {
  const perModel = EVALUATORS.map((e) => {
    const p = a.per_model[e.key];
    if (!p) return '';
    return `<tr><td><code class="id">${esc(e.key)}</code><br><span class="muted">${esc(p.model)}</span></td>
      <td>${p.repeat_calls}</td><td>${p.errors}</td>
      <td>${val(p.usage?.total_tokens, (v) => v.toLocaleString(), 'not reported')}</td>
      <td>${val(p.cost?.total_usd, (v) => `$${v.toFixed(8)}`, 'not available')}
        <br><span class="muted">${esc((p.cost?.basis ?? []).join(', '))}</span></td>
      <td>${val(p.latency_ms?.median, (v) => `${v} ms`, 'not recorded')}</td>
      <td><code class="id">${esc((p.replay_checks?.observed_model_ids ?? []).join(', ') || 'none')}</code></td></tr>`;
  }).join('');

  return `<table>
      <tr><th>model</th><th>calls</th><th>errors</th><th>tokens</th><th>cost</th>
          <th>median latency</th><th>slug returned</th></tr>${perModel}</table>
    <div class="kv">
      <span class="muted">repeat run marker</span><span><code class="id">${esc(a.repeat_run_marker)}</code></span>
      <span class="muted">original run marker</span><span><code class="id">${esc(a.baseline_run_marker)}</code></span>
      <span class="muted">calls</span><span>${a.total_repeat_calls}</span>
      <span class="muted">rubric</span><span><code class="id">${esc(m.rubric_version ?? '?')} / ${esc((m.rubric_hash ?? '').slice(0, 12))}</code>
        ${m.rubric_hash_matches_baseline ? '— identical to the original run' : ''}</span>
      <span class="muted">evidence sent</span><span>byte-identical to the original run on all
        ${a.total_repeat_calls} calls</span>
      <span class="muted">agent</span><span>not re-run; no command was executed again</span>
    </div>
    <p class="note"><b>Model identity.</b> The gateway returns a model slug and no version or
      build id. The slug was the same in both runs, but that does not prove the weights behind
      it were. Every figure here assumes they were, and that assumption cannot be checked.</p>
    <p class="note">Source: <code class="id">artifacts/repeatability/</code> —
      <a href="repeat-report.txt" target="_blank" rel="noopener">REPORT.md</a>.</p>`;
}

/**
 * Hand a command over to the ratings view. The "only disagreed" filter is
 * cleared first, otherwise a command that all three agreed on would be selected
 * into a list that does not contain it.
 */
function openInRatings(scenarioId, spanId) {
  if (state.filter) {
    state.filter = '';
    $('filter').value = '';
  }
  if (state.onlyDisagree) {
    state.onlyDisagree = false;
    $('only-disagree').checked = false;
  }
  state.scenario = scenarioId;
  state.span = spanId;
  state.rawView = false;
  setView('ratings');
  renderScenarios();
  renderSpans();
  renderDetail();
  $('detail').scrollIntoView({ block: 'start' });
}

/* ---------------- controls ---------------- */

$('view-ratings').onclick = () => setView('ratings');
$('view-repeat').onclick = () => setView('repeat');
function setView(view, { updateHash = true } = {}) {
  state.view = view;
  const onRepeat = view === 'repeat';
  // The view is addressable, so it can be linked to and reloaded in place.
  if (updateHash) {
    const want = onRepeat ? '#repeatability' : '';
    const alreadyInView = onRepeat && location.hash.startsWith('#repeat');
    if (!alreadyInView && location.hash !== want) {
      history.replaceState(null, '', want || location.pathname);
    }
  }
  $('ratings-view').hidden = onRepeat;
  $('repeat-view').hidden = !onRepeat;
  for (const [id, on] of [['view-ratings', !onRepeat], ['view-repeat', onRepeat]]) {
    $(id).classList.toggle('active', on);
    $(id).setAttribute('aria-selected', String(on));
  }
}

$('filter').oninput = (e) => {
  state.filter = e.target.value.toLowerCase();
  renderScenarios();
};

$('only-disagree').onchange = (e) => {
  state.onlyDisagree = e.target.checked;
  if (!visibleSpans(state.scenario).some((s) => s.span_id === state.span)) {
    state.span = firstSpanOf(state.scenario);
  }
  renderScenarios();
  renderSpans();
  renderDetail();
};

$('span-list').addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
  e.preventDefault();
  const spans = visibleSpans(state.scenario);
  const i = spans.findIndex((s) => s.span_id === state.span);
  const next = e.key === 'ArrowDown' ? Math.min(spans.length - 1, i + 1) : Math.max(0, i - 1);
  if (spans[next]) {
    state.span = spans[next].span_id;
    state.rawView = false;
    renderSpans();
    renderDetail();
    $('span-list').querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' });
  }
});

/**
 * Open whichever view the URL asks for, defaulting to ratings. Any #repeat…
 * anchor opens the repeat view, so links to a section inside it work too.
 */
function applyHash() {
  setView(location.hash.startsWith('#repeat') ? 'repeat' : 'ratings', { updateHash: false });
  // The view renders after navigation, so the browser has nothing to scroll to
  // when the anchor is first resolved. Do it once the target exists.
  const id = location.hash.slice(1);
  if (id && id !== 'repeatability') {
    document.getElementById(id)?.scrollIntoView({ block: 'start' });
  }
}
addEventListener('hashchange', applyHash);

await load();
applyHash();
