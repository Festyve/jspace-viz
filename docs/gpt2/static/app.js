// Copyright 2026 Michael Zhang
// SPDX-License-Identifier: Apache-2.0
// Live J-space visualizer frontend. No build step, no dependencies.

const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let data = null;          // last /api/read response
let mode = "jlens";
let pinned = [];          // [{id, str}]
let rankViewId = null;    // token id whose ranks the grid shows, or null
let STATIC = false;       // no backend: serve precomputed example grids
let INFO = null;          // /api/info (or data/index.json) payload
let selectedSlug = null;  // active example in static mode
let colLimit = null;      // static demo: cap visible grid columns during reveal

// Render a token for display: make whitespace visible, trim length.
function tokStr(s) {
  if (s === undefined) return "?";
  let out = s.replace(/\n/g, "⏎").replace(/\t/g, "⇥");
  if (out.startsWith(" ")) out = "·" + out.slice(1);
  if (out.trim() === "") out = out.replace(/ /g, "·") || "∅";
  return out.length > 12 ? out.slice(0, 11) + "…" : out;
}

function setStatus(msg, err = false, html = false) {
  const el = $("status");
  if (html) el.innerHTML = msg;
  else el.textContent = msg;
  el.className = err ? "err" : "";
}

// The model's actual next words (output row, last position) — shown in the
// status line so "what does it answer" never has to be hunted for.
function modelSays() {
  if (data.continuation) {
    const text = data.continuation.replace(/\s+/g, " ").trim();
    return `model continues: “…${text}”`;
  }
  const out = data.grid[data.grid.length - 1];
  const t = data.seq_len - 1;
  const words = out.top_ids[t].slice(0, 3).map((id) => tokStr(data.vocab[id]));
  return `model says: ${words.map((w) => `“${w}”`).join(" ")}`;
}

async function init() {
  try {
    const res = await fetch("api/info");
    if (!res.ok) throw new Error();
    INFO = await res.json();
  } catch {
    // No backend — static demo mode: precomputed example grids only.
    STATIC = true;
    INFO = await (await fetch("data/index.json")).json();
    INFO.device = "precomputed";
    // These need a live model run; the demo is precomputed. Keep them visible
    // but locked, and explain why (hover + click) instead of doing nothing.
    $("live-label").hidden = false;
    for (const id of ["topk", "chat", "live"]) {
      const label = $(id).closest("label");
      $(id).readOnly = true; // blocks typing in the number box (no-op on checkboxes)
      label.classList.add("demo-locked");
      label.title = "Precomputed demo — clone and run it locally to use this";
      label.addEventListener("click", (e) => {
        e.preventDefault(); // don't toggle/spin — explain instead
        setStatus("top-k · chat · live only work when you run it locally — the demo is precomputed");
      });
    }
  }
  for (const link of INFO.links || []) {
    const a = document.createElement("a");
    a.href = link.url;
    a.className = "chip";
    a.textContent = link.name;
    document.querySelector("header").appendChild(a);
  }
  $("model-chip").textContent = `${INFO.model_id} · ${INFO.n_layers}L · d=${INFO.d_model} · ${INFO.device}`;
  $("lens-chip").textContent = `lens: ${INFO.fitted_layers.length} layers, ${INFO.lens_n_prompts} prompts`;
  INFO.examples.forEach((ex, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = ex.name;
    $("examples").appendChild(opt);
  });
  // Shareable links: /?prompt=... pre-fills and reads immediately.
  const shared = new URLSearchParams(location.search).get("prompt");
  if (shared && !STATIC) {
    $("prompt").value = shared;
    read();
  } else {
    renderWelcome();
  }
}

// Blank-slate hint shown before the first read: what this is, plus the
// examples as one-click chips.
function renderWelcome() {
  const chips = INFO.examples
    .map((ex, i) => `<button class="ex-chip" data-i="${i}">${esc(ex.name)}</button>`)
    .join(" ");
  const typeHint = STATIC
    ? 'Pick an example below. To type <em>your own</em> prompts, clone the repo — <a href="https://github.com/Festyve/jspace-viz" target="_blank" rel="noopener">github.com/Festyve/jspace-viz</a> — and run it locally (two commands).'
    : "Type any prompt above and hit <b>Read</b> (⌘↵), or start from an example:";
  $("grid").innerHTML =
    `<div class="welcome"><p>Every cell will show what a layer of <b>${esc(INFO.model_id)}</b> ` +
    `is <i>disposed to say</i> at each word of your prompt.</p>` +
    `<p>${typeHint}</p><p>${chips}</p></div>`;
}
$("grid").addEventListener("click", (e) => {
  const chip = e.target.closest(".ex-chip");
  if (!chip) return;
  loadExample(+chip.dataset.i);
});

// Selecting an example types it out while the model reads along, so you watch
// the workspace assemble as context accumulates. The static demo has no
// backend, but the model is causal — each precomputed column equals a fresh
// read of that prefix — so we reveal columns as the words are typed, same
// effect without a server.
let playing = false;
let playToken = 0;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function loadExample(i) {
  const ex = INFO.examples[i];
  $("examples").value = String(i);
  const token = ++playToken;
  playing = true;
  const ta = $("prompt");
  ta.value = "";

  // Static demo: fetch the full precomputed grid up front, then reveal it
  // column by column as the prompt types out (see note above). tokEnd[t] is the
  // typed-character count at which column t's token is fully on screen.
  let tokEnd = null;
  if (STATIC) {
    selectedSlug = ex.slug;
    data = await (await fetch(`data/${selectedSlug}_${mode}.json`)).json();
    if (token !== playToken) return;
    applyStaticPins();
    tokEnd = [];
    let acc = 0;
    for (let t = 0; t < data.seq_len; t++) {
      if (t > 0) acc += (data.vocab[data.context_ids[t]] || "").length;
      tokEnd[t] = acc; // the begin-of-sequence token adds no typed characters
    }
    colLimit = 1; // show that first column immediately, then grow with typing
    renderGrid();
    renderWorkspace();
  }

  // Clock-based so playback stays smooth at any frame rate and can't stall
  // when the browser throttles timers in unfocused tabs.
  const CHARS_PER_SEC = 70;
  const start = performance.now();
  let wordsSinceRead = 0;
  while (true) {
    if (token !== playToken) { colLimit = null; return; } // superseded
    const target = Math.min(
      ex.prompt.length,
      Math.floor(((performance.now() - start) / 1000) * CHARS_PER_SEC),
    );
    if (target > ta.value.length) {
      const added = ex.prompt.slice(ta.value.length, target);
      ta.value = ex.prompt.slice(0, target);
      ta.scrollTop = ta.scrollHeight;
      if (STATIC) {
        // grow the grid to every column whose token has now been typed
        let n = colLimit;
        while (n < data.seq_len && tokEnd[n] <= target) n++;
        if (n !== colLimit) {
          colLimit = Math.max(1, n);
          renderGrid();
          renderWorkspace();
        }
      } else {
        wordsSinceRead += (added.match(/\s+/g) || []).length;
        const nWords = ta.value.trim().split(/\s+/).length;
        if ($("live").checked && !reading && wordsSinceRead >= 2 && nWords >= 4) {
          wordsSinceRead = 0;
          read({ live: true });
        }
      }
    }
    if (target >= ex.prompt.length) break;
    await sleep(16);
  }
  playing = false;
  if (token !== playToken) { colLimit = null; return; }
  if (STATIC) {
    colLimit = null; // reveal the full grid now that the prompt is complete
    renderGrid();
    renderWorkspace();
    renderMetrics();
    setStatus(
      `${data.continuation ? modelSays() + " · " : ""}` +
      `${data.seq_len} tokens × ${data.layers.length} layers · precomputed demo`,
    );
  } else {
    while (reading) await sleep(100); // let the last live read settle
    read();
  }
}

// Static mode: ranks were precomputed for every token in any top-k cell;
// materialize pinned_ranks rows in the shape the renderer expects.
function applyStaticPins() {
  if (!pinned.length) return;
  data.grid.forEach((row, li) => {
    row.pinned_ranks = [];
    for (let t = 0; t < data.seq_len; t++) {
      row.pinned_ranks.push(pinned.map((p) => data.ranks[p.id]?.[li]?.[t] ?? null));
    }
  });
}

let reading = false;   // one read in flight at a time; latest text wins after
let rerunLive = false;

async function read(opts = {}) {
  if (!$("prompt").value.trim()) {
    renderWelcome();
    setStatus("type a prompt first, or pick an example");
    return;
  }
  if (reading) { rerunLive = true; return; }
  reading = true;
  if (!opts.live) setStatus("reading…");
  $("read").disabled = true;
  const t0 = performance.now();
  try {
    if (STATIC) {
      // Only precomputed prompts exist here; anything else → honest pointer.
      const match = INFO.examples.find((ex) => ex.prompt === $("prompt").value);
      if (!match) {
        setStatus(
          'custom prompts need a local clone — <a href="https://github.com/Festyve/jspace-viz" target="_blank" rel="noopener">github.com/Festyve/jspace-viz</a>',
          true, true,
        );
        $("read").disabled = false;
        return;
      }
      selectedSlug = match.slug;
      data = await (await fetch(`data/${selectedSlug}_${mode}.json`)).json();
      applyStaticPins();
      renderGrid();
      renderWorkspace();
      renderMetrics();
      setStatus(`${modelSays()} · ${data.seq_len} tokens × ${data.layers.length} layers · precomputed demo`);
      return;
    }
    const res = await fetch("api/read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: $("prompt").value,
        mode,
        top_k: +$("topk").value,
        chat: $("chat").checked,
        pinned_ids: pinned.map((p) => p.id),
        continuation: !opts.live,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    data = await res.json();
    renderGrid();
    renderWorkspace();
    renderMetrics();
    if (opts.live && !playing) {
      for (const id of ["workspace-panel", "grid-wrap"]) {
        const el = $(id);
        el.classList.remove("flash");
        void el.offsetWidth; // restart the animation
        el.classList.add("flash");
      }
    }
    const says = data.continuation ? `${modelSays()} · ` : "";
    setStatus(`${says}${data.seq_len} tokens × ${data.layers.length} layers · ${((performance.now() - t0) / 1000).toFixed(1)}s${opts.live ? " · live" : ""}`);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    $("read").disabled = false;
    reading = false;
    if (rerunLive) {
      rerunLive = false;
      read({ live: true });
    }
  }
}

// Background for a top-1 probability (0..1) — accent blue ramp.
function probColor(p) {
  const a = Math.min(0.85, Math.pow(p, 0.6));
  return `rgba(56, 139, 253, ${a.toFixed(3)})`;
}
// Background for a rank (0 = top) — orange ramp, fades out past ~1000.
function rankColor(r) {
  const a = Math.max(0, 0.9 - 0.9 * (Math.log10(r + 1) / 3));
  return `rgba(255, 166, 87, ${a.toFixed(3)})`;
}

function renderGrid() {
  const grid = $("grid");
  const { layers, grid: rows, context_ids, vocab, seq_len } = data;
  const cols = colLimit == null ? seq_len : Math.min(seq_len, colLimit);
  grid.style.gridTemplateColumns = `auto repeat(${cols}, max-content)`;
  // Columns past prompt_len are the model's own generated words — tinted, so
  // you can see what it was thinking *while it spoke*.
  const promptLen = data.prompt_len ?? seq_len;
  const genCls = (t) => (t === promptLen ? " gen genb" : t > promptLen ? " gen" : "");
  const parts = ['<div class="hcell corner">layer ╲ pos</div>'];
  for (let t = 0; t < cols; t++) {
    const label = t === promptLen ? '<span class="idx">↳ model says</span>' : `<span class="idx">${t}</span>`;
    parts.push(`<div class="hcell${genCls(t)}">${label}${esc(tokStr(vocab[context_ids[t]]))}</div>`);
  }
  for (let li = 0; li < layers.length; li++) {
    const row = rows[li];
    const cls = row.is_output ? " outrow" : "";
    parts.push(`<div class="lcell${cls}">${row.is_output ? "output" : "L" + row.layer}</div>`);
    for (let t = 0; t < cols; t++) {
      let text, bg;
      const pi = rankViewId === null ? -1 : pinned.findIndex((p) => p.id === rankViewId);
      if (pi >= 0 && row.pinned_ranks && row.pinned_ranks[t][pi] != null) {
        const r = row.pinned_ranks[t][pi];
        text = r === 0 ? "★0" : String(r);
        bg = rankColor(r);
      } else if (pi >= 0 && row.pinned_ranks) {
        text = "–";
        bg = "transparent";
      } else {
        text = tokStr(vocab[row.top_ids[t][0]]);
        bg = probColor(row.top_probs[t][0]);
      }
      parts.push(`<div class="cell${cls}${genCls(t)}" data-l="${li}" data-t="${t}" style="background:${bg}">${esc(text)}</div>`);
    }
  }
  const wrap = $("grid-wrap");
  const { scrollLeft, scrollTop } = wrap;
  grid.innerHTML = parts.join("");
  if (playing) {
    // follow the newest column while an example types itself out
    wrap.scrollLeft = wrap.scrollWidth;
  } else {
    wrap.scrollLeft = scrollLeft;
    wrap.scrollTop = scrollTop;
  }
  // The generated columns are the highlight but sit far to the right — offer a
  // jump when they exist and aren't already the whole grid.
  $("jump-gen").hidden = !(promptLen < seq_len);
}

// Scroll the grid so the "↳ model says" boundary sits near the left edge,
// keeping a couple of prompt columns visible for context. Uses live rects so
// it's immune to the sticky-header / CSS-grid offsetParent quirks.
function scrollToGeneration() {
  const boundary = document.querySelector(".hcell.genb");
  if (!boundary) return;
  const wrap = $("grid-wrap");
  const delta = boundary.getBoundingClientRect().left - wrap.getBoundingClientRect().left - 160;
  wrap.scrollLeft = Math.max(0, wrap.scrollLeft + delta);
}
$("jump-gen").addEventListener("click", scrollToGeneration);

function renderPinned() {
  $("pinned-row").hidden = pinned.length === 0;
  $("view-top1").hidden = rankViewId === null;
  $("pinned-chips").innerHTML = pinned
    .map(
      (p) =>
        `<span class="pin-chip${p.id === rankViewId ? " sel" : ""}" data-id="${p.id}">` +
        `${esc(tokStr(p.str))}<span class="x" data-x="${p.id}">✕</span></span>`
    )
    .join("");
}

// "What is it thinking about": aggregate mid-band lens tokens that are real
// words, not words from the prompt — i.e. the J-space content. Click to trace.
const STOP = new Set("the a an is are was were be been of to in on at for and or but not it its this that with as by from he she they we you i his her their".split(" "));
function renderWorkspace() {
  const { grid, vocab, context_ids, seq_len } = data;
  const cols = colLimit == null ? seq_len : Math.min(seq_len, colLimit);
  const fitted = grid.filter((r) => !r.is_output);
  const band = fitted.slice(Math.floor(fitted.length * 0.25), Math.ceil(fitted.length * 0.85));
  const promptWords = new Set(context_ids.slice(0, cols).map((id) => (vocab[id] || "").trim().toLowerCase()));
  const scores = new Map(); // word -> {score, id, str}
  for (const row of band) {
    for (let t = 4; t < cols; t++) {
      row.top_ids[t].forEach((id, i) => {
        const str = vocab[id] || "";
        const w = str.trim().toLowerCase();
        // word-start tokens only (leading space in BPE) — filters subword junk
        if (!str.startsWith(" ") && !str.startsWith("▁")) return;
        if (!/^[a-z][a-z'’-]{2,}$/i.test(str.trim())) return;
        if (promptWords.has(w) || STOP.has(w)) return;
        const p = row.top_probs[t][i];
        if (p < 0.03) return; // diffuse noise doesn't accumulate
        const e = scores.get(w) || { score: 0, id, str };
        e.score += p;
        if (!scores.has(w)) scores.set(w, e);
      });
    }
  }
  // Subtract each word's baseline hum on neutral text: glitch tokens appear
  // on every prompt and cancel out; prompt-specific content survives.
  const BASE = INFO.chip_baseline || {};
  const top = [...scores.entries()]
    .map(([w, e]) => ({ ...e, adj: e.score - (BASE[w] || 0) }))
    .filter((e) => e.adj > 0.15)
    .sort((a, b) => b.adj - a.adj)
    .slice(0, 10);
  const panel = $("workspace-panel");
  panel.hidden = top.length === 0;
  const previous = renderWorkspace._last || new Set();
  renderWorkspace._last = new Set(top.map((e) => e.str.trim().toLowerCase()));
  $("workspace-chips").innerHTML = top
    .map((e) => {
      const word = e.str.trim();
      const pop = previous.has(word.toLowerCase()) ? "" : " pop";
      return `<span class="ws-chip${pop}" data-id="${e.id}">${esc(word)}<span class="score">${e.adj.toFixed(1)}</span></span>`;
    })
    .join("");
}
$("workspace-chips").addEventListener("click", (e) => {
  const chip = e.target.closest(".ws-chip");
  if (!chip || !data) return;
  const id = +chip.dataset.id;
  if (!pinned.some((p) => p.id === id)) {
    if (pinned.length >= 16) pinned.shift();
    pinned.push({ id, str: data.vocab[id] ?? "?" });
  }
  rankViewId = id;
  renderPinned();
  read();
});

function renderMetrics() {
  const ms = data.layer_metrics;
  drawChart($("m-acc"), ms.map((m) => m.next_token_acc), 0, 1);
  drawChart($("m-kurt"), ms.map((m) => m.mean_kurtosis), 0, null, true);
  drawChart($("m-auto"), ms.map((m) => m.top1_autocorr), 0, 1);
}

// Minimal SVG line chart; optional shaded "workspace band" where the series
// exceeds half its max (heuristic, only drawn when shadeBand is set).
function drawChart(svg, ys, ymin, ymax, shadeBand = false) {
  const W = svg.clientWidth || 280, H = 110, padL = 34, padB = 16, padT = 6;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const lo = ymin ?? Math.min(...ys), hi = (ymax ?? Math.max(...ys)) || 1;
  const x = (i) => padL + (i * (W - padL - 6)) / Math.max(1, ys.length - 1);
  const y = (v) => padT + (H - padB - padT) * (1 - (v - lo) / (hi - lo || 1));
  let out = "";
  if (shadeBand) {
    const cut = Math.max(...ys) / 2;
    let start = null;
    for (let i = 0; i <= ys.length; i++) {
      const inBand = i < ys.length && ys[i] >= cut;
      if (inBand && start === null) start = i;
      if (!inBand && start !== null) {
        out += `<rect x="${x(start)}" y="${padT}" width="${x(i - 1) - x(start)}" height="${H - padB - padT}" fill="rgba(255,166,87,.12)"/>`;
        start = null;
      }
    }
  }
  out += `<line x1="${padL}" y1="${H - padB}" x2="${W - 6}" y2="${H - padB}" stroke="#30363d"/>`;
  out += `<text x="${padL - 4}" y="${y(hi) + 4}" fill="#8b949e" font-size="9" text-anchor="end">${hi.toFixed(hi >= 10 ? 0 : 2)}</text>`;
  out += `<text x="${padL - 4}" y="${H - padB}" fill="#8b949e" font-size="9" text-anchor="end">${lo.toFixed(lo >= 10 ? 0 : 2)}</text>`;
  const lastFitted = ys.length - 2; // final entry is the output row (J = I)
  out += `<polyline fill="none" stroke="#58a6ff" stroke-width="1.5" points="${ys
    .slice(0, lastFitted + 1).map((v, i) => `${x(i)},${y(v)}`).join(" ")}"/>`;
  out += `<circle cx="${x(ys.length - 1)}" cy="${y(ys[ys.length - 1])}" r="2.5" fill="#ffa657"/>`;
  const L = data.layers;
  out += `<text x="${padL}" y="${H - 4}" fill="#8b949e" font-size="9">L${L[0]}</text>`;
  out += `<text x="${W - 6}" y="${H - 4}" fill="#8b949e" font-size="9" text-anchor="end">out</text>`;
  svg.innerHTML = out;
}

// ---- tooltip -------------------------------------------------------------
const tip = $("tooltip");
$("grid").addEventListener("mousemove", (e) => {
  const cell = e.target.closest(".cell");
  if (!cell || !data) { tip.hidden = true; return; }
  const li = +cell.dataset.l, t = +cell.dataset.t;
  const row = data.grid[li];
  const rows = row.top_ids[t]
    .map((id, i) =>
      `<tr><td>${esc(tokStr(data.vocab[id]))}</td><td class="p">${(row.top_probs[t][i] * 100).toFixed(1)}%</td></tr>`)
    .join("");
  tip.innerHTML =
    `<table>${rows}</table>` +
    `<div class="meta">${row.is_output ? "output" : "L" + row.layer} · pos ${t} · ` +
    `H=${row.entropy[t]} · kurt=${row.kurtosis[t]}<br>click to pin “${esc(tokStr(data.vocab[row.top_ids[t][0]]))}”</div>`;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(e.clientX + 14, innerWidth - w - 8) + "px";
  tip.style.top = Math.min(e.clientY + 14, innerHeight - h - 8) + "px";
});
$("grid").addEventListener("mouseleave", () => (tip.hidden = true));

// ---- interactions ----------------------------------------------------------
$("grid").addEventListener("click", (e) => {
  const cell = e.target.closest(".cell");
  if (!cell || !data) return;
  const row = data.grid[+cell.dataset.l];
  const id = row.top_ids[+cell.dataset.t][0];
  if (!pinned.some((p) => p.id === id)) {
    if (pinned.length >= 16) pinned.shift();
    pinned.push({ id, str: data.vocab[id] ?? "?" });
  }
  rankViewId = id;
  renderPinned();
  read(); // refetch so pinned_ranks include the new token
});

$("pinned-chips").addEventListener("click", (e) => {
  const x = e.target.dataset.x, chip = e.target.closest(".pin-chip");
  if (x !== undefined) {
    pinned = pinned.filter((p) => p.id !== +x);
    if (rankViewId === +x) rankViewId = null;
    renderPinned(); read();
  } else if (chip) {
    rankViewId = +chip.dataset.id;
    renderPinned(); renderGrid();
  }
});
$("view-top1").addEventListener("click", () => { rankViewId = null; renderPinned(); renderGrid(); });

$("mode-toggle").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  mode = btn.dataset.mode;
  for (const b of $("mode-toggle").children) b.classList.toggle("active", b === btn);
  read();
});
$("examples").addEventListener("change", (e) => {
  if (e.target.value === "") return;
  loadExample(+e.target.value);
});
$("read").addEventListener("click", read);
$("prompt").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") read();
});
// Editing the prompt by hand means it's no longer the selected example.
// In live mode, re-read as you type (debounced) — watch the thoughts move.
let liveTimer = null;
$("prompt").addEventListener("input", () => {
  if (playing) return; // typewriter playback, not the user
  playToken++; // typing cancels any running playback
  $("examples").value = "";
  if (STATIC || !$("live").checked) return;
  clearTimeout(liveTimer);
  liveTimer = setTimeout(() => read({ live: true }), 650);
});

init().catch((e) => setStatus(e.message, true));
