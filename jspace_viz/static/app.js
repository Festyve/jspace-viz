// Copyright 2026 Michael Zhang
// SPDX-License-Identifier: Apache-2.0
// Live J-space visualizer frontend. No build step, no dependencies.

const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let data = null;          // last /api/read response
let mode = "jlens";
let pinned = [];          // [{id, str}]
let rankViewId = null;    // token id whose ranks the grid shows, or null

// Render a token for display: make whitespace visible, trim length.
function tokStr(s) {
  if (s === undefined) return "?";
  let out = s.replace(/\n/g, "⏎").replace(/\t/g, "⇥");
  if (out.startsWith(" ")) out = "·" + out.slice(1);
  if (out.trim() === "") out = out.replace(/ /g, "·") || "∅";
  return out.length > 12 ? out.slice(0, 11) + "…" : out;
}

function setStatus(msg, err = false) {
  const el = $("status");
  el.textContent = msg;
  el.className = err ? "err" : "";
}

async function init() {
  const info = await (await fetch("/api/info")).json();
  $("model-chip").textContent = `${info.model_id} · ${info.n_layers}L · d=${info.d_model} · ${info.device}`;
  $("lens-chip").textContent = `lens: ${info.fitted_layers.length} layers, ${info.lens_n_prompts} prompts`;
  for (const ex of info.examples) {
    const opt = document.createElement("option");
    opt.value = ex.prompt;
    opt.textContent = ex.name;
    $("examples").appendChild(opt);
  }
  $("prompt").value = info.examples[0].prompt;
  read();
}

async function read() {
  setStatus("reading…");
  $("read").disabled = true;
  const t0 = performance.now();
  try {
    const res = await fetch("/api/read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: $("prompt").value,
        mode,
        top_k: +$("topk").value,
        chat: $("chat").checked,
        pinned_ids: pinned.map((p) => p.id),
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    data = await res.json();
    renderGrid();
    renderMetrics();
    setStatus(`${data.seq_len} tokens × ${data.layers.length} layers · ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    $("read").disabled = false;
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
  grid.style.gridTemplateColumns = `auto repeat(${seq_len}, max-content)`;
  const parts = ['<div class="hcell corner">layer ╲ pos</div>'];
  for (let t = 0; t < seq_len; t++) {
    parts.push(`<div class="hcell"><span class="idx">${t}</span>${esc(tokStr(vocab[context_ids[t]]))}</div>`);
  }
  for (let li = 0; li < layers.length; li++) {
    const row = rows[li];
    const cls = row.is_output ? " outrow" : "";
    parts.push(`<div class="lcell${cls}">${row.is_output ? "output" : "L" + row.layer}</div>`);
    for (let t = 0; t < seq_len; t++) {
      let text, bg;
      const pi = rankViewId === null ? -1 : pinned.findIndex((p) => p.id === rankViewId);
      if (pi >= 0 && row.pinned_ranks) {
        const r = row.pinned_ranks[t][pi];
        text = r === 0 ? "★0" : String(r);
        bg = rankColor(r);
      } else {
        text = tokStr(vocab[row.top_ids[t][0]]);
        bg = probColor(row.top_probs[t][0]);
      }
      parts.push(`<div class="cell${cls}" data-l="${li}" data-t="${t}" style="background:${bg}">${esc(text)}</div>`);
    }
  }
  grid.innerHTML = parts.join("");
}

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
  if (e.target.value) { $("prompt").value = e.target.value; read(); }
});
$("read").addEventListener("click", read);
$("prompt").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") read();
});

init().catch((e) => setStatus(e.message, true));
