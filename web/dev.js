/* Charts of the analysis results, for development only.
 *
 * Everything comes from /api/dev: the results files as the scripts last wrote
 * them, the answer record, and the prior's defaults.  Nothing is typed here.
 */

const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";

let DATA = null;
const state = { window: null, axis: "noncurated", span: "regime", split: false };

const TYPES = {
  fresh: { name: "fresh curated", color: "var(--s3)" },
  offlist: { name: "off-list", color: "var(--s2)" },
  repeat: { name: "repeat", color: "var(--s1)" },
};

// ------------------------------------------------------------------ helpers

function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (parent) parent.appendChild(node);
  return node;
}

function signed(value, digits = 3) {
  if (value == null) return "–";
  const text = Math.abs(value).toFixed(digits);
  return (value < 0 ? "−" : "+") + text;
}

function pm(cell, digits = 3) {
  if (!cell || cell[0] == null) return "–";
  return `${signed(cell[0], digits)} ± ${cell[1] == null ? "–" : cell[1].toFixed(digits)}`;
}

function pct(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function values(script) {
  const file = DATA.results[script];
  return file ? file.values : {};
}

// the plain run of a script if it has one, else its latest
function runOf(script) {
  const file = DATA.results[script];
  if (!file) return null;
  const runs = Object.entries(file.runs);
  const plain = runs.find(([argv]) => argv === `${script}.py`);
  if (plain) return plain[1];
  return runs.map(([, run]) => run).sort((a, b) => (a.at < b.at ? 1 : -1))[0];
}

function provenance(script) {
  const run = runOf(script);
  if (!run) return `<code>analysis/${script}.py</code> has not recorded anything.`;
  return `<code>analysis/${script}.py</code>, measured ${run.at} on answers through ${run.through}.`;
}

function missing(node, script) {
  node.innerHTML = `<div class="missing">Not recorded yet: rerun <code>analysis/${script}.py</code>.</div>`;
}

function segmented(parent, options, current, onPick) {
  const group = document.createElement("div");
  group.className = "seg";
  for (const [value, label] of options) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.setAttribute("aria-pressed", String(value === current));
    button.addEventListener("click", () => onPick(value));
    group.appendChild(button);
  }
  parent.appendChild(group);
}

function legend(node, items) {
  node.innerHTML = items.map(({ name, color, kind = "" }) =>
    `<span><i class="${kind}" style="background:${color};border-color:${color}"></i>${name}</span>`
  ).join("");
}

// ------------------------------------------------------------------- charts

// A line chart with optional error bars or bands, one y axis, and a hover
// layer that snaps to the nearest x.
function lineChart(node, opts) {
  node.innerHTML = "";
  const W = opts.width || 900, H = opts.height || 280;
  const M = { top: 14, right: 18, bottom: 34, left: 58 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, node);
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  node.appendChild(tip);

  const xs = opts.series.flatMap((s) => s.points.map((p) => p.x));
  const ys = opts.series.flatMap((s) => s.points.flatMap((p) =>
    [p.y, p.lo, p.hi].filter((v) => v != null)));
  for (const ref of opts.refs || []) if (ref.y != null) ys.push(ref.y);
  if (opts.zero) ys.push(0);
  const xMin = opts.xMin ?? Math.min(...xs), xMax = opts.xMax ?? Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  const padY = (yMax - yMin || 1) * 0.08;
  yMin -= padY;
  yMax += padY;
  if (opts.yFloor != null) yMin = Math.max(yMin, opts.yFloor);

  const tx = opts.log ? Math.log10 : (v) => v;
  const X = (v) => M.left + (tx(v) - tx(xMin)) / ((tx(xMax) - tx(xMin)) || 1)
    * (W - M.left - M.right);
  const Y = (v) => H - M.bottom - (v - yMin) / (yMax - yMin) * (H - M.top - M.bottom);

  // grid and y ticks
  const step = niceStep((yMax - yMin) / 5);
  for (let v = Math.ceil(yMin / step) * step; v <= yMax; v += step) {
    el("line", { x1: M.left, x2: W - M.right, y1: Y(v), y2: Y(v),
                 stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: M.left - 8, y: Y(v) + 4, "text-anchor": "end" }, svg)
      .textContent = opts.yFmt(Math.abs(v) < step / 1e6 ? 0 : v);
  }
  if (opts.zero && yMin < 0 && yMax > 0) {
    el("line", { x1: M.left, x2: W - M.right, y1: Y(0), y2: Y(0),
                 stroke: "var(--axis)", "stroke-width": 1.5 }, svg);
  }
  el("line", { x1: M.left, x2: W - M.right, y1: H - M.bottom, y2: H - M.bottom,
               stroke: "var(--axis)", "stroke-width": 1 }, svg);
  // drop an x tick that would crowd the one before it
  let lastTick = -Infinity;
  for (const v of opts.xTicks || []) {
    if (X(v) - lastTick < 34) continue;
    lastTick = X(v);
    el("text", { x: X(v), y: H - M.bottom + 16, "text-anchor": "middle" }, svg)
      .textContent = opts.xFmt(v);
  }
  if (opts.xLabel) {
    el("text", { x: (M.left + W - M.right) / 2, y: H - 2, "text-anchor": "middle",
                 class: "label" }, svg).textContent = opts.xLabel;
  }
  if (opts.yLabel) {
    el("text", { x: 12, y: (M.top + H - M.bottom) / 2, "text-anchor": "middle",
                 class: "label",
                 transform: `rotate(-90 12 ${(M.top + H - M.bottom) / 2})` }, svg)
      .textContent = opts.yLabel;
  }

  // reference lines
  for (const ref of opts.refs || []) {
    const dash = { stroke: ref.color || "var(--muted)", "stroke-width": 1.5,
                   "stroke-dasharray": "5 4" };
    if (ref.y != null) {
      el("line", { x1: M.left, x2: W - M.right, y1: Y(ref.y), y2: Y(ref.y), ...dash }, svg);
    } else {
      el("line", { x1: X(ref.x), x2: X(ref.x), y1: M.top, y2: H - M.bottom, ...dash }, svg);
      if (ref.label) {
        el("text", { x: X(ref.x) + 5, y: M.top + 10, class: "label" }, svg)
          .textContent = ref.label;
      }
    }
  }

  // bands, then error bars, then lines and markers
  for (const s of opts.series) {
    const pts = s.points.filter((p) => p.y != null);
    if (s.band) {
      const upper = pts.map((p) => `${X(p.x)},${Y(p.hi)}`);
      const lower = pts.slice().reverse().map((p) => `${X(p.x)},${Y(p.lo)}`);
      el("polygon", { points: upper.concat(lower).join(" "), fill: s.color,
                      "fill-opacity": 0.14 }, svg);
    }
    if (s.bars) {
      for (const p of pts) {
        if (p.lo == null) continue;
        el("line", { x1: X(p.x), x2: X(p.x), y1: Y(p.lo), y2: Y(p.hi),
                     stroke: s.color, "stroke-width": 1.5, "stroke-opacity": 0.55 }, svg);
      }
    }
  }
  for (const s of opts.series) {
    const pts = s.points.filter((p) => p.y != null);
    el("polyline", { points: pts.map((p) => `${X(p.x)},${Y(p.y)}`).join(" "),
                     fill: "none", stroke: s.color, "stroke-width": 2,
                     "stroke-linejoin": "round" }, svg);
    if (s.markers) {
      for (const p of pts) {
        el("circle", { cx: X(p.x), cy: Y(p.y), r: 4, fill: s.color,
                       stroke: "var(--surface)", "stroke-width": 2 }, svg);
      }
    }
  }

  // hover: a crosshair at the nearest x, and the tooltip the caller builds
  const uniq = [...new Set(xs)].sort((a, b) => a - b);
  const cross = el("line", { y1: M.top, y2: H - M.bottom, stroke: "var(--muted)",
                             "stroke-width": 1, visibility: "hidden" }, svg);
  const dots = opts.series.map((s) => el("circle", {
    r: 5, fill: s.color, stroke: "var(--surface)", "stroke-width": 2,
    visibility: "hidden" }, svg));
  const hit = el("rect", { x: M.left, y: M.top, width: W - M.left - M.right,
                           height: H - M.top - M.bottom, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const px = (event.clientX - box.left) * W / box.width;
    let best = uniq[0];
    for (const v of uniq) if (Math.abs(X(v) - px) < Math.abs(X(best) - px)) best = v;
    cross.setAttribute("x1", X(best));
    cross.setAttribute("x2", X(best));
    cross.setAttribute("visibility", "visible");
    opts.series.forEach((s, i) => {
      const p = s.points.find((q) => q.x === best);
      if (p && p.y != null) {
        dots[i].setAttribute("cx", X(p.x));
        dots[i].setAttribute("cy", Y(p.y));
        dots[i].setAttribute("visibility", "visible");
      } else {
        dots[i].setAttribute("visibility", "hidden");
      }
    });
    tip.innerHTML = opts.tip(best);
    tip.hidden = false;
    const left = X(best) * box.width / W;
    const flip = left > box.width * 0.6;
    tip.style.left = flip ? "" : `${left + 12}px`;
    tip.style.right = flip ? `${box.width - left + 12}px` : "";
    tip.style.top = `${M.top * box.height / H}px`;
  });
  hit.addEventListener("pointerleave", () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
  });
}

function niceStep(raw) {
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  return (unit < 1.5 ? 1 : unit < 3.5 ? 2 : unit < 7.5 ? 5 : 10) * power;
}

function tipRow(color, label, value) {
  return `<div><i style="background:${color}"></i>${label} <b>${value}</b></div>`;
}

// ------------------------------------------------------------- answer rates

function drawRates() {
  const { rows, regime_start: start } = DATA.record;
  const { repeat_prob: repeatProb, noncurated_prob: offProb } = DATA.defaults;
  const w = state.window;
  const controls = $("rates-controls");
  controls.innerHTML = "";
  segmented(controls, [25, 50, DATA.defaults.recent].map((n) => [n, `last ${n}`]),
            w, (n) => { state.window = n; drawRates(); });

  const repeat = [], offlist = [], meta = new Map();
  for (let end = start + w - 1; end < rows.length; end++) {
    const slice = rows.slice(end - w + 1, end + 1);
    const reps = slice.filter((r) => r[3] === "repeat").length;
    const offs = slice.filter((r) => r[3] === "offlist").length;
    const n = rows[end][0];
    const r = reps / w, o = offs / (w - reps);
    const rse = Math.sqrt(r * (1 - r) / w), ose = Math.sqrt(o * (1 - o) / (w - reps));
    repeat.push({ x: n, y: r, lo: r - rse, hi: r + rse });
    offlist.push({ x: n, y: o, lo: o - ose, hi: o + ose });
    meta.set(n, { row: rows[end], reps, offs, r, o, rse, ose });
  }
  legend($("rates-legend"), [
    { name: "off-list, given not a repeat", color: "var(--s2)" },
    { name: "repeat", color: "var(--s1)" },
    { name: `defaults ${offProb} and ${repeatProb}`, color: "var(--muted)", kind: "dash" },
  ]);
  // the x axis spans the whole regime, so the strip below lines up with it
  const first = rows[start][0], last = rows[rows.length - 1][0];
  lineChart($("rates-chart"), {
    xMin: first, xMax: last,
    series: [
      { name: "off-list", color: "var(--s2)", points: offlist, band: true },
      { name: "repeat", color: "var(--s1)", points: repeat, band: true },
    ],
    refs: [{ y: offProb, color: "var(--s2)" }, { y: repeatProb, color: "var(--s1)" }],
    yFloor: 0,
    yFmt: (v) => pct(v, 0),
    xTicks: ticksBetween(first, last, 6),
    xFmt: (v) => `#${v}`,
    xLabel: `the puzzle the ${w}-puzzle window ends at`,
    tip: (x) => {
      const m = meta.get(x);
      return `<div><b>#${m.row[0]}</b> ${m.row[1]}, <b>${m.row[2]}</b></div>`
        + tipRow("var(--s2)", "off-list", `${pct(m.o)} ± ${pct(m.ose)} (${m.offs} of ${w - m.reps})`)
        + tipRow("var(--s1)", "repeat", `${pct(m.r)} ± ${pct(m.rse)} (${m.reps} of ${w})`);
    },
  });
  drawStrip(rows.slice(start));
}

function ticksBetween(a, b, count) {
  const step = niceStep((b - a) / count);
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b; v += step) out.push(v);
  return out;
}

// one tick per regime puzzle, colored by what kind of answer it was
function drawStrip(rows) {
  const node = $("rates-strip");
  node.innerHTML = "";
  const W = 900, H = 46, left = 58, right = 18;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, node);
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  node.appendChild(tip);
  const span = (W - left - right) / (rows.length - 1);
  rows.forEach((row, i) => {
    const kind = TYPES[row[3]];
    const fresh = row[3] === "fresh";
    const rect = el("rect", { x: left + (i - 0.5) * span, y: fresh ? 14 : 6, rx: 1,
                              width: Math.max(1, span - 1), height: fresh ? 8 : 16,
                              fill: fresh ? "var(--axis)" : kind.color }, svg);
    rect.addEventListener("pointerenter", () => {
      const box = svg.getBoundingClientRect();
      tip.innerHTML = `<b>#${row[0]}</b> ${row[1]}, <b>${row[2]}</b>, ${kind.name}`;
      tip.hidden = false;
      const x = (left + i * span) * box.width / W;
      tip.style.left = x > box.width * 0.6 ? "" : `${x + 8}px`;
      tip.style.right = x > box.width * 0.6 ? `${box.width - x + 8}px` : "";
      tip.style.top = "26px";
    });
  });
  svg.addEventListener("pointerleave", () => { tip.hidden = true; });
  el("text", { x: left - 8, y: 30, "text-anchor": "end" }, svg).textContent =
    `#${rows[0][0]}`;
  const counts = Object.fromEntries(Object.keys(TYPES).map((k) =>
    [k, rows.filter((r) => r[3] === k).length]));
  const key = document.createElement("div");
  key.className = "legend";
  legend(key, [
    { name: `${counts.offlist} off-list`, color: "var(--s2)", kind: "dot" },
    { name: `${counts.repeat} repeats`, color: "var(--s1)", kind: "dot" },
    { name: `${counts.fresh} fresh curated, every regime answer from #${rows[0][0]}`,
      color: "var(--axis)", kind: "dot" },
  ]);
  node.appendChild(key);
}

// -------------------------------------------------------------------- basin

function drawBasin() {
  const v = values("prior_basin");
  $("basin-note").innerHTML = "Extra guesses per game against the shipped setting, "
    + "with <code>f</code> held at the shipped fit, ±1 standard error. "
    + "Lower is better. " + provenance("prior_basin");
  const controls = $("basin-controls");
  controls.innerHTML = "";
  const redraw = (key, value) => { state[key] = value; drawBasin(); };
  segmented(controls, [["noncurated", "off-list probability"], ["repeat", "repeat probability"]],
            state.axis, (x) => redraw("axis", x));
  segmented(controls, [["regime", "whole regime"],
                       ["recent", `last ${DATA.defaults.recent}`]],
            state.span, (x) => redraw("span", x));
  segmented(controls, [[false, "all answers"], [true, "by answer type"]],
            state.split, (x) => redraw("split", x));

  const sweep = v[`series.${state.axis}`];
  if (!sweep) {
    $("basin-legend").innerHTML = "";
    missing($("basin-chart"), "prior_basin");
  } else {
    const data = sweep[state.span];
    const xs = data.points;
    const shown = state.split
      ? Object.entries(TYPES).map(([k, t]) => ({ ...t, key: k }))
      : [{ key: "all", name: "all answers", color: "var(--s1)" }];
    const series = shown.map((s) => ({
      name: s.name, color: s.color, markers: true, bars: !state.split,
      points: xs.map((x, i) => {
        const [m, se] = data[s.key][i];
        return { x, y: m, lo: m == null || se == null ? null : m - se,
                 hi: m == null || se == null ? null : m + se, cell: data[s.key][i] };
      }),
    }));
    legend($("basin-legend"), state.split ? shown : []);
    const name = state.axis === "noncurated" ? "noncurated_prob" : "repeat_prob";
    lineChart($("basin-chart"), {
      series, log: true, zero: true,
      refs: [{ x: sweep.base, label: `shipped ${sweep.base}` }],
      xTicks: xs, xFmt: (x) => String(x),
      xLabel: `${name}, log scale`,
      yLabel: "guesses vs shipped",
      yFmt: (y) => signed(y, 2),
      tip: (x) => `<div><b>${name} = ${x}</b></div>` + series.map((s) =>
        tipRow(s.color, s.name, pm(s.points.find((p) => p.x === x).cell))).join(""),
    });
  }

  const rep = v["series.tempered.repeat"], cur = v["series.tempered.curation"];
  if (!rep || !cur) {
    $("temper-legend").innerHTML = "";
    missing($("temper-chart"), "prior_basin");
    return;
  }
  const shapes = [{ name: "repeat shape", color: "var(--s1)", d: rep },
                  { name: "curation shape", color: "var(--s2)", d: cur }];
  legend($("temper-legend"), shapes);
  const series = shapes.map((s) => ({
    name: s.name, color: s.color, markers: true, bars: true,
    points: s.d.points.map((x, i) => {
      const [m, se] = s.d.all[i];
      return { x, y: m, lo: m - se, hi: m + se, cell: s.d.all[i] };
    }),
  }));
  lineChart($("temper-chart"), {
    series, zero: true, height: 220,
    xTicks: rep.points, xFmt: (x) => x.toFixed(2),
    xLabel: "temper: the shape's weights raised to this power",
    yLabel: "guesses vs shipped",
    yFmt: (y) => signed(y, 2),
    tip: (x) => `<div><b>temper ${x.toFixed(2)}</b></div>` + series.map((s) =>
      tipRow(s.color, s.name, pm(s.points.find((p) => p.x === x).cell))).join(""),
  });
}

// --------------------------------------------------------------- refit lag

function drawLag() {
  const v = values("refit_lag");
  $("lag-note").innerHTML = "The shipped prior replayed over the regime with the "
    + "answer record truncated by each lag, paired against a record rebuilt every "
    + "day, ±1 standard error. Positive means the stale record costs guesses. "
    + provenance("refit_lag");
  const lags = v["series.lags"], cycle = v["series.cycle"];
  if (!lags) {
    for (const id of ["lag-chart", "mass-chart", "cycle-chart"]) missing($(id), "refit_lag");
    return;
  }
  const points = lags.lags.map((x, i) => {
    const [m, se] = lags.all[i];
    return { x, y: m, lo: m - se, hi: m + se, i };
  });
  lineChart($("lag-chart"), {
    series: [{ name: "all", color: "var(--s1)", points, markers: true, bars: true }],
    zero: true, width: 440, height: 260,
    xTicks: lags.lags, xFmt: (x) => String(x), xLabel: "days of lag",
    yFmt: (y) => signed(y, 2),
    tip: (x) => {
      const i = lags.lags.indexOf(x);
      const typed = ["repeats", "off-list", "other"].map((k) =>
        `<div>${k} ${pm(lags[k][i], 4)}</div>`).join("");
      return `<div><b>${x} day${x > 1 ? "s" : ""}</b></div>`
        + tipRow("var(--s1)", "all", pm(lags.all[i], 4))
        + `<div>${lags.worse[i] + lags.better[i]} games changed: `
        + `${lags.worse[i]} worse, ${lags.better[i]} better</div>` + typed;
    },
  });
  lineChart($("mass-chart"), {
    series: [{ name: "mass", color: "var(--s3)", markers: true,
               points: lags.lags.map((x, i) => ({ x, y: lags.mass[i] / 100 })) }],
    yFloor: 0, width: 440, height: 260,
    refs: [{ y: v.yardstick / 100, label: "repeat shape flattened" }],
    xTicks: lags.lags, xFmt: (x) => String(x), xLabel: "days of lag",
    yFmt: (y) => pct(y, 1),
    tip: (x) => `<div><b>${x} day${x > 1 ? "s" : ""}</b></div>`
      + tipRow("var(--s3)", "mass moved", pct(lags.mass[lags.lags.indexOf(x)] / 100, 2))
      + tipRow("var(--muted)", "repeat shape flattened", pct(v.yardstick / 100, 2)),
  });
  if (!cycle) return missing($("cycle-chart"), "refit_lag");
  lineChart($("cycle-chart"), {
    series: [{ name: "cycle", color: "var(--s1)", markers: true, bars: true,
               points: cycle.every.map((x, i) => {
                 const [m, se] = cycle.all[i];
                 return { x, y: m, lo: m - se, hi: m + se };
               }) }],
    zero: true, height: 220,
    xTicks: cycle.every, xFmt: (x) => String(x),
    xLabel: "refit every k days, the lag uniform over 0…k−1",
    yFmt: (y) => signed(y, 3),
    tip: (x) => `<div><b>every ${x} days</b></div>`
      + tipRow("var(--s1)", "cost", pm(cycle.all[cycle.every.indexOf(x)], 4)),
  });
}

// ------------------------------------------------------------------- tables

function inline(text) {
  return text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/\*([^*]+)\*/g, "<i>$1</i>");
}

function markdownTable(text) {
  const lines = text.trim().split("\n").filter((l) => !/^\|\s*-/.test(l));
  const cells = (line) => line.replace(/^\||\|$/g, "").split(/(?<!\\)\|/)
    .map((c) => inline(c.trim().replace(/\\\|/g, "|")));
  const [head, ...body] = lines.map(cells);
  return `<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`
    + `<tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}`
    + "</tbody></table>";
}

function tablesOf(script) {
  return Object.entries(values(script))
    .filter(([, value]) => typeof value === "string" && value.startsWith("|"));
}

function drawTables() {
  const pick = $("table-script");
  const scripts = Object.keys(DATA.results).filter((s) => tablesOf(s).length);
  if (!pick.options.length) {
    for (const s of scripts) pick.add(new Option(`${s} (${tablesOf(s).length})`, s));
    pick.addEventListener("change", drawTables);
  }
  const script = pick.value;
  $("table-runs").innerHTML = Object.entries(DATA.results[script].runs)
    .map(([argv, run]) => `<code>${argv}</code>: ${run.at}, answers through ${run.through}`)
    .join("<br>");
  $("table-body").innerHTML = tablesOf(script)
    .map(([key, text]) => `<h3><code>${key}</code></h3>${markdownTable(text)}`).join("");
}

// --------------------------------------------------------------------- boot

async function boot() {
  const reply = await fetch("/api/dev");
  DATA = await reply.json();
  const rows = DATA.record.rows;
  const last = rows[rows.length - 1];
  $("lede").textContent = `${rows.length} answers, through #${last[0]} on ${last[1]}. `
    + "Each chart is the results file its script last wrote; reload after a rerun.";
  state.window = DATA.defaults.recent;
  drawRates();
  drawBasin();
  drawLag();
  drawTables();
}

boot().catch((error) => {
  $("lede").textContent = `Could not load /api/dev: ${error}`;
});
