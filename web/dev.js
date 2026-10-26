/* Charts of the analysis results, for development only.
 *
 * Everything comes from /api/dev: the results files as the scripts last wrote
 * them, the answer record, and the prior's defaults.  Nothing is typed here.
 */

const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";

let DATA = null;
const state = { window: null, axis: "noncurated", span: "regime", split: false,
                replay: null };

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

// a run's date, without the time the results file also keeps
function day(at) {
  return at.slice(0, 10);
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
  return `<code>analysis/${script}.py</code>, measured ${day(run.at)} on answers through ${run.through}.`;
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
    // `line: false` draws the markers alone; `dash` marks a guide or projection
    if (s.line !== false) {
      el("polyline", { points: pts.map((p) => `${X(p.x)},${Y(p.y)}`).join(" "),
                       fill: "none", stroke: s.color, "stroke-width": 2,
                       "stroke-linejoin": "round",
                       ...(s.dash ? { "stroke-dasharray": "5 4" } : {}) }, svg);
    }
    if (s.markers) {
      for (const p of pts) {
        el("circle", { cx: X(p.x), cy: Y(p.y), r: 4, fill: s.color,
                       stroke: "var(--surface)", "stroke-width": 2 }, svg);
      }
    }
  }

  // hover: a crosshair at the nearest x, and the tooltip the caller builds
  // a series marked `hover: false` is a guide, and the hover skips it
  const uniq = [...new Set(opts.series.filter((s) => s.hover !== false)
    .flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b);
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

// ---------------------------------------------------------- repeat sources

function drawRepeats() {
  const first = new Map(), events = [];
  for (const [n, date, word] of DATA.record.rows) {
    if (first.has(word)) {
      const [from, fromDate] = first.get(word);
      events.push({ x: n, y: from, date, fromDate, word });
    } else {
      first.set(word, [n, date]);
    }
  }
  if (!events.length) return;
  const lag = events.reduce((sum, e) => sum + e.x - e.y, 0) / events.length;
  $("repeats-note").textContent = `Each of the ${events.length} repeats, by the `
    + "puzzle it fell on and the puzzle that first used the word. A fixed replay "
    + "lag puts the points on a diagonal, and the dashed one is the mean lag of "
    + `${(lag / 365).toFixed(2)} years. A source window that stays put puts `
    + "them in a level band.";
  legend($("repeats-legend"), [
    { name: "a repeat", color: "var(--s1)", kind: "dot" },
    { name: "the mean replay lag", color: "var(--muted)", kind: "dash" },
  ]);
  const x0 = events[0].x, x1 = events[events.length - 1].x;
  lineChart($("repeats-chart"), {
    series: [
      { name: "lag", color: "var(--muted)", dash: true, hover: false,
        points: [{ x: x0, y: x0 - lag }, { x: x1, y: x1 - lag }] },
      { name: "repeat", color: "var(--s1)", line: false, markers: true, points: events },
    ],
    xTicks: ticksBetween(x0, x1, 6), xFmt: (v) => `#${v}`,
    xLabel: "the puzzle it fell on", yLabel: "first used at",
    yFmt: (v) => `#${Math.round(v)}`,
    tip: (x) => {
      const e = events.find((ev) => ev.x === x);
      return `<div><b>${e.word}</b></div><div>#${e.x}, ${e.date}</div>`
        + `<div>first #${e.y}, ${e.fromDate}</div>`
        + `<div>${((e.x - e.y) / 365).toFixed(2)} years later</div>`;
    },
  });
}

// ------------------------------------------------------------ curated pool

function drawPool() {
  const { rows, curated_total: total, regime_start: start } = DATA.record;
  let used = 0;
  const points = rows.map(([n, date, , kind]) => {
    if (kind === "fresh") used += 1;
    return { x: n, y: total - used, date };
  });
  const last = points[points.length - 1];
  const rate = values("regime").fresh_per_year;
  const series = [{ name: "unused", color: "var(--s1)", points }];
  let ends = "";
  if (rate) {
    const days = last.y / rate * 365;
    const end = new Date(`${last.date}T00:00:00Z`);
    end.setUTCDate(end.getUTCDate() + Math.round(days));
    ends = end.toISOString().slice(0, 10);
    series.push({ name: "projected", color: "var(--s1)", dash: true, hover: false,
                  points: [{ x: last.x, y: last.y }, { x: last.x + days, y: 0 }] });
  }
  $("pool-note").innerHTML = `${last.y.toLocaleString()} of the ${total.toLocaleString()} `
    + `curated words are still unused after #${last.x}. `
    + (rate ? `At the last two years' ${rate.toFixed(0)} fresh curated answers a year, `
      + `from <code>analysis/regime.py</code>, the list runs out around ${ends}.` : "");
  legend($("pool-legend"), [
    { name: "curated words never yet an answer", color: "var(--s1)" },
    ...(rate ? [{ name: "projected at the last two years' rate", color: "var(--s1)", kind: "dash" }] : []),
  ]);
  lineChart($("pool-chart"), {
    series, yFloor: 0, height: 240,
    refs: [{ x: rows[start][0], label: "regime begins" }],
    xTicks: ticksBetween(1, series[series.length - 1].points.at(-1).x, 6),
    xFmt: (v) => `#${Math.round(v)}`, xLabel: "puzzle",
    yFmt: (v) => Math.round(v).toLocaleString(),
    tip: (x) => {
      const p = points[x - 1];
      return `<div><b>#${p.x}</b> ${p.date}</div>`
        + tipRow("var(--s1)", "unused", p.y.toLocaleString());
    },
  });
}

// ------------------------------------------------------------- replay arms

// each table, and the baseline `replay.py` pairs its rows against (its BASE)
const ARM_TABLES = [
  ["priors.arm_table", "the four priors", "both priors on"],
  ["shapes.arm_table", "the four shapes", "uniform shapes"],
  ["arm_table", "the curation shape", "the shipped wordlebot shape"],
];
const ARM_COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];

function parsePm(text) {
  const m = text.replace(/\*\*/g, "").replace(/−/g, "-")
    .match(/([-+]?\d*\.?\d+)\s*±\s*(\d*\.?\d+)/);
  return m ? [parseFloat(m[1]), parseFloat(m[2])] : null;
}

function parseTable(text) {
  const lines = text.trim().split("\n").filter((l) => !/^\|\s*-/.test(l));
  const cells = (line) => line.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const [columns, ...rows] = lines.map(cells);
  return { columns, rows };
}

function drawReplay() {
  const v = values("replay");
  const tables = ARM_TABLES.filter(([key]) => v[key]);
  const controls = $("replay-controls");
  controls.innerHTML = "";
  if (!tables.length) return missing($("replay-chart"), "replay");
  if (!tables.some(([key]) => key === state.replay)) state.replay = tables[0][0];
  segmented(controls, tables.map(([key, name]) => [key, name]), state.replay,
            (key) => { state.replay = key; drawReplay(); });
  const [key, , base] = tables.find(([k]) => k === state.replay);
  $("replay-note").innerHTML = `Guesses per game against ${base}, on the same real `
    + "answers, ±1 standard error. Negative means the row needs fewer guesses. "
    + "Blended weights the three kinds of answer at the regime's mixture. "
    + provenance("replay");
  const { columns, rows } = parseTable(v[key]);
  const series = columns.slice(1).map((name, i) => ({
    name, color: ARM_COLORS[i], cells: rows.map((r) => parsePm(r[i + 1])),
  }));
  legend($("replay-legend"), series);
  forestChart($("replay-chart"), rows.map((r) => r[0].replace(/[`*]/g, "")), series);
}

// one band per row, one error bar per series inside it, on a shared x axis
function forestChart(node, labels, series) {
  node.innerHTML = "";
  const W = 900, band = 16 * series.length + 16;
  const M = { top: 8, right: 18, bottom: 34, left: 210 };
  const H = M.top + band * labels.length + M.bottom;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, node);
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  node.appendChild(tip);

  const ends = series.flatMap((s) => s.cells.flatMap((c) => (c ? [c[0] - c[1], c[0] + c[1]] : [])));
  ends.push(0);
  const pad = (Math.max(...ends) - Math.min(...ends)) * 0.05;
  const lo = Math.min(...ends) - pad, hi = Math.max(...ends) + pad;
  const X = (v) => M.left + (v - lo) / (hi - lo) * (W - M.left - M.right);
  const bottom = H - M.bottom;

  const step = niceStep((hi - lo) / 6);
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    el("line", { x1: X(v), x2: X(v), y1: M.top, y2: bottom,
                 stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: X(v), y: bottom + 16, "text-anchor": "middle" }, svg)
      .textContent = signed(Math.abs(v) < step / 1e6 ? 0 : v, 2);
  }
  el("line", { x1: X(0), x2: X(0), y1: M.top, y2: bottom,
               stroke: "var(--axis)", "stroke-width": 1.5 }, svg);
  el("text", { x: (M.left + W - M.right) / 2, y: H - 2, "text-anchor": "middle",
               class: "label" }, svg).textContent = "guesses per game against the baseline";

  labels.forEach((label, r) => {
    const y0 = M.top + r * band;
    if (r > 0) {
      el("line", { x1: 8, x2: W - M.right, y1: y0, y2: y0,
                   stroke: "var(--grid)", "stroke-width": 1 }, svg);
    }
    el("text", { x: M.left - 12, y: y0 + band / 2 + 4, "text-anchor": "end",
                 class: "label" }, svg).textContent = label;
    series.forEach((s, i) => {
      const c = s.cells[r];
      if (!c) return;
      const y = y0 + 8 + 16 * i + 8;
      el("line", { x1: X(c[0] - c[1]), x2: X(c[0] + c[1]), y1: y, y2: y,
                   stroke: s.color, "stroke-width": 2, "stroke-linecap": "round" }, svg);
      el("circle", { cx: X(c[0]), cy: y, r: 4.5, fill: s.color,
                     stroke: "var(--surface)", "stroke-width": 2 }, svg);
    });
    const hit = el("rect", { x: 0, y: y0, width: W, height: band, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (event) => {
      const box = svg.getBoundingClientRect();
      tip.innerHTML = `<div><b>${label}</b></div>` + series.map((s) =>
        tipRow(s.color, s.name, pm(s.cells[r]))).join("");
      tip.hidden = false;
      const left = event.clientX - box.left;
      const flip = left > box.width * 0.6;
      tip.style.left = flip ? "" : `${left + 14}px`;
      tip.style.right = flip ? `${box.width - left + 14}px` : "";
      tip.style.top = `${(y0 + band) * box.height / H}px`;
    });
    hit.addEventListener("pointerleave", () => { tip.hidden = true; });
  });
}

// -------------------------------------------------------------------- fits

const FIT_COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];

function drawFits() {
  const fits = DATA.fits.map((fit, i) => ({ ...fit, color: FIT_COLORS[i] }));
  if (!fits.length) return;
  const forms = [...new Set(fits.map((f) => f.form))];
  $("fits-note").innerHTML = "<code>f(n)</code> estimates the guesses to finish from "
    + "a bucket of effective size n, 2 to the power of its entropy. Buckets of one "
    + "or two candidates are valued exactly, so the curve's left end is never "
    + `consulted. Every prior is fitted on its own, in ${forms.join(" and ")}; `
    + fits.map((f) => `${f.label} on ${f.fitted_at}`).join(", ") + ".";
  legend($("fits-legend"), fits.map((f) => ({ name: f.label, color: f.color })));
  const series = fits.map((f) => ({
    name: f.label, color: f.color,
    points: f.n.map((n, i) => ({ x: n, y: f.f[i] })),
  }));
  const top = fits[0].n[fits[0].n.length - 1];
  lineChart($("fits-chart"), {
    series, log: true,
    xTicks: [1, 10, 100, 1000, 10000, top], xFmt: (v) => v.toLocaleString(),
    xLabel: "effective size n, log scale", yLabel: "guesses to finish",
    yFmt: (v) => v.toFixed(1),
    tip: (x) => `<div><b>n = ${x.toLocaleString()}</b></div>` + series.map((s) =>
      tipRow(s.color, s.name, s.points.find((p) => p.x === x).y.toFixed(3))).join(""),
  });

  // the seed and round 1 sit far above the rest, so the chart starts at round 2
  const gaps = fits.map((f) => {
    const best = Math.min(...f.rounds.map((r) => r[1]));
    return f.rounds.map(([label, mean], i) => ({ x: i, y: mean - best, mean, label }));
  });
  const early = gaps.flatMap((g) => g.slice(0, 2).map((p) => p.y));
  $("rounds-note").textContent = "Each round plays the same sampled games under the "
    + "policy the last round fitted. The entropy seed and round 1 are left off; they "
    + `sit ${Math.min(...early).toFixed(3)} to ${Math.max(...early).toFixed(3)} above `
    + "each prior's best round.";
  const rounds = fits.map((f, i) => ({
    name: f.label, color: f.color, markers: true, points: gaps[i].slice(2),
  }));
  const most = Math.max(...gaps.map((g) => g.length)) - 1;
  lineChart($("rounds-chart"), {
    series: rounds, yFloor: 0, zero: true, height: 240,
    xTicks: Array.from({ length: most - 1 }, (_, i) => i + 2), xFmt: (v) => `round ${v}`,
    yLabel: "above the best round", yFmt: (v) => `+${v.toFixed(3)}`,
    tip: (x) => `<div><b>round ${x}</b></div>` + rounds.map((s) => {
      const p = s.points.find((q) => q.x === x);
      return p ? tipRow(s.color, s.name, `${p.mean.toFixed(4)} (+${p.y.toFixed(4)})`) : "";
    }).join(""),
  });
}

// ------------------------------------------------------------ head to head

// grouped vertical bars from a zero baseline, negatives below it
function barChart(node, { categories, series, fmt, yLabel, height = 240 }) {
  node.innerHTML = "";
  const W = 900, H = height, M = { top: 12, right: 18, bottom: 30, left: 58 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, node);
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  node.appendChild(tip);
  const all = series.flatMap((s) => s.values).concat(0);
  const pad = (Math.max(...all) - Math.min(...all)) * 0.08;
  const lo = Math.min(0, Math.min(...all) - (Math.min(...all) < 0 ? pad : 0));
  const hi = Math.max(...all) + pad;
  const Y = (v) => H - M.bottom - (v - lo) / (hi - lo) * (H - M.top - M.bottom);
  const step = niceStep((hi - lo) / 5);
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    el("line", { x1: M.left, x2: W - M.right, y1: Y(v), y2: Y(v),
                 stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: M.left - 8, y: Y(v) + 4, "text-anchor": "end" }, svg)
      .textContent = fmt(Math.abs(v) < step / 1e6 ? 0 : v);
  }
  el("line", { x1: M.left, x2: W - M.right, y1: Y(0), y2: Y(0),
               stroke: "var(--axis)", "stroke-width": 1.5 }, svg);
  if (yLabel) {
    const mid = (M.top + H - M.bottom) / 2;
    el("text", { x: 12, y: mid, "text-anchor": "middle", class: "label",
                 transform: `rotate(-90 12 ${mid})` }, svg).textContent = yLabel;
  }
  const slot = (W - M.left - M.right) / categories.length;
  const bar = Math.min(44, (slot * 0.7) / series.length);
  categories.forEach((name, c) => {
    const x0 = M.left + c * slot + (slot - bar * series.length - 2 * (series.length - 1)) / 2;
    el("text", { x: M.left + (c + 0.5) * slot, y: H - M.bottom + 16,
                 "text-anchor": "middle", class: "label" }, svg).textContent = name;
    series.forEach((s, i) => {
      const v = s.values[c];
      const top = Math.min(Y(v), Y(0)), h = Math.max(1, Math.abs(Y(v) - Y(0)));
      el("rect", { x: x0 + i * (bar + 2), y: top, width: bar, height: h, rx: 3,
                   fill: s.color }, svg);
    });
    const hit = el("rect", { x: M.left + c * slot, y: M.top, width: slot,
                             height: H - M.top - M.bottom, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", () => {
      const box = svg.getBoundingClientRect();
      tip.innerHTML = `<div><b>${name}</b></div>` + series.map((s) =>
        tipRow(s.color, s.name, fmt(s.values[c], true))).join("");
      tip.hidden = false;
      const left = (M.left + (c + 1) * slot) * box.width / W;
      const flip = left > box.width * 0.6;
      tip.style.left = flip ? "" : `${left + 4}px`;
      tip.style.right = flip ? `${box.width - (M.left + c * slot) * box.width / W + 4}px` : "";
      tip.style.top = `${M.top * box.height / H}px`;
    });
    hit.addEventListener("pointerleave", () => { tip.hidden = true; });
  });
}

// a table cell as a number, or null; `%` and the documents' minus understood
function parseNum(cell) {
  const text = cell.replace(/\*\*/g, "").replace(/−/g, "-").replace(/,/g, "");
  const m = text.match(/^\s*([-+]?\d*\.?\d+)\s*(%|s)?\s*$/);
  return m ? parseFloat(m[1]) : null;
}

function drawHeadToHead() {
  const v = values("head_to_head");
  const options = [["table", "against IG + 2·p_win"], ["baseline.table", "against greedy entropy"]]
    .filter(([key]) => v[key]);
  if (!options.length) return missing($("h2h-chart"), "head_to_head");
  if (!options.some(([key]) => key === state.h2h)) state.h2h = options[0][0];
  const controls = $("h2h-controls");
  controls.innerHTML = "";
  segmented(controls, options, state.h2h, (key) => { state.h2h = key; drawHeadToHead(); });
  $("h2h-note").innerHTML = "Every word as the answer, weighted by the default prior: "
    + "the share of games each solver finishes in each number of guesses. "
    + provenance("head_to_head");
  const { columns, rows } = parseTable(v[state.h2h]);
  const names = columns.slice(1).map((c) => c.replace(/`/g, ""));
  const series = names.map((name, i) => ({
    name, color: ARM_COLORS[i], values: rows.map((r) => parseNum(r[i + 1])),
  }));
  legend($("h2h-legend"), series);
  const categories = rows.map((r) => r[0]);
  const share = (x, exact) => `${x.toFixed(exact ? 4 : 0)}%`;
  barChart($("h2h-chart"), { categories, series, fmt: share, yLabel: "share of games" });
  barChart($("h2h-diff"), {
    categories, height: 200, yLabel: "points",
    series: [{ name: `${names[0]} − ${names[1]}`, color: "var(--s1)",
               values: rows.map((_, c) => series[0].values[c] - series[1].values[c]) }],
    fmt: (x, exact) => `${x < 0 ? "−" : "+"}${Math.abs(x).toFixed(exact ? 4 : 1)}`,
  });
}

// ----------------------------------------------------------- column panels

// one dot plot per numeric column of a table, each on its own axis; the
// default row, which the documents bold, is ringed
function columnPanels(node, text) {
  node.innerHTML = "";
  const { columns, rows } = parseTable(text);
  const labels = rows.map((r) => r[0].replace(/[`*]/g, ""));
  const bold = rows.map((r) => r.some((c) => c.includes("**")));
  // every panel goes into the grid before any is measured
  const panels = [];
  columns.slice(1).forEach((column, i) => {
    const vals = rows.map((r) => parseNum(r[i + 1]));
    if (vals.every((x) => x == null)) return;
    const box = document.createElement("div");
    box.innerHTML = `<h3>${inline(column)}</h3>`;
    const chart = document.createElement("div");
    chart.className = "chart";
    box.appendChild(chart);
    node.appendChild(box);
    panels.push([chart, vals, i]);
  });
  for (const [chart, vals, i] of panels) {
    // drawn at the panel's own width, so the text stays at its set size
    const W = Math.round(chart.getBoundingClientRect().width) || 420;
    const row = 26, M = { top: 6, right: 70, bottom: 8, left: 170 };
    const H = M.top + row * rows.length + M.bottom;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, chart);
    const known = vals.filter((x) => x != null);
    let lo = Math.min(...known), hi = Math.max(...known);
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.06;
    const X = (x) => M.left + (x - lo + pad) / (hi - lo + 2 * pad) * (W - M.left - M.right);
    rows.forEach((r, k) => {
      const y = M.top + row * k + row / 2;
      el("line", { x1: M.left, x2: W - M.right, y1: y, y2: y,
                   stroke: "var(--grid)", "stroke-width": 1 }, svg);
      const name = el("text", { x: M.left - 8, y: y + 4, "text-anchor": "end",
                                class: "label" }, svg);
      name.textContent = labels[k].length > 26 ? `${labels[k].slice(0, 25)}…` : labels[k];
      if (bold[k]) name.setAttribute("font-weight", "600");
      el("title", {}, name).textContent = labels[k];
      const x = vals[k];
      if (x == null) return;
      el("circle", { cx: X(x), cy: y, r: bold[k] ? 6 : 4.5,
                     fill: bold[k] ? "var(--s2)" : "var(--s1)",
                     stroke: "var(--surface)", "stroke-width": 2 }, svg);
      el("text", { x: W - M.right + 8, y: y + 4, class: "label" }, svg)
        .textContent = r[i + 1].replace(/\*\*/g, "");
    });
  }
}

function tablePanels(script, options, ids, note) {
  const v = values(script);
  const shown = options.filter(([key]) => v[key]);
  const controls = $(ids.controls);
  controls.innerHTML = "";
  $(ids.note).innerHTML = `${note} ${provenance(script)}`;
  if (!shown.length) return missing($(ids.panels), script);
  if (!shown.some(([key]) => key === state[script])) state[script] = shown[0][0];
  segmented(controls, shown, state[script], (key) => {
    state[script] = key;
    tablePanels(script, options, ids, note);
  });
  columnPanels($(ids.panels), v[state[script]]);
}

function drawLoss() {
  tablePanels("loss_penalty", [
    ["penalty.table", "the loss penalty"], ["reach.table", "the search's reach"],
    ["search.table", "how much is searched"], ["policies.table", "the policies"],
  ], { controls: "loss-controls", panels: "loss-panels", note: "loss-note" },
  "Every word as the answer, weighted by the default prior. The default setting "
    + "is the orange dot; each panel has its own axis.");
}

function drawRegret() {
  tablePanels("exact_decisions", [
    ["blended.table", "positions from real games"], ["shipped.table", "under the shipped prior"],
    ["who.table", "who decides each turn"],
  ], { controls: "regret-controls", panels: "regret-panels", note: "regret-note" },
  "Positions the exact search takes over, grouped by candidates left: how often "
    + "the estimator would have picked differently, and what that costs in guesses. "
    + "Each panel has its own axis.");
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
    .map(([argv, run]) => `<code>${argv}</code>: ${day(run.at)}, answers through ${run.through}`)
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
  drawRepeats();
  drawPool();
  drawReplay();
  drawFits();
  drawHeadToHead();
  drawLoss();
  drawRegret();
  drawLag();
  drawTables();
}

boot().catch((error) => {
  $("lede").textContent = `Could not load /api/dev: ${error}`;
});
