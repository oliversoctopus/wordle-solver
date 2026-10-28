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

// a note: one short line, and the full explanation collapsed under it; a
// redraw keeps the explanation open if it was
const opened = new Set();
function note(id, short, long) {
  const node = $(id);
  node.innerHTML = `${short}<details><summary>How to read this</summary>${long}</details>`;
  const details = node.querySelector("details");
  details.open = opened.has(id);
  details.addEventListener("toggle", () => {
    if (details.open) opened.add(id); else opened.delete(id);
  });
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
  // drawn at the container's own width, so the text stays at its set size
  const W = opts.width || node.clientWidth || 900, H = opts.height || 280;
  const M = { top: 14, right: opts.right ?? 18, bottom: 34, left: 58 };
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
  // `yLog` scales y by its logarithm; `yMin`/`yMax` fix the range, and a
  // point above `yMax` is drawn at the top edge as a triangle
  const ty = opts.yLog ? Math.log10 : (v) => v;
  const fy = opts.yLog ? (v) => 10 ** v : (v) => v;
  let lo = ty(Math.min(...ys)), hi = ty(Math.max(...ys.filter((v) => opts.yMax == null || v <= opts.yMax)));
  const padY = (hi - lo || 1) * 0.08;
  lo -= padY;
  hi += padY;
  if (opts.yFloor != null) lo = Math.max(lo, ty(opts.yFloor));
  if (opts.yMin != null) lo = ty(opts.yMin);
  if (opts.yMax != null) hi = ty(opts.yMax);
  const yMin = fy(lo), yMax = fy(hi);

  const tx = opts.log ? Math.log10 : (v) => v;
  const X = (v) => M.left + (tx(v) - tx(xMin)) / ((tx(xMax) - tx(xMin)) || 1)
    * (W - M.left - M.right);
  const Y = (v) => H - M.bottom
    - (ty(Math.min(v, yMax)) - lo) / (hi - lo) * (H - M.top - M.bottom);

  // grid and y ticks: decades on a log axis, round steps otherwise
  const ticks = [];
  if (opts.yLog) {
    for (let e = Math.ceil(lo); e <= hi; e++) ticks.push(10 ** e);
  } else {
    const step = niceStep((yMax - yMin) / 5);
    for (let v = Math.ceil(yMin / step) * step; v <= yMax; v += step) {
      ticks.push(Math.abs(v) < step / 1e6 ? 0 : v);
    }
  }
  for (const v of ticks) {
    el("line", { x1: M.left, x2: W - M.right, y1: Y(v), y2: Y(v),
                 stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: M.left - 8, y: Y(v) + 4, "text-anchor": "end" }, svg)
      .textContent = opts.yFmt(v);
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

  // shaded x ranges, labeled at their top left
  for (const span of opts.spans || []) {
    el("rect", { x: X(span.x0), y: M.top, width: X(span.x1) - X(span.x0),
                 height: H - M.top - M.bottom, fill: "var(--ink)", "fill-opacity": 0.06 }, svg);
    if (span.label) {
      el("text", { x: X(span.x0) + 6, y: M.top + 12, class: "label" }, svg).textContent = span.label;
    }
  }

  // reference lines: dashed, or dotted with `dotted`
  for (const ref of opts.refs || []) {
    const dash = { stroke: ref.color || "var(--muted)", "stroke-width": 1.5,
                   "stroke-dasharray": ref.dotted ? "2 4" : "5 4" };
    if (ref.y != null) {
      el("line", { x1: M.left, x2: W - M.right, y1: Y(ref.y), y2: Y(ref.y), ...dash }, svg);
      // labeled at the left end, which `xMin` leaves empty on the rates chart
      if (ref.label) {
        el("text", { x: M.left + 6, y: Y(ref.y) - 5, class: "label" }, svg)
          .textContent = ref.label;
      }
    } else {
      el("line", { x1: X(ref.x), x2: X(ref.x), y1: M.top, y2: H - M.bottom, ...dash }, svg);
      if (ref.label) {
        // a label near the right edge goes on the line's left
        const right = X(ref.x) > W * 0.8;
        // `row` drops a label a line, below a neighbor it would overlap
        el("text", { x: X(ref.x) + (right ? -5 : 5), y: M.top + 10 + 14 * (ref.row || 0),
                     class: "label",
                     "text-anchor": right ? "end" : "start" }, svg).textContent = ref.label;
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
                      "fill-opacity": s.bandOpacity ?? 0.14 }, svg);
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
                       ...(s.dash ? { "stroke-dasharray": s.dotted ? "2 4" : "5 4" } : {}) },
         svg);
    }
    if (s.markers) {
      for (const p of pts) {
        if (p.y > yMax) {
          const x = X(p.x), y = Y(yMax);
          el("path", { d: `M${x - 5},${y + 4}L${x + 5},${y + 4}L${x},${y - 4}Z`,
                       fill: s.color }, svg);
          if (p.clipLabel) {
            el("text", { x: x + 8, y: y + 4, class: "label" }, svg).textContent = p.clipLabel;
          }
          continue;
        }
        el("circle", { cx: X(p.x), cy: Y(p.y), r: s.ring?.(p) ? 6 : s.r || 4, fill: s.color,
                       stroke: s.ring?.(p) ? "var(--ink)" : "var(--surface)",
                       "stroke-width": 2 }, svg);
        // `name` says which setting a point is, beside it
        if (p.name) {
          el("text", { x: X(p.x) + 9, y: Y(p.y) - 7, class: "label" }, svg).textContent = p.name;
        }
      }
    }
  }

  // a name at the right end of each line, nudged apart where they would touch
  if (opts.endLabels) {
    const ends = opts.series.map((s) => {
      const last = s.points.filter((p) => p.y != null).at(-1);
      return { s, y: Y(last.y) };
    }).sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) {
      ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 14);
    }
    for (const { s, y } of ends) {
      el("circle", { cx: W - M.right + 10, cy: y, r: 3.5, fill: s.color }, svg);
      el("text", { x: W - M.right + 18, y: y + 4, class: "label" }, svg).textContent = s.name;
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
      const p = s.hover === false ? null : s.points.find((q) => q.x === best);
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
  note("rates-note", "At each puzzle, the share of the last 25, 50 or 100 answers that "
    + "were off-list or repeats. The dashed lines are the probabilities the prior "
    + "assumes for each.",
  "Each point is a share of the window ending at that puzzle: how many answers were "
    + "off the curated list, and how many repeated an earlier answer. The off-list "
    + "share counts only the answers that were not repeats, because the prior decides "
    + "in that order, first whether the answer is a repeat and then, if not, whether "
    + "it is off-list. The shaded band is ±1 standard error, wide because a window "
    + "holds few such answers. The lines start once a full window has been played. "
    + "The strip below has one mark per answer since the regime began, colored by "
    + "kind.");
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
    { name: "off-list, as a share of the answers that were not repeats", color: "var(--s2)" },
    { name: "repeat, as a share of all answers", color: "var(--s1)" },
    { name: "the prior's probability for each", color: "var(--muted)", kind: "dash" },
  ]);
  // the x axis spans the whole regime, so the strip below lines up with it
  const first = rows[start][0], last = rows[rows.length - 1][0];
  lineChart($("rates-chart"), {
    xMin: first, xMax: last, height: 280,
    series: [{ name: "off-list", color: "var(--s2)", points: offlist, band: true },
             { name: "repeat", color: "var(--s1)", points: repeat, band: true }],
    refs: [{ y: offProb, color: "var(--s2)", label: `off-list default ${offProb}` },
           { y: repeatProb, color: "var(--s1)", label: `repeat default ${repeatProb}` }],
    yFloor: 0, yFmt: (v) => pct(v, 0), yLabel: `share of the last ${w} answers`,
    xTicks: ticksBetween(first, last, 6), xFmt: (v) => `#${v}`,
    xLabel: "the puzzle the window ends at",
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
  const W = node.clientWidth || 900, H = 46, left = 58, right = 18;
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
  note("basin-note", "One of the prior's two probabilities is set to each value across, "
    + "and every real puzzle is replayed; up is how many more guesses per game that "
    + "costs than the shipped value. The shaded range costs about nothing; the dotted "
    + "line is the rate the answer record shows.",
  "This asks how much the prior's two probabilities matter. One of them is swept "
    + "across a range of settings while everything else "
    + "stays shipped, including <code>f</code>, which is held at the shipped prior's "
    + "fit. At each setting the solver replays every real puzzle since the regime "
    + "began, and each point is how many more guesses per game it needs than at the "
    + "shipped setting, on the same puzzles, with ±1 standard error. Lower is better, "
    + "and the shipped setting is zero by construction. Settings far from the shipped "
    + "one read a little worse than a refit would leave them, because they borrow an "
    + "<code>f</code> fitted for another prior.<br><br>"
    + "The shaded range is the basin: the settings whose cost over the whole regime "
    + "stays within <code>prior_basin.py</code>'s <code>FLAT</code> bound of the "
    + "shipped one. The dotted line is the rate the answer record actually shows. "
    + "<b>Last 100</b> scores the same runs on the latest puzzles only, and <b>by "
    + "answer type</b> splits each point into repeats, off-list and fresh answers, "
    + "which can move in opposite directions as a setting changes. "
    + provenance("prior_basin"));
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
    // the basin `prior_basin.py` found, and the rate the record shows: the
    // regime's for repeats, the last 100 puzzles' for off-list
    const from = v[`${state.axis}.flat_from`], to = v[`${state.axis}.flat_to`];
    const measured = state.axis === "noncurated"
      ? [v.recent_noncurated, `last ${DATA.defaults.recent}: ${v.recent_noncurated?.toFixed(3)}`]
      : [v.regime_repeat, `regime: ${v.regime_repeat?.toFixed(3)}`];
    lineChart($("basin-chart"), {
      series, zero: true, height: 300, xMin: 0, xMax: 1,
      spans: from != null ? [{ x0: from, x1: to, label: "the basin" }] : [],
      refs: [{ x: sweep.base, label: `shipped ${sweep.base}` },
             ...(measured[0] != null ? [{ x: measured[0], dotted: true, label: measured[1], row: 1 }] : [])],
      xTicks: ticksBetween(0, 1, 10), xFmt: (x) => x.toFixed(1),
      xLabel: `${name}, the setting replayed`,
      yLabel: "extra guesses per game",
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
    yLabel: "extra guesses per game",
    yFmt: (y) => signed(y, 2),
    tip: (x) => `<div><b>temper ${x.toFixed(2)}</b></div>` + series.map((s) =>
      tipRow(s.color, s.name, pm(s.points.find((p) => p.x === x).cell))).join(""),
  });
}

// --------------------------------------------------------------- refit lag

function drawLag() {
  const v = values("refit_lag");
  note("lag-note", "Extra guesses per game when the answer record is k days behind, and "
    + "how much of the prior that moves.",
  "This asks what it costs to update the answer record late. "
    + "With a lag of k days, the solver plays each regime puzzle with a record that "
    + "is missing its last k answers, so it still treats those k words as unused and "
    + "gives them a fresh word's weight. Nothing is refitted. Each point on the left "
    + "is the average extra guesses per game against a record updated every day, on "
    + "the same puzzles, with ±1 standard error; positive means the stale record "
    + "costs guesses. The tooltip says how many games changed at all.<br><br>"
    + "On the right is how far the lag moves the prior: the share of its probability "
    + "that sits on different words, averaged over the puzzles. The dashed line is a "
    + "yardstick, how much flattening the repeat shape moves it. Below, refitting "
    + "every k days averages the lags from 0 to k − 1, since the record is that "
    + "stale for one day each. " + provenance("refit_lag"));
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
    zero: true, height: 260, log: true,
    xTicks: lags.lags, xFmt: (x) => String(x), xLabel: "days of lag, log scale",
    yLabel: "extra guesses per game", yFmt: (y) => signed(y, 2),
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
    yFloor: 0, height: 260, log: true,
    refs: [{ y: v.yardstick / 100, label: "repeat shape flattened" }],
    xTicks: lags.lags, xFmt: (x) => String(x), xLabel: "days of lag, log scale",
    yLabel: "prior mass moved", yFmt: (y) => pct(y, 1),
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
    yLabel: "extra guesses per game", yFmt: (y) => signed(y, 3),
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
  // the two stories: a fixed window at the median source (slope 0), and a
  // fixed replay lag at the median lag (slope 1)
  const median = (xs) => {
    const s = xs.slice().sort((a, b) => a - b), k = (s.length - 1) / 2;
    return (s[Math.floor(k)] + s[Math.ceil(k)]) / 2;
  };
  const source = median(events.map((e) => e.y));
  const lag = median(events.map((e) => e.x - e.y));
  const x0 = events[0].x, x1 = events[events.length - 1].x;
  const flat = [x0, x1].map((x) => ({ x, y: source }));
  const steep = [x0, x1].map((x) => ({ x, y: x - lag }));
  // `drift.py`'s Theil-Sen line, its band pivoting at the median repeat
  const d = values("drift");
  const fit = d.theil_sen != null && d["theil_sen.intercept"] != null;
  const pivot = median(events.map((e) => e.x));
  const at = pivot * (fit ? d.theil_sen : 0) + (fit ? d["theil_sen.intercept"] : 0);
  const trend = [x0, x1].map((x) => {
    const a = at + (fit ? d["theil_sen.lo"] : 0) * (x - pivot);
    const b = at + (fit ? d["theil_sen.hi"] : 0) * (x - pivot);
    return { x, y: at + (fit ? d.theil_sen : 0) * (x - pivot),
             lo: Math.min(a, b), hi: Math.max(a, b) };
  });
  // the median source over each repeat and its four neighbors on either side
  const running = events.map((e, i) => ({
    x: e.x, y: median(events.slice(Math.max(0, i - 4), i + 5).map((f) => f.y)) }));
  // a few far-off sources would squash the band the points sit in, so the
  // axis stops at half again the 90th percentile and they sit on its edge
  const sources = events.map((e) => e.y).sort((a, b) => a - b);
  const cap = Math.ceil(1.5 * sources[Math.floor(0.9 * (sources.length - 1))] / 50) * 50;
  for (const e of events) e.clipLabel = `#${e.y}`;
  const clipped = events.filter((e) => e.y > cap).length;
  note("repeats-note", "Each dot is a repeat: across is the puzzle it fell on, up is "
    + "the puzzle that first used the word. The two dashed lines are what two "
    + "different habits of the editor would produce; the solid lines are what the "
    + "dots do.",
  `There are ${events.length} repeats: across, the puzzle each fell on; up, the `
    + "puzzle that first used the word. Two "
    + "stories would explain where the editor takes repeats from. If she recycles "
    + "words a fixed time after they were first used, the source moves forward one "
    + "puzzle for every puzzle that passes, and the dots climb along the steep line "
    + "(slope 1). If she draws from a fixed early block of the archive, the source "
    + "stays put however late the repeat falls, and the dots lie along the flat line "
    + `(slope 0). The flat line sits at the median source, #${source}; the steep `
    + `line trails each puzzle by the median lag, ${lag} puzzles. Both are medians, `
    + "which the far-off sources barely move."
    + (fit ? "<br><br>The black line and its fan are the trend, a Theil–Sen line from "
      + "<code>analysis/drift.py</code>: the median of the slopes between every pair of "
      + `repeats, ${signed(d.theil_sen, 2)} source puzzles per puzzle. The fan is its `
      + `68% bootstrap interval, ${signed(d["theil_sen.lo"], 2)} to `
      + `${signed(d["theil_sen.hi"], 2)}; the 95% interval is `
      + `${signed(d["theil_sen.lo95"], 2)} to ${signed(d["theil_sen.hi95"], 2)}. `
      + `Each repeat is in ${events.length - 1} of the `
      + `${events.length * (events.length - 1) / 2} pairs, so a few far-off ones `
      + "barely move the median." : "")
    + " The thin line is a running median: the median source over each repeat and "
    + "the four on either side of it, which follows the dots without assuming a "
    + "straight line."
    + (clipped ? ` The ${clipped} repeats that came from later than #${cap} sit on the `
      + "top edge as triangles, labeled with where they came from." : ""));
  legend($("repeats-legend"), [
    { name: "a repeat", color: "var(--s1)", kind: "dot" },
    { name: "a fixed window (slope 0)", color: "var(--s3)", kind: "dash" },
    { name: "a fixed replay lag (slope 1)", color: "var(--s2)", kind: "dash" },
    ...(fit ? [{ name: "the trend (Theil–Sen), 68% band", color: "var(--ink)" }] : []),
    { name: "running median of 9", color: "var(--s5)" },
  ]);
  const guides = [
    { name: "fixed window", color: "var(--s3)", dash: true, hover: false, points: flat },
    { name: "replay lag", color: "var(--s2)", dash: true, hover: false, points: steep },
    { name: "running median", color: "var(--s5)", hover: false, points: running },
  ];
  if (fit) {
    guides.push({ name: "trend", color: "var(--ink)", band: true, bandOpacity: 0.08,
                  hover: false, points: trend });
  }
  lineChart($("repeats-chart"), {
    series: [...guides,
             { name: "repeat", color: "var(--s1)", line: false, markers: true, points: events }],
    yMin: 0, yMax: cap, height: 300,
    xTicks: ticksBetween(x0, x1, 6), xFmt: (v) => `#${v}`,
    xLabel: "the puzzle the repeat fell on", yLabel: "the puzzle that first used it",
    yFmt: (v) => `#${Math.round(v)}`,
    tip: (x) => {
      const e = events.find((ev) => ev.x === x);
      return `<div><b>${e.word}</b></div><div>#${e.x}, ${e.date}</div>`
        + `<div>first #${e.y}, ${e.fromDate}</div>`
        + `<div>running median #${running.find((p) => p.x === x).y}</div>`
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
  note("pool-note", `Curated words never yet an answer: ${last.y.toLocaleString()} left `
    + "after #" + last.x + (rate ? `, projected to run out around ${ends}.` : "."),
  `The curated list is the ${total.toLocaleString()} words the solver treats as the `
    + "editor's likely answers. The line falls by one each time a curated word is used "
    + "for the first time; repeats and off-list answers leave it alone. "
    + (rate ? "The dashed line carries it forward at the rate curated words were used "
      + `for the first time since the regime began, ${rate.toFixed(0)} a year, from `
      + "<code>analysis/regime.py</code>, and assumes that rate holds." : ""));
  legend($("pool-legend"), [
    { name: "curated words never yet an answer", color: "var(--s1)" },
    ...(rate ? [{ name: "projected at the regime's rate", color: "var(--s1)", kind: "dash" }] : []),
  ]);
  lineChart($("pool-chart"), {
    series, yFloor: 0, height: 240,
    refs: [{ x: rows[start][0], label: "regime begins" },
           ...(rate ? [{ x: series[1].points[1].x, label: `runs out ${ends.slice(0, 7)}` }] : [])],
    xTicks: ticksBetween(1, series[series.length - 1].points.at(-1).x, 6),
    xFmt: (v) => `#${Math.round(v)}`, xLabel: "puzzle", yLabel: "words unused",
    yFmt: (v) => Math.round(v).toLocaleString(),
    tip: (x) => {
      const p = points[x - 1];
      return `<div><b>#${p.x}</b> ${p.date}</div>`
        + tipRow("var(--s1)", "unused", p.y.toLocaleString());
    },
  });
}

// ------------------------------------------------------------- replay arms

// each table, the baseline `replay.py` pairs its rows against (its BASE), and
// what its rows change
const ARM_TABLES = [
  ["priors.arm_table", "the four priors", "the shipped prior, both priors on",
   "Each row turns one or both priors off and keeps everything else. <b>none</b> has "
   + "neither, so every word in the list is equally likely; <b>curation only</b> turns "
   + "the repeat prior off; <b>repeat only</b> turns the curation prior off. A dot right "
   + "of zero means the solver needs more guesses without that prior, so how far right "
   + "it sits is what the missing prior is worth on that kind of answer. The repeat "
   + "prior sets the share of probability that already-used words get, so it acts "
   + "mostly on repeats; the curation prior splits the rest between the curated list "
   + "and every other word, so it acts on off-list and fresh answers.",
   "the prior turned off",
   { "none": "both priors off", "curation only": "repeat prior off",
     "repeat only": "curation prior off" }],
  ["shapes.arm_table", "the four shapes", "both shapes flat",
   "Every row has both priors on. What changes is each prior's shape: how it spreads "
   + "its share of probability over its own words, either flat (every word equal) or "
   + "sharpened (some words weighted more). The repeat shape is <code>binned</code>, "
   + "weighting used words by where in the old answer order repeats have come from; "
   + "the curation shapes are <code>wordlebot</code>, WordleBot's weights, and "
   + "<code>frequency</code>, word frequency. A dot left of zero means the sharpened "
   + "shape saves guesses against leaving it flat. The last row is the shipped pair.",
   "the shapes sharpened",
   { "curation wordlebot alone": "curation shape only",
     "repeat binned alone": "repeat shape only",
     "binned + frequency": "both, curation by frequency",
     "binned + wordlebot": "both, as shipped" }],
  ["arm_table", "the curation shape", "the shipped wordlebot shape",
   "The one row swaps the curation prior's shape from WordleBot's weights, which ship, "
   + "to word frequency, and keeps everything else shipped. A dot right of zero means "
   + "word frequency needs more guesses than the shipped shape on that kind of answer.",
   "the curation shape used", { "frequency": "word frequency" }],
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
  const [key, , base, rowsMean, rowsTitle, rename] = tables.find(([k]) => k === state.replay);
  // the arm sizes and the mixture are recorded under the table's own prefix
  const prefix = key.slice(0, key.length - "arm_table".length);
  const count = (name) => v[`${prefix}arms.${name}`];
  note("replay-note", "Each row is a change to the solver, replayed on real answers. "
    + "A dot is how many more guesses per game the change costs on one kind of answer, "
    + `against ${base}: right of zero costs guesses, left saves them.`,
  "This replays real answers. Each one is played twice "
    + "with the answer record as it stood the day before, once under a row's setting "
    + `and once under the baseline, ${base}, and each dot is the average difference `
    + "in guesses per game, with ±1 standard error. The answers come in three kinds, "
    + "because a prior can help on one kind and cost on another: "
    + (count("repeats") != null
      ? `the ${count("repeats")} repeats, the ${count("offlist")} answers ever taken `
        + `from off the curated list, and the ${count("fresh")} fresh curated answers `
        + "since the regime began. "
      : "repeats, off-list answers and fresh curated answers. ")
    + "<b>Blended</b> combines the three in the proportions the regime has shown"
    + (v[`${prefix}mixture`] ? ` (${v[`${prefix}mixture`]}, in that order)` : "")
    + ", which is what a game drawn now would face.<br><br>" + rowsMean + "<br><br>"
    + provenance("replay"));
  const { columns, rows } = parseTable(v[key]);
  // blended is the summary of the other three, so it wears the ink color
  const kinds = { repeat: "repeat answers", "off-list": "off-list answers",
                  fresh: "fresh curated answers", blended: "blended, as a game now would face" };
  const series = columns.slice(1).map((name, i) => ({
    name: kinds[name] || name, color: name === "blended" ? "var(--ink)" : ARM_COLORS[i],
    cells: rows.map((r) => parsePm(r[i + 1])),
  }));
  legend($("replay-legend"), series);
  const labels = rows.map((r) => {
    const raw = r[0].replace(/[`*]/g, "");
    return rename[raw] || raw;
  });
  forestChart($("replay-chart"), labels, series,
               `extra guesses per game against ${base}`, rowsTitle);
}

// one band per row, one error bar per series inside it, on a shared x axis
function forestChart(node, labels, series, xLabel, rowsTitle) {
  node.innerHTML = "";
  const W = node.clientWidth || 900, band = 16 * series.length + 16;
  const M = { top: rowsTitle ? 26 : 8, right: 18, bottom: 34, left: 210 };
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
               class: "label" }, svg).textContent = xLabel;
  el("text", { x: M.left, y: H - 2, class: "label" }, svg).textContent = "← saves guesses";
  el("text", { x: W - M.right, y: H - 2, "text-anchor": "end", class: "label" }, svg)
    .textContent = "costs guesses →";
  if (rowsTitle) {
    el("text", { x: M.left - 12, y: 14, "text-anchor": "end", class: "label",
                 "font-weight": 600 }, svg).textContent = rowsTitle;
  }

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
  note("fits-note", "The estimated guesses to finish from a group of n equally likely "
    + "candidates, one curve per prior. Below: each curve minus the shipped prior's.",
  "To score a guess, the solver splits the remaining "
    + "candidates by the feedback the guess would get, and needs a value for each of "
    + "those groups: how many more guesses it takes to finish from there. "
    + "<code>f(n)</code> is that estimate, as a curve in n, the group's effective size, "
    + "2 to the power of its entropy, which is the plain count when every candidate is "
    + "equally likely. Groups of one or two candidates are valued exactly, so the "
    + "curve's left end is never used. Each prior gets its own curve, fitted to "
    + "games played under it, in " + forms.join(" and ") + ". A lower curve means "
    + "that prior expects to finish a group of the same effective size in fewer "
    + "guesses. The second chart subtracts "
    + "the shipped prior's curve, so the differences the first chart compresses show.");
  legend($("fits-legend"), fits.map((f) => ({
    name: `${f.label} (fitted ${f.fitted_at})`, color: f.color })));
  const series = fits.map((f) => ({
    name: f.label, color: f.color,
    points: f.n.map((n, i) => ({ x: n, y: f.f[i] })),
  }));
  const top = fits[0].n[fits[0].n.length - 1];
  lineChart($("fits-chart"), {
    series, log: true, endLabels: true, right: 130, height: 320,
    xTicks: [1, 10, 100, 1000, 10000, top], xFmt: (v) => v.toLocaleString(),
    xLabel: "effective size n, log scale", yLabel: "guesses to finish",
    yFmt: (v) => v.toFixed(1),
    tip: (x) => `<div><b>n = ${x.toLocaleString()}</b></div>` + series.map((s) =>
      tipRow(s.color, s.name, s.points.find((p) => p.x === x).y.toFixed(3))).join(""),
  });

  // the same curves less the shipped prior's, which is the `both` fit
  const shipped = fits.find((f) => f.label === "both");
  if (shipped) {
    const diffs = fits.filter((f) => f !== shipped).map((f) => ({
      name: f.label, color: f.color,
      points: f.n.map((n, i) => ({ x: n, y: f.f[i] - shipped.f[i] })),
    }));
    lineChart($("fits-diff"), {
      series: diffs, log: true, zero: true, endLabels: true, right: 130, height: 260,
      xTicks: [1, 10, 100, 1000, 10000, top], xFmt: (v) => v.toLocaleString(),
      xLabel: "effective size n, log scale", yLabel: "guesses, minus shipped",
      yFmt: (v) => signed(v, 2),
      tip: (x) => `<div><b>n = ${x.toLocaleString()}</b></div>` + diffs.map((s) =>
        tipRow(s.color, s.name, signed(s.points.find((p) => p.x === x).y, 3))).join(""),
    });
  }

  // one panel per prior on a shared axis, from round 1; the entropy seed is
  // left off, and a point above the axis sits on its edge as a triangle
  const gaps = fits.map((f) => {
    const best = Math.min(...f.rounds.map((r) => r[1]));
    return f.rounds.map(([label, mean], i) => ({
      x: i, y: mean - best, mean, label, clipLabel: `+${(mean - best).toFixed(3)}`,
    }));
  });
  const seeds = gaps.map((g) => g[0].y);
  const later = gaps.flatMap((g) => g.slice(2).map((p) => p.y));
  const cap = Math.max(...later) * 1.3;
  const most = Math.max(...gaps.map((g) => g.length)) - 1;
  note("rounds-note", "Each round's mean guesses above the prior's best round. The "
    + "ringed round's curve is the one that ships.",
  "f is fitted by policy iteration. The first round, "
    + "the seed, plays 2,000 sampled games by greedy entropy and records how many "
    + "guesses each position really took; a curve is fitted to that; the next round "
    + "plays the same 2,000 games using the new curve, and so on. Each panel is one "
    + "prior, and each point is a round's mean guesses above that prior's best round, "
    + "so the best round sits at zero, ringed. The loop does not settle on its best "
    + "round, so it keeps the curve from the best round, and that is the curve that "
    + `ships. The seed is left off; it sits ${Math.min(...seeds).toFixed(2)} to `
    + `${Math.max(...seeds).toFixed(2)} above the best round. All panels share one `
    + "axis, and a point above it sits on its edge as a labeled triangle.");
  const grid = $("rounds-chart");
  grid.className = "panels";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(250px, 1fr))";
  grid.innerHTML = "";
  const charts = fits.map((f) => {
    const box = document.createElement("div");
    box.innerHTML = `<h3>${f.label}</h3>`;
    const chart = document.createElement("div");
    chart.className = "chart";
    box.appendChild(chart);
    grid.appendChild(box);
    return chart;
  });
  fits.forEach((f, i) => {
    const s = { name: f.label, color: f.color, markers: true, ring: (p) => p.y === 0,
                points: gaps[i].slice(1) };
    lineChart(charts[i], {
      series: [s], yMin: 0, yMax: cap, height: 170, xMin: 1, xMax: most,
      xTicks: Array.from({ length: most }, (_, k) => k + 1), xFmt: (v) => String(v),
      xLabel: "round", yLabel: "guesses above best", yFmt: (v) => `+${v.toFixed(3)}`,
      tip: (x) => {
        const p = s.points.find((q) => q.x === x);
        return `<div><b>${f.label}, round ${x}</b></div>`
          + tipRow(f.color, "mean", p.mean.toFixed(4))
          + tipRow(f.color, "above the best", `+${p.y.toFixed(4)}`);
      },
    });
  });
}

// ------------------------------------------------------------ head to head

// grouped vertical bars from a zero baseline, negatives below it
function barChart(node, { categories, series, fmt, yLabel, height = 240, labels = false }) {
  node.innerHTML = "";
  const W = node.clientWidth || 900, H = height, M = { top: 12, right: 18, bottom: 30, left: 58 };
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
      // `colors` gives each bar its own color, where the sign carries meaning
      el("rect", { x: x0 + i * (bar + 2), y: top, width: bar, height: h, rx: 3,
                   fill: s.colors?.[c] || s.color }, svg);
      // `labels` prints the value of a bar too short to read, at its end
      if (labels && h < 14) {
        const below = v < 0;
        el("text", { x: x0 + i * (bar + 2) + bar / 2, y: below ? Y(v) + 14 : Y(v) - 5,
                     "text-anchor": "middle", class: "label" }, svg)
          .textContent = fmt(v, true);
      }
    });
    const hit = el("rect", { x: M.left + c * slot, y: M.top, width: slot,
                             height: H - M.top - M.bottom, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", () => {
      const box = svg.getBoundingClientRect();
      tip.innerHTML = `<div><b>${name}</b></div>` + series.map((s) =>
        tipRow(s.colors?.[c] || s.color, s.describe?.(s.values[c]) || s.name,
               fmt(s.values[c], true))).join("");
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
  note("h2h-note", "The shipped solver against another, with every word in the list "
    + "played as the answer. The top chart is the share of games each solves in 2, 3, "
    + "4, 5 or 6 guesses, or loses; the bottom one is how far ahead the shipped solver "
    + "is at each point of the game.",
  "Both solvers play every one of the 12,947 words as the "
    + "answer, and each game counts by how likely the default prior thinks that word "
    + "is, so these are exact shares with no sampling error. The other solver is "
    + "either <code>IG + 2·p_win</code>, which scores a guess by the information it "
    + "gains plus two bits for the chance it is the answer and needs no fitted curve, "
    + "or greedy entropy, which takes the guess that splits the candidates most "
    + "evenly.<br><br>The top chart is the share of games each finishes in 2, 3, 4, 5 "
    + "or 6 guesses, or loses, on a log scale so the rare rows show; each gridline is "
    + "ten times the one below. The bottom chart adds those shares up, the share "
    + "finished within k guesses, and subtracts the other solver's from the shipped "
    + "one's, in percentage points: green where the shipped solver has finished more "
    + "games by then, red where it has finished fewer. " + provenance("head_to_head"));
  const { columns, rows } = parseTable(v[state.h2h]);
  const names = columns.slice(1).map((c) => c.replace(/`/g, ""));
  const series = names.map((name, i) => ({
    name, color: ARM_COLORS[i], values: rows.map((r) => parseNum(r[i + 1])),
  }));
  legend($("h2h-legend"), series.map((s) => ({ ...s, kind: "dot" })));
  const categories = rows.map((r) => r[0]);
  // the two solvers side by side in each column
  const offset = (i) => (i - (series.length - 1) / 2) * 0.16;
  const dots = series.map((s, i) => ({
    name: s.name, color: s.color, line: false, markers: true, r: 6.5,
    points: s.values.map((y, c) => ({ x: c + offset(i), y })),
  }));
  lineChart($("h2h-chart"), {
    series: dots, yLog: true, height: 260, xMin: -0.5, xMax: categories.length - 0.5,
    xTicks: categories.map((_, c) => c), xFmt: (c) => categories[c],
    xLabel: "guesses taken to solve", yLabel: "share of games, log scale",
    yFmt: (y) => `${y >= 1 ? y.toFixed(0) : y >= 0.1 ? y.toFixed(1) : y.toPrecision(1)}%`,
    tip: (x) => {
      const c = Math.round(x);
      return `<div><b>${categories[c]}</b></div>` + series.map((s) =>
        tipRow(s.color, s.name, `${s.values[c]}%`)).join("")
        + `<div>ratio ${(series[0].values[c] / series[1].values[c]).toFixed(2)}</div>`;
    },
  });
  // the share finished within k guesses, so ahead always means better
  const within = categories.slice(0, -1);
  const cumulative = (s) => within.map((_, c) =>
    s.values.slice(0, c + 1).reduce((a, b) => a + b, 0));
  const ahead = cumulative(series[0]).map((x, c) => x - cumulative(series[1])[c]);
  barChart($("h2h-diff"), {
    categories: within.map((k) => `within ${k}`), height: 220, yLabel: "percentage points",
    labels: true,
    series: [{ name: `${names[0]} − ${names[1]}`, values: ahead,
               describe: (x) => `${names[0]} ${x >= 0 ? "ahead" : "behind"}`,
               colors: ahead.map((x) => (x >= 0 ? "var(--good)" : "var(--bad)")) }],
    fmt: (x, exact) => `${x < 0 ? "−" : "+"}${Math.abs(x).toFixed(exact ? 2 : 1)}`,
  });
}

// ----------------------------------------------------------- column panels

// one dot plot per numeric column of a table on its own axis, the bold row ringed;
// `opts` titles the rows, renames, skips, or `split`s a column into a two-part bar
function columnPanels(node, text, opts = {}) {
  node.innerHTML = "";
  const { columns, rows } = parseTable(text);
  const labels = rows.map((r) => r[0].replace(/[`*]/g, ""));
  const bold = rows.map((r) => r.some((c) => c.includes("**")));
  const plain = (c) => c.replace(/[`*]/g, "");
  const index = (name) => columns.findIndex((c) => plain(c) === name);
  // every panel goes into the grid before any is measured
  const panels = [];
  columns.slice(1).forEach((column, k) => {
    const i = k + 1, name = plain(column);
    if ((opts.skip || []).includes(name)) return;
    const vals = rows.map((r) => parseNum(r[i]));
    if (vals.every((x) => x == null)) return;
    const split = opts.split?.[name];
    const box = document.createElement("div");
    box.innerHTML = `<h3>${opts.headings?.[name] || inline(column)}</h3>`;
    if (split) {
      box.insertAdjacentHTML("beforeend", '<div class="legend"></div>');
      legend(box.lastChild, [{ name: split.part, color: "var(--s3)", kind: "dot" },
                             { name: split.rest, color: "var(--s2)", kind: "dot" }]);
    }
    const chart = document.createElement("div");
    chart.className = "chart";
    box.appendChild(chart);
    node.appendChild(box);
    // a split's part, where the table can also say "all of them"
    const part = split && rows.map((r, n) => parseNum(r[index(split.column)]) ?? vals[n]);
    panels.push([chart, vals, i, part]);
  });
  for (const [chart, vals, i, part] of panels) {
    // drawn at the panel's own width, so the text stays at its set size
    const W = Math.round(chart.getBoundingClientRect().width) || 420;
    const row = 26, M = { top: opts.rows ? 22 : 6, right: 18, bottom: 24, left: 170 };
    const H = M.top + row * rows.length + M.bottom;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, chart);
    const tip = document.createElement("div");
    tip.className = "tip";
    tip.hidden = true;
    chart.appendChild(tip);
    if (opts.rows) {
      el("text", { x: M.left - 8, y: 12, "text-anchor": "end", class: "label",
                   "font-weight": 600 }, svg).textContent = opts.rows;
    }
    const known = vals.filter((x) => x != null);
    let lo = Math.min(...known), hi = Math.max(...known);
    // a share or a count starts at zero; anything else is zoomed to its own
    // range, and the heading says how narrow that range is
    const share = rows.some((r) => r[i].includes("%"));
    if (part || (known.every((x) => x >= 0)
                 && (share || lo === 0 || known.every(Number.isInteger)))) {
      lo = 0;
    } else {
      const span = hi - lo;
      const digits = Math.max(0, -Math.floor(Math.log10(span || 1)) + 1);
      chart.parentNode.querySelector("h3").insertAdjacentHTML("beforeend",
        ` <span class="zoom">zoomed: ${lo.toFixed(digits)} to ${hi.toFixed(digits)}, `
        + `a spread of ${span.toFixed(digits)}</span>`);
    }
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.06;
    const X = (x) => M.left + (x - lo + pad) / (hi - lo + 2 * pad) * (W - M.left - M.right);
    // the x axis the dots are placed on, with a few round ticks
    const base = M.top + row * rows.length;
    el("line", { x1: M.left, x2: W - M.right, y1: base, y2: base,
                 stroke: "var(--axis)", "stroke-width": 1 }, svg);
    const step = niceStep((hi - lo) / 3);
    const decimals = Math.max(0, -Math.floor(Math.log10(step)));
    for (let t = Math.ceil(lo / step) * step; t <= hi + step / 1e6; t += step) {
      el("line", { x1: X(t), x2: X(t), y1: base, y2: base + 4,
                   stroke: "var(--axis)", "stroke-width": 1 }, svg);
      el("text", { x: X(t), y: base + 16, "text-anchor": "middle" }, svg)
        .textContent = t.toFixed(decimals) + (share ? "%" : "");
    }
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
      if (part) {
        // the part from zero, the rest after it, a 2px gap between
        const mid = X(part[k]);
        el("rect", { x: X(0), y: y - 7, width: Math.max(0, mid - X(0) - 1), height: 14,
                     rx: 2, fill: "var(--s3)" }, svg);
        if (x > part[k]) {
          el("rect", { x: mid + 1, y: y - 7, width: Math.max(1, X(x) - mid - 1), height: 14,
                       rx: 2, fill: "var(--s2)" }, svg);
        }
      } else {
        el("circle", { cx: X(x), cy: y, r: bold[k] ? 6 : 4.5, fill: "var(--s1)",
                       stroke: bold[k] ? "var(--ink)" : "var(--surface)", "stroke-width": 2 }, svg);
      }
      // the exact value is in a tooltip over the row
      const hit = el("rect", { x: 0, y: y - row / 2, width: W, height: row,
                               fill: "transparent" }, svg);
      hit.addEventListener("pointermove", () => {
        const unit = share ? "%" : "";
        tip.innerHTML = `<div><b>${labels[k]}</b></div>` + (part
          ? tipRow("var(--s3)", opts.split[plain(columns[i])].part, `${part[k]}${unit}`)
            + tipRow("var(--s2)", opts.split[plain(columns[i])].rest,
                     `${(x - part[k]).toFixed(1)}${unit}`)
          : tipRow("var(--s1)", plain(columns[i]), r[i].replace(/\*\*/g, "")));
        tip.hidden = false;
        const box = svg.getBoundingClientRect();
        const px = X(x) * box.width / W;
        const flip = px > box.width * 0.6;
        tip.style.left = flip ? "" : `${px + 12}px`;
        tip.style.right = flip ? `${box.width - px + 12}px` : "";
        tip.style.top = `${(y + row / 2) * box.height / H}px`;
      });
      hit.addEventListener("pointerleave", () => { tip.hidden = true; });
    });
  }
}

function tablePanels(script, options, ids, short, long) {
  const v = values(script);
  const shown = options.filter(([key]) => v[key]);
  const controls = $(ids.controls);
  controls.innerHTML = "";
  note(ids.note, short, `${long} ${provenance(script)}`);
  if (!shown.length) return missing($(ids.panels), script);
  if (!shown.some(([key]) => key === state[script])) state[script] = shown[0][0];
  segmented(controls, shown, state[script], (key) => {
    state[script] = key;
    tablePanels(script, options, ids, short, long);
  });
  // an option's third entry draws its table some other way than as panels
  const [, , draw] = shown.find(([key]) => key === state[script]);
  const node = $(ids.panels);
  node.className = draw ? "" : "panels";
  (draw || columnPanels)(node, v[state[script]]);
}

// one table as a trade-off: each row a point at (x column, y column), the
// default ringed; the axes are titled `xLabel` and `yLabel`
function tradeoff(xName, yName, xLabel, yLabel) {
  return (node, text) => {
    node.innerHTML = '<div class="chart"></div>';
    const { columns, rows } = parseTable(text);
    const xi = columns.indexOf(xName), yi = columns.indexOf(yName);
    const head = columns[0].replace(/`/g, "");
    const points = rows.map((r) => ({
      x: parseNum(r[xi]), y: parseNum(r[yi]), row: r,
      name: r[0].replace(/[`*]/g, "").replace(" (default)", ""),
      bold: r.some((c) => c.includes("**")),
    }));
    const s = { name: head, color: "var(--s1)", markers: true, r: 5,
                ring: (p) => p.bold, points };
    lineChart(node.firstChild, {
      series: [s], height: 300, right: 40,
      xTicks: ticksBetween(Math.min(...points.map((p) => p.x)),
                           Math.max(...points.map((p) => p.x)), 5),
      xFmt: (x) => `${x.toFixed(3)}%`, xLabel, yLabel,
      yFmt: (y) => y.toFixed(4),
      tip: (x) => {
        const p = points.find((q) => q.x === x);
        return `<div><b>${head} ${p.row[0].replace(/[`*]/g, "")}</b></div>`
          + columns.slice(1).map((c, i) => tipRow("var(--s1)", c, p.row[i + 1].replace(/\*\*/g, "")))
            .join("");
      },
    });
  };
}

// one table as stacked bars: a bar per row, a segment per column
function stacked(node, text) {
  node.innerHTML = '<div class="legend"></div><div class="chart"></div>';
  const { columns, rows } = parseTable(text);
  const names = columns.slice(1).map((c) => c.replace(/`/g, ""));
  const colors = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
  legend(node.firstChild, names.map((name, i) => ({ name, color: colors[i] })));
  const chart = node.lastChild;
  const W = chart.clientWidth || 900, H = 260, M = { top: 12, right: 18, bottom: 36, left: 58 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, chart);
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  chart.appendChild(tip);
  const counts = rows.map((r) => names.map((_, i) => parseNum(r[i + 1]) || 0));
  const top = Math.max(...counts.map((c) => c.reduce((a, b) => a + b, 0)));
  const Y = (v) => H - M.bottom - v / top * (H - M.top - M.bottom);
  const step = niceStep(top / 4);
  for (let v = 0; v <= top; v += step) {
    el("line", { x1: M.left, x2: W - M.right, y1: Y(v), y2: Y(v),
                 stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: M.left - 8, y: Y(v) + 4, "text-anchor": "end" }, svg)
      .textContent = v.toLocaleString();
  }
  const mid = (M.top + H - M.bottom) / 2;
  el("text", { x: 12, y: mid, "text-anchor": "middle", class: "label",
               transform: `rotate(-90 12 ${mid})` }, svg).textContent = "decisions made";
  const slot = (W - M.left - M.right) / rows.length, bar = Math.min(56, slot * 0.6);
  rows.forEach((r, k) => {
    const x = M.left + k * slot + (slot - bar) / 2;
    let acc = 0;
    counts[k].forEach((n, i) => {
      if (!n) return;
      // a 2px gap of the surface color between segments
      el("rect", { x, y: Y(acc + n), width: bar, height: Math.max(1, Y(acc) - Y(acc + n) - 2),
                   fill: colors[i] }, svg);
      acc += n;
    });
    el("text", { x: x + bar / 2, y: H - M.bottom + 16, "text-anchor": "middle",
                 class: "label" }, svg).textContent = r[0].replace(/[`*]/g, "");
    const hit = el("rect", { x: M.left + k * slot, y: M.top, width: slot,
                             height: H - M.top - M.bottom, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", () => {
      const box = svg.getBoundingClientRect();
      tip.innerHTML = `<div><b>${columns[0].replace(/`/g, "")} ${r[0]}</b></div>`
        + names.map((name, i) => tipRow(colors[i], name, r[i + 1])).join("");
      tip.hidden = false;
      const px = (M.left + (k + 1) * slot) * box.width / W;
      const flip = px > box.width * 0.6;
      tip.style.left = flip ? "" : `${px}px`;
      tip.style.right = flip ? `${box.width - (M.left + k * slot) * box.width / W}px` : "";
      tip.style.top = `${M.top * box.height / H}px`;
    });
    hit.addEventListener("pointerleave", () => { tip.hidden = true; });
  });
  el("text", { x: (M.left + W - M.right) / 2, y: H - 2, "text-anchor": "middle",
               class: "label" }, svg).textContent = columns[0].replace(/`/g, "");
}

// the loss tables' columns, said in full
const LOSS_PANELS = {
  rows: "policy", skip: ["solve rate"],
  headings: { "mean": "Mean guesses per game",
              "loss probability": "Share of games lost (not solved in six)",
              "wall clock": "Seconds to play every word" },
};

function drawLoss() {
  const panels = (node, text) => columnPanels(node, text, LOSS_PANELS);
  tablePanels("loss_penalty", [
    ["penalty.table", "the loss penalty",
     tradeoff("loss probability", "mean (loss = 7)", "share of games lost",
              "mean guesses, a loss counted as 7")],
    ["reach.table", "the search's reach",
     tradeoff("loss probability", "mean (loss = 7)", "share of games lost",
              "mean guesses, a loss counted as 7")],
    ["search.table", "how much is searched", panels], ["policies.table", "the policies", panels],
  ], { controls: "loss-controls", panels: "loss-panels", note: "loss-note" },
  "How often the solver loses a game against how many guesses it takes, as two of "
    + "its settings change. In the first two views each point is one setting, and "
    + "down and to the left is better on both; the default is ringed.",
  "A loss is a game not solved in six. <code>--loss-penalty</code> sets how heavily "
    + "the endgame weighs a loss: at 1 a loss counts the same as a seventh guess, "
    + "and higher settings make the solver spend turns on insurance "
    + "against unlikely answers, losing less at some cost in mean guesses. "
    + "<code>--exact-max</code> is the largest set of candidates the exact search will "
    + "take on in place of the estimate.<br><br>"
    + "<b>The loss penalty</b> and <b>the search's reach</b> play every word as the "
    + "answer under each of the four prior combinations and average them equally. "
    + "Each point is one setting, at its share of games lost across and its mean "
    + "guesses up, with each loss counted as seven guesses. <b>How much is "
    + "searched</b> and <b>the policies</b> play every word under the default prior; "
    + "each row is a way of playing, and each panel is one measure of it on its own "
    + "axis. The default setting is ringed throughout.");
}

function drawRegret() {
  const panels = (node, text) => columnPanels(node, text, {
    rows: "candidates left", skip: ["of which ties"],
    headings: {
      "positions": "Positions checked",
      "estimator differs": "Share of positions where the estimator would pick another "
        + "guess than the exact search",
      "estimator regret": "Guesses lost per position by taking the estimator's pick",
    },
    split: { "estimator differs": { column: "of which ties",
                                    part: "another guess worth exactly as much",
                                    rest: "a worse guess" } },
  });
  tablePanels("exact_decisions", [
    ["blended.table", "the four priors together", panels],
    ["shipped.table", "the shipped prior alone", panels],
    ["who.table", "who decides each turn", stacked],
  ], { controls: "regret-controls", panels: "regret-panels", note: "regret-note" },
  "In positions where the exact search picks the guess, this asks what would happen "
    + "if the estimator <code>f</code> had picked instead: how often it would choose a "
    + "different guess, and how many guesses that would lose. Each row is a group of "
    + "positions by how many candidate answers are left.",
  "With three or more turns left and few enough candidates, the exact search picks "
    + "the guess and its value is the true one. That makes these positions the one "
    + "place the estimate can be checked against the truth. The positions come from "
    + "simulated games, weighted by how often play reaches them.<br><br>"
    + "<b>Positions checked</b> is how many were scored. The middle panel is how often "
    + "the estimator's pick is not the search's, split into picks that are a "
    + "different word of exactly the same true value, which cost nothing, and picks "
    + "that are worse. The last panel is how many guesses per position the "
    + "estimator's pick costs on average, measured with the true values. The first "
    + "view pools positions from games under each of the four prior combinations; "
    + "the second uses the shipped prior alone. <b>Who decides each turn</b> counts "
    + "every decision in a batch of simulated games by the number of turns left, split "
    + "by which part of the solver made it: <code>f</code>, the exact search, the "
    + "closed form that solves one or two turns or candidates outright, or a single "
    + "candidate left to guess.");
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
  // a link to every section, named by its heading
  $("toc").innerHTML = [...document.querySelectorAll("section")].map((s) =>
    `<a href="#${s.id}">${s.querySelector("h2").innerHTML}</a>`).join("");
  drawCharts();
  drawTables();
  // the charts are drawn at their containers' widths, so a resize redraws them
  let pending = null;
  window.addEventListener("resize", () => {
    clearTimeout(pending);
    pending = setTimeout(drawCharts, 150);
  });
}

function drawCharts() {
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
}

boot().catch((error) => {
  $("lede").textContent = `Could not load /api/dev: ${error}`;
});
