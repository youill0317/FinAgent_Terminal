(() => {
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pad2 = (n) => String(n).padStart(2, "0");
const num = (v, d) => v.toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});

const KEY_HELP = {
  FRED: {env: "FRED_API_KEY", url: "https://fredaccount.stlouisfed.org/apikeys"},
  ECOS: {env: "ECOS_API_KEY", url: "https://ecos.bok.or.kr/api/#/AuthKeyApply"},
};
const TAPE_LABELS = {us_fftu: "연준 상단", us_10y: "UST 10Y", us_10y3m: "10Y−3M", us_dollar: "달러 광의",
  wti: "WTI", us_vix: "VIX", kr_usdkrw: "USD/KRW", kr_kospi: "KOSPI"};

function fmt(v, f) {
  if (v === null || v === undefined) return "—";
  const sign = v < 0 ? "−" : (f.signed && v > 0 ? "+" : "");
  return sign + f.pre + num(Math.abs(v), f.dec) + f.suf;
}
function change(row) {
  if (row.value === null || row.prev === null) return {cls: "flat", t: ""};
  const d = row.value - row.prev, dec = row.fmt.dec;
  if (Math.abs(d) < Math.pow(10, -dec) / 2) return {cls: "flat", t: "— 0"};
  return {cls: d > 0 ? "up" : "down", t: `${d > 0 ? "▲" : "▼"} ${num(Math.abs(d), dec)}`};
}
function shortDate(row) {
  const s = row.date || "";
  if (row.freq === "Q") return s.slice(2);
  if (row.freq === "M") return s.slice(2).replace("-", ".");
  return s.slice(5).replace("-", "/");
}

// ---------------------------------------------------------------- charts
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function niceTicks(lo, hi, count) {
  const span = hi - lo || 1, raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let t = Math.floor(lo / step) * step; t <= hi + step * 1e-6; t += step) out.push(+t.toFixed(6));
  return {ticks: out, dec: Math.max(0, -Math.floor(Math.log10(step) + 1e-9))};
}
function lineChart(host, o) {
  host.innerHTML = "";
  const {series, labels, W, H, pad} = o, n = labels.length;
  const all = series.flatMap((s) => s.v.filter((v) => v !== null));
  if (!all.length || n < 2) { host.innerHTML = '<div class="empty">데이터 없음</div>'; return; }
  let lo = Math.min(...all), hi = Math.max(...all);
  if (o.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const {ticks, dec} = niceTicks(lo, hi, 5);
  lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img"}, host);
  const x = (i) => pad.l + i / (n - 1) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo || 1)) * (H - pad.t - pad.b);
  (o.bands || []).forEach(([a, b]) => el("rect", {x: x(a) - 1, y: pad.t, width: Math.max(3, x(b) - x(a) + 2), height: H - pad.t - pad.b, fill: "var(--band)"}, svg));
  const g = el("g", {class: "gl"}, svg);
  ticks.forEach((t) => {
    el("line", {x1: pad.l, x2: W - pad.r, y1: y(t), y2: y(t)}, g);
    el("text", {x: pad.l - 6, y: y(t) + 3.5, "text-anchor": "end", class: "ax"}, svg).textContent = t.toFixed(dec);
  });
  if (o.zero) el("line", {x1: pad.l, x2: W - pad.r, y1: y(0), y2: y(0), class: "zero"}, svg);
  o.xIdx.forEach((i) => { el("text", {x: x(i), y: H - pad.b + 14, "text-anchor": "middle", class: "ax"}, svg).textContent = o.xFmt(labels[i]); });
  series.forEach((s) => {
    let d = "", pen = false;
    s.v.forEach((v, i) => { if (v === null) { pen = false; return; } d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`; pen = true; });
    if (s.area) {
      const base = y(Math.max(lo, 0));
      el("path", {d: `${d}L${x(n - 1)},${base}L${x(0)},${base}Z`, fill: s.color, "fill-opacity": 0.12}, svg);
    }
    el("path", {d, fill: "none", stroke: s.color, "stroke-width": s.w || 2, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round"}, svg);
    const last = s.v[n - 1];
    if (s.end && last !== null) {
      el("circle", {cx: x(n - 1), cy: y(last), r: 3.5, fill: s.color, stroke: "var(--panel)", "stroke-width": 2}, svg);
      if (s.label) el("text", {x: x(n - 1) + 7, y: y(last) + 3.5, class: "ax", style: "fill:var(--amber)"}, svg).textContent = s.label;
    }
  });
  const cross = el("line", {y1: pad.t, y2: H - pad.b, stroke: "var(--ink-2)", "stroke-dasharray": "2 3", visibility: "hidden"}, svg);
  const dots = series.map((s) => el("circle", {r: 3.5, fill: s.color, stroke: "var(--panel)", "stroke-width": 2, visibility: "hidden"}, svg));
  const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; host.appendChild(tip);
  const hit = el("rect", {x: pad.l - 8, y: 0, width: W - pad.l - pad.r + 16, height: H, fill: "transparent", style: "cursor:crosshair"}, svg);
  const move = (ev) => {
    const b = svg.getBoundingClientRect(), px = (ev.clientX - b.left) * W / b.width;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - pad.l) / (W - pad.l - pad.r) * (n - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    const ys = [];
    series.forEach((s, k) => {
      const v = s.v[i];
      if (v === null) { dots[k].setAttribute("visibility", "hidden"); return; }
      dots[k].setAttribute("cx", x(i)); dots[k].setAttribute("cy", y(v)); dots[k].setAttribute("visibility", "visible"); ys.push(y(v));
    });
    tip.innerHTML = o.tip(i);
    tip.style.left = (x(i) / W * 100) + "%";
    tip.style.top = ((ys.length ? Math.min(...ys) : pad.t) / H * b.height) + "px";
    tip.hidden = false;
  };
  hit.addEventListener("pointermove", move); hit.addEventListener("pointerdown", move);
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); tip.hidden = true; });
}
function spark(host, row) {
  host.innerHTML = "";
  const h = row.history, n = h.length;
  if (n < 2) return;
  const vals = h.map((p) => p[1]), W = 70, H = 20, lo = Math.min(...vals), hi = Math.max(...vals), span = (hi - lo) || 1;
  const x = (i) => 1 + i / (n - 1) * (W - 5), y = (v) => 2 + (1 - (v - lo) / span) * (H - 4);
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true"}, host);
  el("path", {d: vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(""), fill: "none", stroke: "var(--amber)", "stroke-width": 1.3, "stroke-linejoin": "round"}, svg);
  el("circle", {cx: x(n - 1), cy: y(vals[n - 1]), r: 2.2, fill: "var(--yellow)"}, svg);
  const dot = el("circle", {r: 2.2, fill: "var(--ink)", visibility: "hidden"}, svg);
  const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; host.appendChild(tip);
  svg.addEventListener("pointermove", (ev) => {
    const b = svg.getBoundingClientRect();
    const i = Math.max(0, Math.min(n - 1, Math.round(((ev.clientX - b.left) / b.width * W - 1) / (W - 5) * (n - 1))));
    dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(vals[i])); dot.setAttribute("visibility", "visible");
    tip.textContent = `${h[i][0]}  ${fmt(vals[i], row.fmt)}`;
    tip.style.left = (x(i) / W * 100) + "%"; tip.style.top = (y(vals[i]) / H * b.height) + "px"; tip.hidden = false;
  });
  svg.addEventListener("pointerleave", () => { dot.setAttribute("visibility", "hidden"); tip.hidden = true; });
}

// ---------------------------------------------------------------- render
let data = null, spreadRange = 0;

function renderStatus(d) {
  const when = new Date(d.generated_at);
  const stamp = `${when.getFullYear()}-${pad2(when.getMonth() + 1)}-${pad2(when.getDate())} ${pad2(when.getHours())}:${pad2(when.getMinutes())}:${pad2(when.getSeconds())}`;
  const total = d.sources.reduce((a, s) => a + s.ok, 0);
  $("status").innerHTML = d.sources.map((s) =>
    `<span class="src" data-s="${esc(s.state)}" title="${esc(s.message || "")}"><i aria-hidden="true"></i><b>${esc(s.name)}</b>${s.ok}/${s.total}</span>`).join("")
    + `<span class="mem">메모리 ${total}개 시리즈 · 디스크 저장 0 · ${d.elapsed_seconds.toFixed(1)}s · 갱신 <span style="color:var(--amber)">${stamp}</span></span>`;
}
function renderNotices(d) {
  const out = [];
  if (d.demo) out.push('<div class="notice">데모 모드: 수치는 모두 예시 값입니다. 실제 데이터는 <code>finagent-macro</code>를 키와 함께 실행하세요.</div>');
  d.sources.filter((s) => s.state === "missing_key").forEach((s) => {
    const k = KEY_HELP[s.id];
    if (k) out.push(`<div class="notice">${esc(s.name)} 키가 없어 ${s.total}개 시리즈를 받지 못했습니다. 무료 키를 <a href="${k.url}" target="_blank" rel="noopener">여기서</a> 발급받아 <code>${k.env}</code> 환경변수로 설정한 뒤 다시 실행하세요.</div>`);
  });
  d.sources.filter((s) => s.state === "error" || s.state === "partial").forEach((s) => {
    out.push(`<div class="notice err">${esc(s.name)}: ${s.total - s.ok}개 시리즈를 받지 못했습니다. ${esc(s.message || "")}</div>`);
  });
  $("notices").innerHTML = out.join("");
}
function rowsById(d) {
  const m = {};
  d.panels.forEach((p) => p.rows.forEach((r) => { m[r.id] = r; }));
  return m;
}
function renderTape(d) {
  const rows = rowsById(d);
  $("tape").innerHTML = d.tape.map((id) => {
    const r = rows[id];
    if (!r) return "";
    const c = change(r);
    return `<div class="tk" title="${esc(r.code)}"><span class="c">${esc(TAPE_LABELS[id] || r.label)}</span><span class="v">${fmt(r.value, r.fmt)}</span><span class="d ${c.cls}">${c.t}</span></div>`;
  }).join("");
}
function renderRegime(d) {
  const icon = {good: "●", warn: "▲", crit: "■", neutral: "◆"};
  $("regime").innerHTML = d.regime.map((g) =>
    `<div class="rg"><div class="rg-t"><span>${esc(g.label)}</span><span class="tag" data-t="${esc(g.tone)}">${icon[g.tone] || ""} ${esc(g.tag)}</span></div>`
    + `<div class="rg-v">${esc(g.value)}<small>${esc(g.unit)}</small></div><div class="rg-r">${esc(g.rule)}</div></div>`).join("");
}
const CURVE_STYLE = [{color: "var(--amber)", w: 2.5, end: true}, {color: "var(--cyan)", w: 1.5}, {color: "var(--muted)", w: 1.5, dash: "4 4"}];
function renderCurve(d) {
  const c = d.curve;
  if (!c) { $("ycLg").innerHTML = ""; $("yc").innerHTML = '<div class="empty">FRED 키가 있어야 표시됩니다</div>'; return; }
  const series = c.lines.map((l, i) => ({...CURVE_STYLE[i], v: l.values, name: `${l.label} ${l.date.slice(2)}`}));
  $("ycLg").innerHTML = series.map((s) => `<span><i style="border-color:${s.color};${s.dash ? "border-top-style:dashed" : ""}"></i>${esc(s.name)}</span>`).join("");
  lineChart($("yc"), {series, labels: c.tenors, W: 500, H: 230, pad: {l: 32, r: 12, t: 10, b: 22},
    xIdx: c.tenors.map((_, i) => i), xFmt: (l) => l,
    tip: (i) => `<b style="color:var(--amber)">${esc(c.tenors[i])}</b><br>` + series.map((s) =>
      `<span class="sw" style="background:${s.color}"></span>${esc(s.name)} ${s.v[i] === null ? "—" : s.v[i].toFixed(2) + "%"}`).join("<br>")});
}
function renderSpread(d) {
  const s = d.spread;
  if (!s || s.points.length < 2) { $("sp").innerHTML = '<div class="empty">FRED 키가 있어야 표시됩니다</div>'; return; }
  const pts = spreadRange ? s.points.slice(-spreadRange) : s.points;
  const labels = pts.map((p) => p[0]), v = pts.map((p) => p[1]);
  const idx = (m) => labels.indexOf(m);
  const bands = s.recessions.map(([a, b]) => [idx(a), idx(b)]).filter(([a, b]) => a >= 0 || b >= 0).map(([a, b]) => [Math.max(a, 0), b < 0 ? labels.length - 1 : b]);
  const yearStep = labels.length > 72 ? 2 : 1;
  const ticks = [];
  labels.forEach((l, i) => {
    if (labels.length <= 24 ? /-(01|07)$/.test(l) : (l.endsWith("-01") && Number(l.slice(0, 4)) % yearStep === 0)) ticks.push(i);
  });
  const last = v[v.length - 1];
  lineChart($("sp"), {series: [{v, color: "var(--amber)", end: true, area: true, label: (last > 0 ? "+" : "") + last.toFixed(2)}],
    labels, W: 680, H: 230, pad: {l: 32, r: 42, t: 10, b: 22}, zero: true, bands,
    xIdx: ticks, xFmt: (l) => labels.length <= 24 ? l.slice(2).replace("-", ".") : l.slice(0, 4),
    tip: (i) => `<b style="color:var(--amber)">${esc(labels[i])}</b><br>${v[i] > 0 ? "+" : ""}${v[i].toFixed(2)}%p${v[i] < 0 ? " · 역전" : ""}`});
}
function renderPanels(d) {
  const host = $("panels");
  host.innerHTML = d.panels.map((p) => `<article class="pn" id="${esc(p.id)}">
    <div class="pn-h"><b><span>${esc(p.key)}</span>${esc(p.title)}</b><small>${esc(p.note)}</small></div>
    <div class="tw"><table class="t"><thead><tr><th>지표</th><th>최신</th><th>변화</th><th>추이</th><th>기준</th></tr></thead><tbody>
    ${p.rows.map((r) => {
      const code = `<code>${esc(r.code)}${r.copyright ? " · © " + esc(r.copyright) : ""}</code>`;
      const name = `<td class="n"><div>${esc(r.label)}</div>${code}</td>`;
      if (r.status !== "ok") {
        const text = r.status === "missing_key" ? "키 없음" : "받지 못함";
        return `<tr>${name}<td class="${r.status === "missing_key" ? "miss" : "err"}" colspan="4" title="${esc(r.message || "")}">${text}</td></tr>`;
      }
      const c = change(r);
      return `<tr data-k="${esc(r.id)}">${name}<td class="l">${fmt(r.value, r.fmt)}</td><td class="${c.cls}">${c.t}</td><td class="sp"><div class="spark"></div></td><td class="dt">${esc(shortDate(r))}</td></tr>`;
    }).join("")}
    </tbody></table></div></article>`).join("") + calendarPanel(d);
  d.panels.forEach((p) => p.rows.forEach((r) => {
    const cell = host.querySelector(`[data-k="${CSS.escape(r.id)}"] .spark`);
    if (cell) spark(cell, r);
  }));
}
function calendarPanel(d) {
  const fredOk = d.sources.some((s) => s.id === "FRED" && s.state !== "missing_key");
  const weekday = ["일", "월", "화", "수", "목", "금", "토"];
  const body = d.calendar.length
    ? `<table class="cal"><tbody>${d.calendar.map((c) => {
        const day = new Date(c.date + "T00:00:00");
        return `<tr><td class="d">${c.date.slice(5).replace("-", "/")} ${weekday[day.getDay()]}</td><td>${esc(c.name)}</td><td class="s">${esc(c.source)}</td></tr>`;
      }).join("")}</tbody></table>`
    : `<div class="cal-empty">${fredOk ? "앞으로 7일 안에 주요 발표가 없습니다." : "FRED 키가 있으면 이번 주 발표 일정이 표시됩니다."}</div>`;
  return `<article class="pn" id="calendar"><div class="pn-h"><b><span>CAL</span>앞으로 7일 발표</b><small>FRED releases/dates</small></div><div class="pn-b">${body}</div></article>`;
}
function render(d) {
  data = d;
  renderStatus(d); renderNotices(d); renderTape(d); renderRegime(d);
  renderCurve(d); renderSpread(d); renderPanels(d);
}

// ---------------------------------------------------------------- load / refresh
const btn = $("refreshBtn"), toast = $("toast");
let busy = false, toastTimer;
function say(msg) { toast.textContent = msg; toast.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { toast.hidden = true; }, 2800); }

async function load(refresh) {
  if (busy) return;
  busy = true; btn.disabled = true;
  if (refresh && data) {
    document.querySelectorAll("#status .src").forEach((s) => { s.dataset.s = "loading"; });
  }
  try {
    const res = await fetch(`/api/dashboard${refresh ? "?refresh=true" : ""}`, {cache: "no-store"});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    render(d);
    if (refresh) {
      document.querySelectorAll("#panels tbody tr").forEach((r) => { r.classList.remove("fresh"); void r.offsetWidth; r.classList.add("fresh"); });
      const ok = d.sources.reduce((a, s) => a + s.ok, 0), total = d.sources.reduce((a, s) => a + s.total, 0);
      say(`${ok}/${total}개 시리즈 갱신 · ${d.elapsed_seconds.toFixed(1)}s`);
    }
  } catch (err) {
    $("notices").innerHTML = `<div class="notice err">대시보드 서버에 연결하지 못했습니다 (${esc(err.message)}). 터미널에서 finagent-macro가 실행 중인지 확인하세요.</div>`;
    if (data) renderStatus(data);
  } finally {
    busy = false; btn.disabled = false;
  }
}

function clocks() {
  const zones = [["NY", "America/New_York"], ["SEOUL", "Asia/Seoul"], ["UTC", "UTC"]];
  $("clocks").innerHTML = zones.map(([k, tz]) => {
    let t;
    try { t = new Date().toLocaleTimeString("en-GB", {timeZone: tz, hour: "2-digit", minute: "2-digit"}); } catch (e) { t = "--:--"; }
    return `<span><b>${k}</b>${t}</span>`;
  }).join("");
}

btn.addEventListener("click", () => load(true));
document.addEventListener("keydown", (e) => {
  if ((e.key === "r" || e.key === "R") && !e.metaKey && !e.ctrlKey && !e.altKey && !/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) load(true);
});
document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  spreadRange = Number(b.dataset.r);
  if (data) renderSpread(data);
}));
const links = [...document.querySelectorAll(".fkeys a")];
links.forEach((a) => a.addEventListener("click", () => links.forEach((x) => x.classList.toggle("on", x === a))));

clocks(); setInterval(clocks, 30000);
load(false);
})();
