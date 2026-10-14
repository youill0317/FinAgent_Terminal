// AI analysis panel: manual runs only, conversations live in this page and in
// the server's memory. Answers stream in as text; when a run finishes the
// server's verification report replaces it, with each number marked
// (matches the data / differs / no cited source) and citations as links.
(() => {
"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const PLACEHOLDER = "{지표}";
const STATUS_TEXT = {ready: "준비됨", missing_key: "키 없음", login_needed: "로그인 필요", not_installed: "설치 안 됨", error: "확인 실패"};
const NUMBER_TEXT = {ok: "데이터와 일치", mismatch: "데이터와 다름", unsupported: "인용한 근거에 없는 숫자"};

const panel = $("analysis"), openBtn = $("aiBtn"), pick = $("aiProvider"), modelIn = $("aiModel");
const input = $("aiInput"), runBtn = $("aiRun"), stopBtn = $("aiStop"), log = $("aiLog");
let meta = null, data = null, preset = "", scope = "view";
let conversation = "", snapshot = "", running = null;

// ---------------------------------------------------------------- open / meta
function setOpen(open) {
  panel.hidden = !open;
  openBtn.setAttribute("aria-expanded", String(open));
  if (open) {
    loadMeta();
    panel.scrollIntoView({block: "start", behavior: "smooth"});
    input.focus({preventScroll: true});
  }
}
async function loadMeta() {
  try {
    const res = await fetch("/api/analysis/meta", {cache: "no-store"});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    meta = await res.json();
  } catch (err) {
    $("aiStatus").textContent = `분석 기능에 연결하지 못했습니다 (${err.message})`;
    $("aiStatus").dataset.s = "error";
    return;
  }
  pick.innerHTML = meta.providers.map((p) =>
    `<option value="${esc(p.id)}">${esc(p.title)} · ${esc(STATUS_TEXT[p.status] || p.status)}</option>`).join("");
  pick.value = meta.choice.provider;
  showProvider();
  if (!$("aiPresets").childElementCount) {
    $("aiPresets").innerHTML = meta.presets.map((p) =>
      `<button type="button" data-preset="${esc(p.id)}" aria-pressed="false" title="입력창에 채우기">${esc(p.title)}</button>`).join("");
  }
}
function provider() { return meta && meta.providers.find((p) => p.id === pick.value); }
function showProvider() {
  const p = provider();
  if (!p) return;
  modelIn.value = p.model;
  modelIn.placeholder = p.default_model;
  const st = $("aiStatus");
  st.dataset.s = p.status;
  st.textContent = `${STATUS_TEXT[p.status] || p.status}${p.message ? " · " + p.message : ""}`;
}
async function saveChoice() {
  const p = provider();
  if (!p) return;
  try {
    const res = await fetch("/api/analysis/choice", {method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({provider: p.id, model: modelIn.value.trim()})});
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
    meta.choice.provider = body.provider;
    p.model = body.models[p.id] || p.default_model;
  } catch (err) {
    $("aiStatus").textContent = `선택을 저장하지 못했습니다: ${err.message}`;
    $("aiStatus").dataset.s = "error";
  }
}

// ---------------------------------------------------------------- presets / scope
function indicators() {
  if (!data) return [];
  return data.panels.flatMap((p) => p.rows.filter((r) => r.status === "ok").map((r) => ({id: r.id, label: r.label, panel: p.title})));
}
function fillIndicators() {
  const rows = indicators(), sel = $("aiIndicator"), keep = sel.value;
  sel.innerHTML = rows.map((r) => `<option value="${esc(r.id)}">${esc(r.panel)} · ${esc(r.label)}</option>`).join("");
  if (keep && rows.some((r) => r.id === keep)) sel.value = keep;
}
function choosePreset(id) {
  const p = meta && meta.presets.find((x) => x.id === id);
  if (!p) return;
  preset = id;
  input.value = p.text;
  document.querySelectorAll("#aiPresets button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.preset === id)));
  const needs = p.text.includes(PLACEHOLDER);
  $("aiPick").hidden = !needs;
  if (needs) fillIndicators();
  input.focus();
}
function clearPreset() {
  preset = "";
  $("aiPick").hidden = true;
  document.querySelectorAll("#aiPresets button").forEach((b) => b.setAttribute("aria-pressed", "false"));
}
function setScope(value) {
  scope = value;
  document.querySelectorAll("#aiScope button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.scope === value)));
}
function lockScope() {
  document.querySelectorAll("#aiScope button").forEach((b) => {
    b.disabled = Boolean(conversation);
    b.title = conversation ? "범위는 새 대화에서 바꿀 수 있습니다" : "";
  });
}
function requestText() {
  let text = input.value.trim();
  if (text.includes(PLACEHOLDER)) {
    const sel = $("aiIndicator"), row = indicators().find((r) => r.id === sel.value);
    if (row) text = text.split(PLACEHOLDER).join(`${row.label} (${row.id})`);
  }
  return text;
}

// ---------------------------------------------------------------- conversation
function newConversation() {
  if (running) return;
  conversation = ""; snapshot = "";
  log.innerHTML = "";
  $("aiStale").hidden = true;
  $("aiConv").textContent = "";
  $("aiSnap").textContent = "수동 실행 · 대화는 메모리에만";
  const recv = $("aiRecv");
  recv.hidden = true; recv.querySelector("ul").innerHTML = "";
  lockScope();
}
function showStale() {
  const box = $("aiStale");
  box.innerHTML = `이 대화는 ${esc(shortTime(snapshot))} 데이터에 고정돼 있습니다. 새로 받은 데이터로 보려면 <button class="more" type="button" id="aiStaleNew">새 대화</button>를 시작하세요.`;
  box.hidden = false;
}
function shortTime(iso) { return String(iso || "").replace("T", " ").slice(0, 16); }

async function run() {
  if (running) return;
  const text = requestText();
  if (!text) { input.focus(); return; }
  const p = provider();
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `<div class="q">${esc(text)}</div><div class="run">준비 중…</div><ul class="tools"></ul><div class="ans live"></div><div class="vsum"></div>`;
  log.append(turn);
  turn.scrollIntoView({block: "nearest"});
  const controller = new AbortController();
  running = controller;
  runBtn.disabled = true; stopBtn.hidden = false; $("aiNew").disabled = true;
  const answer = turn.querySelector(".ans");
  let raw = "", finished = false;
  const body = {request: text, provider: p ? p.id : "", model: modelIn.value.trim(), conversation, preset: conversation ? "" : preset, scope};
  try {
    const res = await fetch("/api/analysis", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body), signal: controller.signal});
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const ev = JSON.parse(line.slice(6));
          if (ev.type === "text") { raw += ev.text; answer.innerHTML = render(liveLines(raw)); }
          else if (ev.type === "done") { finished = true; onDone(turn, ev); }
          else handle(turn, ev);
        }
      }
    }
    if (!finished && !turn.querySelector(".notice")) note(turn, "응답이 끝나기 전에 연결이 닫혔습니다.");
  } catch (err) {
    if (err.name === "AbortError") note(turn, "멈췄습니다. 이 턴은 대화에 남지 않습니다.", true);
    else note(turn, `실행하지 못했습니다: ${err.message}`);
  } finally {
    answer.classList.remove("live");
    if (!answer.textContent.trim()) answer.remove();
    running = null;
    runBtn.disabled = false; stopBtn.hidden = true; $("aiNew").disabled = false;
    if (finished) { input.value = ""; clearPreset(); }
    lockScope();
  }
}
function handle(turn, ev) {
  if (ev.type === "start") {
    conversation = ev.conversation; snapshot = ev.snapshot;
    turn.querySelector(".run").textContent = `${ev.provider_title} · ${ev.model} · ${ev.view} · 데이터 ${shortTime(ev.snapshot)}`;
    $("aiConv").textContent = `대화 ${ev.conversation}`;
    $("aiSnap").textContent = `데이터 ${shortTime(ev.snapshot)}에 고정 · 메모리에만`;
    if (ev.stale) showStale();
    lockScope();
  } else if (ev.type === "tool") {
    const li = document.createElement("li");
    li.textContent = ev.summary;
    if (!ev.ok) li.className = "bad";
    turn.querySelector(".tools").append(li);
  } else if (ev.type === "usage") {
    // One usage event per model round; show the run's totals.
    const run = turn.querySelector(".run");
    const total = (key, add) => { const v = Number(run.dataset[key] || 0) + (add || 0); run.dataset[key] = String(v); return v; };
    const used = `토큰 입력 ${total("in", ev.input).toLocaleString()} · 출력 ${total("out", ev.output).toLocaleString()}${ev.note ? " · " + ev.note : ""}`;
    let span = run.querySelector(".use");
    if (!span) { span = document.createElement("span"); span.className = "use"; run.append(span); }
    span.textContent = ` · ${used}`;
  } else if (ev.type === "error") {
    note(turn, ev.message);
  }
}
function note(turn, message, quiet) {
  const box = document.createElement("div");
  box.className = quiet ? "notice" : "notice err";
  box.textContent = message;
  turn.append(box);
}
function onDone(turn, ev) {
  const answer = turn.querySelector(".ans");
  answer.innerHTML = render(ev.verify.lines);
  answer.classList.remove("live");
  const c = ev.verify.counts;
  const bits = [
    `<span class="ok">일치 <b>${c.ok}</b></span>`,
    `<span class="${c.mismatch ? "bad" : ""}">다름 <b>${c.mismatch}</b></span>`,
    `<span class="${c.unsupported ? "warn" : ""}">근거 없음 <b>${c.unsupported}</b></span>`,
  ];
  if (c.unknown_id) bits.push(`<span class="bad">모르는 근거 <b>${c.unknown_id}</b></span>`);
  bits.push(`<span>도구 ${ev.tool_calls}회</span>`);
  turn.querySelector(".vsum").innerHTML = `숫자 검증 ${bits.join("")}`;
  const recv = $("aiRecv");
  recv.querySelector("ul").innerHTML = ev.received.map((r) => `<li><code>${esc(r.cite)}</code> ${esc(r.info)}</li>`).join("");
  recv.hidden = ev.received.length === 0;
}

// ---------------------------------------------------------------- markdown
// Lines are lists of pieces: {t} text, {n,s,a} a checked number, {c,k} a citation.
function liveLines(text) {
  const shown = text.replace(/\s?\[\[[^\]]*\]\]/g, "").replace(/\s?\[\[[^\]]*$/, "");
  return shown.split("\n").map((line) => [{t: line}]);
}
function inline(html) {
  return html.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
}
function pieceHtml(p) {
  if (p.t !== undefined) return esc(p.t);
  if (p.n !== undefined) {
    const tip = p.s === "mismatch" && p.a ? `${NUMBER_TEXT.mismatch} · 실제 ${p.a}` : (NUMBER_TEXT[p.s] || "");
    return `<span class="nb" data-s="${esc(p.s)}" title="${esc(tip)}">${esc(p.n)}</span>`;
  }
  return p.c.map((id, i) => p.k[i]
    ? `<sup class="ct" data-cite="${esc(id)}" title="${esc(id)} · 눌러서 지표로 이동">[${esc(id)}]</sup>`
    : `<sup class="ct unk" title="문서와 받은 데이터에 없는 근거">[${esc(id)}?]</sup>`).join("");
}
function render(lines) {
  const rows = lines.map((pieces) => ({raw: pieces.map((p) => p.t ?? p.n ?? "").join(""), html: inline(pieces.map(pieceHtml).join(""))}));
  const out = [];
  let para = [], list = null, table = null;
  const flush = () => {
    if (para.length) { out.push(`<p>${para.join("<br>")}</p>`); para = []; }
    if (list) { out.push(`<${list.tag}>${list.items.map((i) => `<li>${i}</li>`).join("")}</${list.tag}>`); list = null; }
    if (table) { out.push(tableHtml(table)); table = null; }
  };
  for (const {raw, html} of rows) {
    const trimmed = raw.trim();
    if (trimmed.startsWith("|")) {
      if (!table) { flush(); table = []; }
      if (!/^\|[\s:|-]+\|?$/.test(trimmed)) table.push(cells(html));
      continue;
    }
    if (table) flush();
    let m;
    if (!trimmed) { flush(); continue; }
    if ((m = /^\s*(#{1,4})\s+/.exec(raw))) {
      flush();
      const tag = m[1].length <= 2 ? "h3" : "h4";
      out.push(`<${tag}>${html.replace(/^\s*#{1,4}\s+/, "")}</${tag}>`);
      continue;
    }
    const bullet = /^\s*[-*•]\s+/.test(raw), ordered = /^\s*\d+[.)]\s+/.test(raw);
    if (bullet || ordered) {
      const tag = ordered ? "ol" : "ul";
      if (para.length || (list && list.tag !== tag)) flush();
      if (!list) list = {tag, items: []};
      list.items.push(html.replace(/^\s*(?:[-*•]|\d+[.)])\s+/, ""));
      continue;
    }
    if (list) flush();
    para.push(html);
  }
  flush();
  return out.join("");
}
function cells(html) {
  return html.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}
function tableHtml(rows) {
  if (!rows.length) return "";
  const [head, ...body] = rows;
  return `<div class="tw"><table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${
    body.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

// ---------------------------------------------------------------- events
openBtn.addEventListener("click", () => setOpen(panel.hidden));
$("aiClose").addEventListener("click", () => setOpen(false));
pick.addEventListener("change", () => { showProvider(); saveChoice(); });
modelIn.addEventListener("change", saveChoice);
$("aiPresets").addEventListener("click", (e) => { const b = e.target.closest("button[data-preset]"); if (b) choosePreset(b.dataset.preset); });
$("aiScope").addEventListener("click", (e) => { const b = e.target.closest("button[data-scope]"); if (b && !b.disabled) setScope(b.dataset.scope); });
input.addEventListener("input", () => { if (!input.value.trim()) clearPreset(); });
input.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); } });
runBtn.addEventListener("click", run);
stopBtn.addEventListener("click", () => { if (running) running.abort(); });
$("aiNew").addEventListener("click", newConversation);
$("aiStale").addEventListener("click", (e) => { if (e.target.id === "aiStaleNew") newConversation(); });
log.addEventListener("click", (e) => {
  const sup = e.target.closest(".ct[data-cite]");
  if (!sup) return;
  const row = document.querySelector(`#panels [data-k="${CSS.escape(sup.dataset.cite)}"]`);
  if (!row) return;
  row.scrollIntoView({block: "center", behavior: "smooth"});
  row.classList.remove("cited"); void row.offsetWidth; row.classList.add("cited");
});
document.addEventListener("finagent:data", (e) => {
  data = e.detail;
  if (!$("aiPick").hidden) fillIndicators();
});
document.addEventListener("finagent:refreshed", () => { if (conversation) showStale(); });
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || document.querySelector("dialog[open]") || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
  if (e.key === "a" || e.key === "A") { e.preventDefault(); setOpen(panel.hidden); }
});
lockScope();
})();
