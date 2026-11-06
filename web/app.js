/* Lee Kuan Yew chatbot UI. Plain JS, no framework.
   Talks to GET /health and POST /chat on API_BASE (see config.js).
   Answers are rendered with textContent only, never innerHTML from the
   model. Chat history lives in localStorage, never on the server. */
(function () {
  "use strict";

  var API_BASE = (window.LKY_CONFIG && window.LKY_CONFIG.API_BASE) || "http://localhost:8000";
  var apiOverride = new URLSearchParams(location.search).get("api");
  if (apiOverride) API_BASE = apiOverride.replace(/\/$/, "");

  var HISTORY_KEY = "lky-chat-history-v1";
  var HISTORY_MAX = 50;
  var POST_COOLDOWN_S = 3;

  /* 4 questions from eval/questions.json, each ranked first in
     eval/results.json (first_hit_rank 1). The 5th is deliberately
     out of scope: it shows the honest refusal. */
  var TRY_QUESTIONS = [
    { q: "What did Lee Kuan Yew say about education and nation-building?", tag: "Education" },
    { q: "What was his view on the Hong Kong handover and US-China relations?", tag: "Geopolitics" },
    { q: "What was his economic outlook at the turn of the millennium?", tag: "Economy" },
    { q: "What did he say about his own experience learning languages?", tag: "Language" },
  ];
  var OUT_OF_SCOPE_Q = "What do you think about cryptocurrency?";

  var TOPICS = ["Leadership", "Economy", "Geopolitics", "Education",
                "Society", "History", "National Development",
                "Foreign Relations", "Youth"];

  var FLAG_LABELS = {
    is_draft: "NAS draft",
    is_translation: "English translation (likely delivered in Malay)",
    is_partial_malay: "Partly in Malay",
    is_letter: "Letter, not a speech",
    is_toast: "Short toast, not a full speech",
    is_qa: "Press-conference Q&A transcript",
  };

  var state = {
    server: "unknown",          // unknown | waking | live | down
    lastSuccessAt: 0,
    awakePromise: null,
    inFlight: false,
    cooldownUntil: 0,
    corpus: null,
    health: null,
    msgCount: 0,
  };

  var el = {};
  ["live-pill", "pill-stats", "model-badge", "status-pill", "banner",
   "thread", "empty-state", "chips", "try-list", "composer", "input",
   "counter", "send", "new-chat", "clear-history", "tts-stop",
   "tab-chat", "tab-examples", "tab-corpus", "tab-about"]
    .forEach(function (id) { el[id] = document.getElementById(id); });

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  /* ---------------- collapsible side panels ---------------- */
  var PANEL_KEYS = { sidebar: "lky-sidebar-open", panel: "lky-panel-open" };
  function readPanelOpen(which) {
    try { return localStorage.getItem(PANEL_KEYS[which]) !== "0"; } catch (e) { return true; }
  }
  function applyPanel(which, open) {
    var layout = document.querySelector(".layout");
    var isLeft = which === "sidebar";
    layout.classList.toggle(isLeft ? "no-sidebar" : "no-panel", !open);
    var btn = document.getElementById(isLeft ? "rail-left" : "rail-right");
    btn.classList.toggle("closed", !open);
    btn.textContent = isLeft ? (open ? "‹" : "›") : (open ? "›" : "‹");
    var label = (open ? "Hide " : "Show ") + (isLeft ? "sidebar" : "panel");
    btn.setAttribute("aria-label", label);
    btn.title = label;
    try { localStorage.setItem(PANEL_KEYS[which], open ? "1" : "0"); } catch (e) { /* ignore */ }
  }
  function togglePanel(which) { applyPanel(which, !readPanelOpen(which)); }
  document.getElementById("rail-left").addEventListener("click", function () { togglePanel("sidebar"); });
  document.getElementById("rail-right").addEventListener("click", function () { togglePanel("panel"); });
  applyPanel("sidebar", readPanelOpen("sidebar"));
  applyPanel("panel", readPanelOpen("panel"));

  /* ---------------- text-to-speech (browser speechSynthesis, free) ----------------
     No auto-read: a speaker button on each assistant message reads that
     message aloud on tap; tapping it again (or the topbar Stop button)
     stops the reading. If speechSynthesis is unavailable the buttons
     are never rendered, so nothing errors. */
  var TTS_SUPPORTED = ("speechSynthesis" in window) && ("SpeechSynthesisUtterance" in window);
  var tts = { currentBtn: null, currentMsg: null };

  function cleanSpeechText(answer) {
    return String(answer)
      .replace(/\[S\d+(?:\s*,\s*S\d+)*\]/g, "")  // [S1], [S1, S4]
      .replace(/[*_`#]+/g, "")                    // markdown-ish artifacts
      .replace(/\s+/g, " ")
      .trim();
  }
  function pickEnglishVoice() {
    try {
      var voices = window.speechSynthesis.getVoices() || [];
      var en = voices.filter(function (v) { return v.lang && v.lang.toLowerCase().indexOf("en") === 0; });
      if (!en.length) return null;
      var us = en.filter(function (v) { return v.lang.toLowerCase().indexOf("en-us") === 0; });
      return us[0] || en[0];
    } catch (e) { return null; }
  }
  function ttsResetButton() {
    if (tts.currentBtn) {
      tts.currentBtn.classList.remove("speaking");
      tts.currentBtn.innerHTML = "&#128266; Listen";
      tts.currentBtn.setAttribute("aria-label", "Listen to this answer");
      tts.currentBtn = null;
    }
    tts.currentMsg = null;
    el["tts-stop"].hidden = true;
  }
  function stopSpeech() {
    if (!TTS_SUPPORTED) return;
    try { window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
    ttsResetButton();
  }
  function speakMessage(msgId, answer, btn) {
    if (!TTS_SUPPORTED) return;
    // Tapping the message that is already speaking stops it.
    if (tts.currentMsg === msgId) { stopSpeech(); return; }
    stopSpeech();
    var text = cleanSpeechText(answer);
    if (!text) return;
    var u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    u.rate = 1;
    var voice = pickEnglishVoice();
    if (voice) u.voice = voice;
    u.onend = function () { ttsResetButton(); };
    u.onerror = function () { ttsResetButton(); };
    tts.currentMsg = msgId;
    tts.currentBtn = btn;
    btn.classList.add("speaking");
    btn.innerHTML = "&#9632; Stop";
    btn.setAttribute("aria-label", "Stop reading this answer");
    el["tts-stop"].hidden = false;
    try { window.speechSynthesis.speak(u); } catch (e) { ttsResetButton(); }
  }

  function fmtNum(n) { return Number(n).toLocaleString("en-US"); }
  function fmtTime(ts) {
    var d = new Date(ts);
    return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
  }
  function shortModel(id) {
    if (!id) return "model";
    var known = {
      "google/gemma-4-26b-a4b-it:free": "Gemma 4 26B",
      "google/gemma-4-31b-it:free": "Gemma 4 31B",
      "nvidia/nemotron-3.5-lightning:free": "Nemotron 3.5 Lightning",
      "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "Nemotron 3 Nano",
      "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
      "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
      "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    };
    if (known[id]) return known[id];
    var base = id.split("/").pop().replace(/:free$/, "").replace(/-/g, " ");
    return base.charAt(0).toUpperCase() + base.slice(1);
  }
  function flagKeys(flags) {
    if (!flags) return [];
    if (Array.isArray(flags)) return flags;
    return Object.keys(flags).filter(function (k) { return flags[k]; });
  }

  /* ---------------- tabs ---------------- */
  function setTab(name) {
    var valid = ["chat", "examples", "corpus", "about"];
    if (valid.indexOf(name) < 0) name = "chat";
    $all(".tabs button").forEach(function (b) {
      var on = b.getAttribute("data-tab") === name;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });
    $all(".sidenav button").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-tab") === name);
    });
    ["chat", "examples", "corpus", "about"].forEach(function (t) {
      el["tab-" + t].hidden = (t !== name);
    });
    try { history.replaceState(null, "", "#" + name); } catch (e) { /* ignore */ }
    if (window.innerWidth <= 640) {
      var panel = document.getElementById("panel");
      if (panel) panel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
  $all("[data-tab]").forEach(function (b) {
    b.addEventListener("click", function () { setTab(b.getAttribute("data-tab")); });
  });

  /* ---------------- corpus data ---------------- */
  function loadJSON(path) {
    return fetch(path).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  function loadCorpus() {
    return loadJSON("./corpus.json").then(function (d) {
      state.corpus = d;
      if (d.flag_labels) FLAG_LABELS = d.flag_labels;
      el["pill-stats"].textContent =
        d.stats.speeches + " speeches · " + fmtNum(d.stats.total_words_raw) + " words";
      renderChatTab(d);
      renderCorpusTab(d);
      renderAboutTab(d);
    }).catch(function () {
      el["pill-stats"].textContent = "33 speeches · 59,362 words";
      el["tab-chat"].textContent = "Could not load corpus.json. The chat still works; the corpus panels need the file.";
    });
  }

  function statBox(num, lbl, full) {
    return '<div class="stat' + (full ? " full" : "") + '"><div class="num">' + num +
      '</div><div class="lbl">' + lbl + "</div></div>";
  }

  function renderChatTab(d) {
    var s = d.stats;
    var html = "";
    html += '<div class="banner-img"><img src="./assets/banner.jpg" alt="Marina Bay skyline at dusk" loading="lazy">' +
      "<blockquote>&ldquo;A small island can do great things.&rdquo;<footer>&mdash; LEE KUAN YEW</footer></blockquote></div>";
    html += "<h2>Corpus Overview</h2><div class=\"stat-grid\">" +
      statBox(s.speeches, "Speeches &amp; Articles") +
      statBox(fmtNum(s.total_words_raw), "Total Words") +
      statBox(s.date_min.slice(0, 4) + " &ndash; " + s.date_max.slice(0, 4), "Coverage Period", true) +
      "</div>";
    html += "<h2>Topics Covered</h2><div class=\"chip-row\">" +
      TOPICS.map(function (t) { return '<span class="topic">' + t + "</span>"; }).join("") + "</div>";
    html += "<h2>Corpus Details</h2><dl class=\"kv\">" +
      "<dt>Source</dt><dd>" + d.sources.join(", ") + "</dd>" +
      "<dt>Total Documents</dt><dd>" + s.speeches + "</dd>" +
      "<dt>Total Chunks</dt><dd>" + s.chunks + " (~500 words each)</dd>" +
      "<dt>Embedding Model</dt><dd>BAAI/bge-small-en-v1.5<span class=\"sub\">384 dim, ONNX</span></dd>" +
      "<dt>Retrieval Method</dt><dd>Hybrid (Dense + BM25)<span class=\"sub\">RRF (k=60), top-5</span></dd>" +
      "<dt>Evaluation (" + d.eval.n + " questions)</dt><dd>Recall@3 " + d.eval.recall_at_3.toFixed(2) +
      "<span class=\"sub\">Recall@5 " + d.eval.recall_at_5.toFixed(2) + " &middot; MRR " + d.eval.mrr.toFixed(2) + "</span></dd>" +
      "</dl>";
    html += '<div class="note-box"><h3>&#9432; Important Note</h3><p>This archive contains ' + s.speeches +
      " speeches, not the complete works of Lee Kuan Yew. Notable works such as the &ldquo;Graduate Mothers&rdquo; speech (1983) are only available as audio. Some documents are drafts, translations, or partial transcripts.</p></div>";
    el["tab-chat"].innerHTML = html;
  }

  function renderCorpusTab(d) {
    var html = "<h2>All Speeches (" + d.speeches.length + ")</h2><div class=\"speech-list\">";
    d.speeches.forEach(function (sp) {
      var flags = (sp.flags || []).map(function (f) {
        return '<span class="flag">' + (FLAG_LABELS[f] || f) + "</span>";
      }).join("");
      html += '<div class="speech-card"><div class="t"><a href="' + sp.url +
        '" target="_blank" rel="noopener">' + escapeHtml(sp.title) + "</a></div>" +
        '<div class="s">' + sp.date + " &middot; " + escapeHtml(sp.source) +
        " &middot; " + fmtNum(sp.words_clean) + " words &middot; " + sp.n_chunks + " chunks</div>" +
        (flags ? '<div>' + flags + "</div>" : "") + "</div>";
    });
    el["tab-corpus"].innerHTML = html + "</div>";
  }

  function renderAboutTab(d) {
    var s = d.stats;
    var html = "<h2>About the Corpus</h2><div class=\"stat-grid\">" +
      statBox(s.speeches, "Speeches") +
      statBox(fmtNum(s.total_words_raw), "Total Words") +
      statBox(s.date_min.slice(0, 4) + " &ndash; " + s.date_max.slice(0, 4), "Coverage Period") +
      statBox(s.chunks, "Chunks") + "</div>";
    html += "<h2>Speeches per Decade</h2><div class=\"decades\">";
    var max = Math.max.apply(null, d.decades.map(function (x) { return x.count; }));
    d.decades.forEach(function (x) {
      html += '<div class="row"><span>' + x.decade + "</span>" +
        '<span class="bar"><i style="width:' + Math.round(x.count / max * 100) + '%"></i></span>' +
        "<span>" + x.count + "</span></div>";
    });
    html += "</div>";
    html += "<h2>What This Is / What It Is Not</h2><p class=\"small\">Public speeches from " +
      d.sources.join(", ") + ". Not the 19-volume memoirs (copyrighted). Some famous speeches exist only as audio and are not included.</p>";
    html += "<h2>Metadata Flags</h2><p class=\"small\">One speech each is: " +
      Object.keys(d.flag_counts).map(function (k) {
        return (FLAG_LABELS[k] || k) + " (" + d.flag_counts[k] + ")";
      }).join(", ") + ".</p>";
    html += "<h2>Retrieval Evaluation</h2><table class=\"eval-table\"><tr><th>Metric</th><th>Score</th></tr>" +
      "<tr><td>Questions (n)</td><td>" + d.eval.n + "</td></tr>" +
      "<tr><td>Recall@3</td><td>" + d.eval.recall_at_3.toFixed(2) + "</td></tr>" +
      "<tr><td>Recall@5</td><td>" + d.eval.recall_at_5.toFixed(2) + "</td></tr>" +
      "<tr><td>MRR</td><td>" + d.eval.mrr.toFixed(2) + "</td></tr></table>" +
      "<p class=\"small muted\">Known weak spots, reported as-is: the youth-policy question ranks the 2003 youth speech outside the top 5 (outranked by a 1959 teachers&rsquo; rally speech), and the 1965 separation question ranks the expected speech 4th.</p>";
    html += "<h2>Pipeline</h2><p class=\"small\">Embedding &rarr; hybrid search (dense + BM25) &rarr; RRF fusion (k=60) &rarr; top-5 passages &rarr; model fallback chain.</p>";
    html += "<h2>Model Chain &amp; Cost</h2><table class=\"chain-table\"><tr><th>Model</th><th>Price / 1M tokens</th><th>Per answer</th></tr>";
    d.model_chain.forEach(function (m) {
      html += "<tr><td><span class=\"mono\">" + escapeHtml(m.model) + "</span><br><span class=\"muted\">" +
        m.tier + " &middot; " + m.provider + "</span></td><td>Rp " + fmtNum(m.price_in_idr_per_million) +
        " in / Rp " + fmtNum(m.price_out_idr_per_million) + " out</td><td>&asymp; Rp " + m.est_cost_idr_per_answer + "</td></tr>";
    });
    html += "</table>";
    html += "<h2>Word Counts, Honestly</h2><p class=\"small\">" + fmtNum(s.total_words_raw) +
      " words in the original speech texts; " + fmtNum(s.total_words_clean) +
      " after cleaning; " + fmtNum(s.total_chunk_words) +
      " counting chunk overlap (~80 words between neighbouring chunks).</p>";
    el["tab-about"].innerHTML = html;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  /* ---------------- examples ---------------- */
  function loadExamples() {
    loadJSON("./chats.json").then(function (d) {
      var items = d.items || d.chats || [];
      renderExamples(items, d);
    }).catch(function () {
      el["tab-examples"].innerHTML = '<div class="example-empty">Could not load example chats.</div>';
    });
  }

  function renderExamples(items, d) {
    if (!items || items.length === 0) {
      el["tab-examples"].innerHTML = '<div class="example-empty">No example chats generated yet. ' +
        "Example chats are produced by running the real system (retrieval + generation) and are added here with their sources &mdash; nothing is invented.</div>";
      return;
    }
    var html = "<h2>Real Example Conversations</h2>" +
      '<p class="small muted">These are real outputs from the live pipeline, saved so they stay available while the server sleeps. Not edited.</p>';
    el["tab-examples"].innerHTML = html;
    items.forEach(function (it) {
      var card = document.createElement("div");
      card.className = "example-card";
      var q = document.createElement("div");
      q.className = "q";
      q.textContent = "Q: " + it.question;
      card.appendChild(q);
      card.appendChild(renderAssistant(it.answer, it.sources || [], it.meta || {},
        it.generated_at || d.generated_at, true));
      el["tab-examples"].appendChild(card);
    });
  }

  /* ---------------- try-this questions / chips ---------------- */
  function renderTry() {
    TRY_QUESTIONS.forEach(function (item) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "try-card";
      var span = document.createElement("span");
      span.textContent = item.q;
      var arrow = document.createElement("span");
      arrow.className = "arrow";
      arrow.textContent = "›";
      b.appendChild(span);
      b.appendChild(arrow);
      b.addEventListener("click", function () { send(item.q); });
      el["try-list"].appendChild(b);
    });
    var out = document.createElement("button");
    out.type = "button";
    out.className = "try-card out";
    var span = document.createElement("span");
    span.textContent = OUT_OF_SCOPE_Q;
    var tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "Something he never covered";
    span.appendChild(tag);
    var arrow = document.createElement("span");
    arrow.className = "arrow";
    arrow.textContent = "›";
    out.appendChild(span);
    out.appendChild(arrow);
    out.addEventListener("click", function () { send(OUT_OF_SCOPE_Q); });
    el["try-list"].appendChild(out);

    TRY_QUESTIONS.forEach(function (item) {
      var c = document.createElement("button");
      c.type = "button";
      c.className = "chip";
      c.textContent = item.q;
      c.addEventListener("click", function () { send(item.q); });
      el["chips"].appendChild(c);
    });
    var oc = document.createElement("button");
    oc.type = "button";
    oc.className = "chip out";
    oc.textContent = OUT_OF_SCOPE_Q + "  ·  Something he never covered";
    oc.addEventListener("click", function () { send(OUT_OF_SCOPE_Q); });
    el["chips"].appendChild(oc);
  }

  /* ---------------- server status ---------------- */
  function setStatus(s) {
    state.server = s;
    var pill = el["status-pill"];
    if (s === "unknown") { pill.hidden = true; pill.textContent = ""; return; }
    pill.hidden = false;
    if (s === "live") { pill.className = "status-pill live"; pill.textContent = "● Live"; }
    if (s === "waking") { pill.className = "status-pill waking"; pill.textContent = "Waking up… (~1 min)"; }
    if (s === "down") { pill.className = "status-pill down"; pill.textContent = "● Unreachable"; }
  }

  function showBanner(kind, html) {
    el.banner.hidden = false;
    el.banner.className = "banner " + kind;
    el.banner.innerHTML = html;
  }
  function hideBanner() { el.banner.hidden = true; el.banner.innerHTML = ""; }

  function pingHealthOnce(timeoutMs) {
    var ctrl = new AbortController();
    var t = setTimeout(function () { ctrl.abort(); }, timeoutMs);
    return fetch(API_BASE + "/health", { signal: ctrl.signal })
      .then(function (r) {
        clearTimeout(t);
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      }).catch(function (e) {
        clearTimeout(t);
        throw e;
      });
  }

  function pingHealth() {
    var started = Date.now();
    var slowTimer = setTimeout(function () {
      if (state.server === "unknown") {
        setStatus("waking");
        showBanner("warn", "The server runs on free hosting and sleeps when idle. It&rsquo;s waking up now; your question will be sent automatically.");
      }
    }, 2000);
    var attempt = function () {
      return pingHealthOnce(20000).then(function (h) {
        clearTimeout(slowTimer);
        state.health = h;
        state.lastSuccessAt = Date.now();
        setStatus("live");
        hideBanner();
        if (h.free_model) {
          el["model-badge"].title = "Primary model: " + h.free_model +
            ". If it is unavailable, a low-cost backup model answers instead.";
        }
        return true;
      }).catch(function () {
        if (Date.now() - started > 150000) {
          clearTimeout(slowTimer);
          setStatus("down");
          showBanner("err", "Couldn't reach the server. You can still read real example conversations. " +
            '<button type="button" class="btn" id="banner-examples">See examples</button> ' +
            '<button type="button" class="btn" id="banner-retry">Try again</button>');
          var be = document.getElementById("banner-examples");
          if (be) be.addEventListener("click", function () { setTab("examples"); });
          var br = document.getElementById("banner-retry");
          if (br) br.addEventListener("click", function () { hideBanner(); setStatus("unknown"); pingHealth(); });
          return false;
        }
        return new Promise(function (res) { setTimeout(res, 4000); }).then(attempt);
      });
    };
    return attempt();
  }

  function ensureAwake() {
    if (state.server === "live" && Date.now() - state.lastSuccessAt < 14 * 60 * 1000) {
      return Promise.resolve(true);
    }
    if (state.awakePromise) return state.awakePromise;
    state.awakePromise = pingHealth().then(function (ok) {
      state.awakePromise = null;
      return ok;
    });
    return state.awakePromise;
  }

  /* ---------------- history ---------------- */
  function loadHistory() {
    try {
      var raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) { return []; }
  }
  function saveHistory(items) {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(-HISTORY_MAX))); } catch (e) { /* ignore */ }
  }
  function pushHistory(entry) {
    var items = loadHistory();
    items.push(entry);
    saveHistory(items);
  }
  function clearHistory() {
    stopSpeech();
    try { localStorage.removeItem(HISTORY_KEY); } catch (e) { /* ignore */ }
    el.thread.innerHTML = "";
    el.thread.appendChild(el["empty-state"]);
    el["empty-state"].style.display = "";
  }
  function hideEmptyState() { el["empty-state"].style.display = "none"; }

  function restoreHistory() {
    var items = loadHistory();
    if (!items.length) return;
    hideEmptyState();
    items.forEach(function (it) {
      el.thread.appendChild(renderUserBubble(it.question, it.ts));
      el.thread.appendChild(renderAssistant(it.answer, it.sources || [], it.meta || {}, it.ts, false));
    });
    el.thread.scrollTop = el.thread.scrollHeight;
  }

  /* ---------------- rendering ---------------- */
  function renderUserBubble(question, ts) {
    var wrap = document.createElement("div");
    wrap.className = "msg user-msg";
    var b = document.createElement("div");
    b.className = "user-bubble";
    b.textContent = question;
    var t = document.createElement("div");
    t.className = "time";
    t.textContent = fmtTime(ts || Date.now());
    wrap.appendChild(b);
    wrap.appendChild(t);
    return wrap;
  }

  function attemptsTooltip(meta) {
    var attempts = (meta && meta.attempts) || [];
    if (!attempts.length) return "";
    var errWord = { rate_limited: "busy", timeout: "busy", model_unavailable: "unavailable" };
    var parts = attempts.map(function (a) {
      var s = shortModel(a.model) + " (" + (a.ok ? "answered" : (errWord[a.error] || "failed")) + ")";
      return s;
    });
    return "Tried: " + parts.join(" → ");
  }

  function renderBadge(meta) {
    var badge = document.createElement("span");
    var tier = meta.tier || "free";
    if (tier === "paid" && meta.fallback_used) {
      badge.className = "badge paid";
      badge.textContent = "Backup model · " + shortModel(meta.model) + " (paid)";
    } else if (tier === "paid") {
      badge.className = "badge paid";
      badge.textContent = "Paid model · " + shortModel(meta.model);
    } else {
      badge.className = "badge free";
      badge.textContent = "Free model · " + shortModel(meta.model);
    }
    var tip = attemptsTooltip(meta);
    if (tip) badge.title = tip;
    return badge;
  }

  function renderAnswerText(container, answer, sources) {
    var paras = String(answer).split(/\n\s*\n/);
    paras.forEach(function (para) {
      para = para.trim();
      if (!para) return;
      // A paragraph that opens with a quotation is a quoted passage.
      // The closing quote may sit before a trailing citation, so look
      // for any second quote mark, not only one at the very end.
      var isQuote = para.charAt(0) === '"' && para.indexOf('"', 1) > 0;
      var node = document.createElement(isQuote ? "blockquote" : "p");
      if (isQuote) {
        para = para.slice(1);
        var lastQ = para.lastIndexOf('"');
        if (lastQ >= 0) para = para.slice(0, lastQ) + para.slice(lastQ + 1);
      }
      // Split on [S1]..[S5], including groups like [S1, S4] (the real
      // model output uses both). Only numbers within the source range
      // become buttons; anything else stays plain text.
      var parts = para.split(/(\[S\d+(?:\s*,\s*S\d+)*\])/g);
      parts.forEach(function (part) {
        var m = part.match(/^\[S([\d,\s]+)\]$/);
        if (!m) { node.appendChild(document.createTextNode(part)); return; }
        var nums = m[1].split(",").map(function (x) { return parseInt(x.trim(), 10); });
        var made = 0;
        nums.forEach(function (n) {
          if (n >= 1 && n <= sources.length) {
            if (made > 0) node.appendChild(document.createTextNode(", "));
            node.appendChild(makeCiteButton(n));
            made++;
          }
        });
        if (!made) node.appendChild(document.createTextNode(part));
      });
      container.appendChild(node);
    });
  }

  function makeCiteButton(n) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "cite";
    btn.textContent = n;
    btn.title = "See source " + n;
    btn.setAttribute("aria-label", "See source " + n);
    btn.addEventListener("click", function () { openSource(btn, n); });
    return btn;
  }

  function openSource(btn, n) {
    var details = btn.closest(".msg").querySelector("details.sources");
    if (!details) return;
    details.open = true;
    var li = details.querySelector("#src-" + state.msgCount + "-" + n) ||
             details.querySelector("li[data-rank='" + n + "']");
    if (li) {
      li.scrollIntoView({ behavior: "smooth", block: "center" });
      li.classList.add("flash");
      setTimeout(function () { li.classList.remove("flash"); }, 1500);
    }
  }

  function renderSources(sources, msgId, open) {
    var details = document.createElement("details");
    details.className = "sources";
    if (open) details.open = true;
    var summary = document.createElement("summary");
    var s1 = document.createElement("span");
    s1.textContent = "Sources (" + sources.length + ")";
    var s2 = document.createElement("span");
    s2.className = "view-all";
    s2.textContent = "View all →";
    summary.appendChild(s1);
    summary.appendChild(s2);
    details.appendChild(summary);
    var ol = document.createElement("ol");
    sources.forEach(function (src, i) {
      var li = document.createElement("li");
      li.setAttribute("data-rank", String(i + 1));
      li.id = "src-" + msgId + "-" + (i + 1);
      var rank = document.createElement("span");
      rank.className = "rank";
      rank.textContent = "#" + (i + 1);
      var main = document.createElement("div");
      main.className = "src-main";
      var title = document.createElement("div");
      title.className = "src-title";
      if (src.url) {
        var a = document.createElement("a");
        a.href = src.url;
        a.target = "_blank";
        a.rel = "noopener";
        a.textContent = src.title;
        title.appendChild(a);
      } else {
        title.textContent = src.title;
      }
      var sub = document.createElement("div");
      sub.className = "src-sub";
      sub.textContent = (src.date || "") + " · " + (src.source || "");
      main.appendChild(title);
      main.appendChild(sub);
      flagKeys(src.flags).forEach(function (f) {
        var fl = document.createElement("span");
        fl.className = "flag";
        fl.textContent = FLAG_LABELS[f] || f;
        main.appendChild(fl);
      });
      var score = document.createElement("span");
      score.className = "src-score";
      var sc = (typeof src.score === "number") ? src.score : null;
      score.textContent = sc === null ? "" : sc.toFixed(4);
      score.title = sc === null ? "RRF score not reported"
        : "RRF score " + sc.toFixed(4) + " (scale 0–0.033, not a percentage)";
      li.appendChild(rank);
      li.appendChild(main);
      li.appendChild(score);
      ol.appendChild(li);
    });
    details.appendChild(ol);
    return details;
  }

  function renderAssistant(answer, sources, meta, ts, isExample) {
    var msgId = ++state.msgCount;
    var wrap = document.createElement("div");
    wrap.className = "msg asst-msg";
    var row = document.createElement("div");
    row.className = "asst";
    var img = document.createElement("img");
    img.className = "avatar";
    img.src = "./assets/portrait-avatar.jpg";
    img.alt = "Lee Kuan Yew";
    var body = document.createElement("div");
    body.className = "body";
    var who = document.createElement("div");
    who.className = "who";
    var name = document.createElement("span");
    name.textContent = "Lee Kuan Yew";
    var tag = document.createElement("span");
    tag.className = "rag-tag";
    tag.textContent = "RAG";
    who.appendChild(name);
    who.appendChild(tag);
    body.appendChild(who);
    var ans = document.createElement("div");
    ans.className = "answer";
    if (answer === null || answer === undefined) {
      var p = document.createElement("p");
      p.textContent = "I couldn't generate an answer right now. These are the speech passages I retrieved for your question:";
      ans.appendChild(p);
    } else {
      renderAnswerText(ans, answer, sources || []);
    }
    body.appendChild(ans);
    var metaLine = document.createElement("div");
    metaLine.className = "meta-line";
    // Separator only between items, never leading (the badge is absent
    // when generation failed, so a hard-coded "· " would lead the line).
    function sep() {
      if (metaLine.childNodes.length) {
        var s = document.createElement("span");
        s.textContent = " · ";
        metaLine.appendChild(s);
      }
    }
    if (meta && (meta.provider || meta.model)) {
      metaLine.appendChild(renderBadge(meta));
      if (meta.fallback_used && meta.tier === "paid") {
        var fn = document.createElement("div");
        fn.className = "fallback-note";
        fn.textContent = "The free model was unavailable, so a low-cost backup model answered.";
        body.appendChild(fn);
      }
    }
    if (sources && sources.length) {
      sep();
      var dot = document.createElement("span");
      dot.textContent = sources.length + " sources";
      metaLine.appendChild(dot);
    }
    if (meta && meta.latency_ms && meta.latency_ms.total) {
      sep();
      var lat = document.createElement("span");
      lat.textContent = "answered in " + (meta.latency_ms.total / 1000).toFixed(1) + " s";
      metaLine.appendChild(lat);
    }
    if (meta && typeof meta.est_cost_idr === "number" && meta.est_cost_idr > 0) {
      sep();
      var cost = document.createElement("span");
      cost.textContent = "≈ Rp " + meta.est_cost_idr;
      metaLine.appendChild(cost);
    }
    // Speaker button: tap to hear this answer read aloud, tap again to
    // stop. Only when the browser has speechSynthesis; never auto-reads.
    if (TTS_SUPPORTED && typeof answer === "string" && answer) {
      if (metaLine.childNodes.length) sep();
      var listen = document.createElement("button");
      listen.type = "button";
      listen.className = "listen-btn";
      listen.innerHTML = "&#128266; Listen";
      listen.setAttribute("aria-label", "Listen to this answer");
      (function (mid, txt, b) {
        b.addEventListener("click", function () { speakMessage(mid, txt, b); });
      })(msgId, answer, listen);
      metaLine.appendChild(listen);
    }
    if (metaLine.childNodes.length) body.appendChild(metaLine);
    if (sources && sources.length) {
      body.appendChild(renderSources(sources, msgId, answer === null || answer === undefined));
    }
    var t = document.createElement("div");
    t.className = "time";
    t.textContent = fmtTime(ts || Date.now());
    body.appendChild(t);
    row.appendChild(img);
    row.appendChild(body);
    wrap.appendChild(row);
    return wrap;
  }

  function showErrorBox(text, actions) {
    var wrap = document.createElement("div");
    wrap.className = "msg asst-msg";
    var box = document.createElement("div");
    box.className = "error-box";
    box.textContent = text;
    if (actions && actions.length) {
      var row = document.createElement("div");
      row.className = "actions";
      actions.forEach(function (a) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "btn" + (a.primary ? " primary" : "");
        b.textContent = a.label;
        b.addEventListener("click", a.fn);
        row.appendChild(b);
      });
      box.appendChild(row);
    }
    wrap.appendChild(box);
    el.thread.appendChild(wrap);
    el.thread.scrollTop = el.thread.scrollHeight;
  }

  /* ---------------- composer ---------------- */
  function setComposerEnabled(on) {
    el.input.disabled = !on;
    el.send.disabled = !on;
  }

  function lockComposerFor(seconds, label) {
    var until = Date.now() + seconds * 1000;
    state.cooldownUntil = until;
    setComposerEnabled(false);
    el.counter.hidden = false;
    var tick = function () {
      var left = Math.ceil((until - Date.now()) / 1000);
      if (left <= 0) {
        el.counter.hidden = true;
        el.counter.textContent = "";
        setComposerEnabled(true);
        el.input.focus();
        return;
      }
      el.counter.textContent = (label || "Ask again in") + " " + left + "s";
      setTimeout(tick, 250);
    };
    tick();
  }

  function autogrow() {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, 110) + "px";
    var len = el.input.value.length;
    if (len > 400) {
      el.counter.hidden = false;
      el.counter.textContent = len + "/500";
    } else if (Date.now() >= state.cooldownUntil) {
      el.counter.hidden = true;
      el.counter.textContent = "";
    }
  }

  function send(question) {
    var q = (question !== undefined ? question : el.input.value).trim();
    if (state.inFlight) return;
    if (q.length < 3 || q.length > 500) {
      showErrorBox("Please ask a question between 3 and 500 characters.");
      return;
    }
    if (Date.now() < state.cooldownUntil) return;
    state.inFlight = true;
    setComposerEnabled(false);
    hideEmptyState();
    el.input.value = "";
    autogrow();
    el.thread.appendChild(renderUserBubble(q, Date.now()));
    var pending = document.createElement("div");
    pending.className = "msg asst-msg";
    var prow = document.createElement("div");
    prow.className = "asst";
    var pimg = document.createElement("img");
    pimg.className = "avatar";
    pimg.src = "./assets/portrait-avatar.jpg";
    pimg.alt = "Lee Kuan Yew";
    var pbody = document.createElement("div");
    pbody.className = "body";
    var who = document.createElement("div");
    who.className = "who";
    who.textContent = "Lee Kuan Yew";
    var ptext = document.createElement("div");
    ptext.className = "pending";
    pbody.appendChild(who);
    pbody.appendChild(ptext);
    prow.appendChild(pimg);
    prow.appendChild(pbody);
    pending.appendChild(prow);
    el.thread.appendChild(pending);
    el.thread.scrollTop = el.thread.scrollHeight;

    var t0 = Date.now();
    var timer = setInterval(function () {
      var n = Math.floor((Date.now() - t0) / 1000);
      ptext.innerHTML = "";
      var label = state.server === "live"
        ? "Searching the speeches"
        : "Waking the server (free hosting, about a minute)";
      ptext.textContent = label + "… " + n + "s";
    }, 1000);
    ptext.textContent = state.server === "live"
      ? "Searching the speeches… 0s"
      : "Waking the server (free hosting, about a minute)… 0s";

    var finish = function () {
      clearInterval(timer);
      state.inFlight = false;
      lockComposerFor(POST_COOLDOWN_S, "Ask again in");
    };

    var doPost = function () {
      var ctrl = new AbortController();
      var to = setTimeout(function () { ctrl.abort(); }, 100000);
      return fetch(API_BASE + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, k: 5 }),
        signal: ctrl.signal,
      }).then(function (r) {
        clearTimeout(to);
        return r.json().then(function (d) { return { status: r.status, data: d }; });
      }).catch(function (e) {
        clearTimeout(to);
        throw e;
      });
    };

    ensureAwake().then(function (awake) {
      if (!awake) {
        pending.remove();
        finish();
        showErrorBox("Couldn't reach the server. It may still be waking up.", [
          { label: "Try again", primary: true, fn: function () { send(q); } },
          { label: "See examples", fn: function () { setTab("examples"); } },
        ]);
        return;
      }
      return doPost().then(function (res) {
        pending.remove();
        if (res.status === 200) {
          var d = res.data;
          el.thread.appendChild(renderAssistant(
            d.answer === null ? null : d.answer, d.sources || [], d.meta || {}, Date.now(), false));
          pushHistory({ question: q, answer: d.answer, sources: d.sources || [], meta: d.meta || {}, ts: Date.now() });
          el.thread.scrollTop = el.thread.scrollHeight;
          finish();
        } else if (res.status === 429) {
          var ra = (res.data.detail && res.data.detail.retry_after) || 60;
          finish();
          showErrorBox("You're asking quickly. Please try again in " + ra + " s.");
          lockComposerFor(ra, "Ask again in");
        } else if (res.status === 422) {
          finish();
          showErrorBox("Please ask a question between 3 and 500 characters.");
        } else {
          finish();
          showErrorBox("Something went wrong on the server. Please try again.");
        }
      }).catch(function (e) {
        if (e && e.name === "AbortError") {
          pending.remove();
          finish();
          showErrorBox("The request timed out. The server may still be waking up.", [
            { label: "Try again", primary: true, fn: function () { send(q); } },
            { label: "See examples", fn: function () { setTab("examples"); } },
          ]);
          return;
        }
        // Network / CORS failure: re-check health once, then retry once.
        setStatus("unknown");
        return ensureAwake().then(function (ok) {
          if (!ok) {
            pending.remove();
            finish();
            showErrorBox("Couldn't reach the server. It may still be waking up.", [
              { label: "Try again", primary: true, fn: function () { send(q); } },
              { label: "See examples", fn: function () { setTab("examples"); } },
            ]);
            return;
          }
          return doPost().then(function (res) {
            pending.remove();
            if (res.status === 200) {
              var d = res.data;
              el.thread.appendChild(renderAssistant(
                d.answer === null ? null : d.answer, d.sources || [], d.meta || {}, Date.now(), false));
              pushHistory({ question: q, answer: d.answer, sources: d.sources || [], meta: d.meta || {}, ts: Date.now() });
              el.thread.scrollTop = el.thread.scrollHeight;
            } else {
              showErrorBox("Something went wrong on the server. Please try again.");
            }
            finish();
          }).catch(function () {
            pending.remove();
            finish();
            showErrorBox("Couldn't reach the server. It may still be waking up.", [
              { label: "Try again", primary: true, fn: function () { send(q); } },
              { label: "See examples", fn: function () { setTab("examples"); } },
            ]);
          });
        });
      });
    });
  }

  el.composer.addEventListener("submit", function (ev) {
    ev.preventDefault();
    send();
  });
  el.input.addEventListener("input", autogrow);
  el.input.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      send();
    }
  });
  el["new-chat"].addEventListener("click", clearHistory);
  el["clear-history"].addEventListener("click", clearHistory);
  if (TTS_SUPPORTED) el["tts-stop"].addEventListener("click", stopSpeech);

  /* ---------------- init ---------------- */
  renderTry();
  restoreHistory();
  var hash = (location.hash || "#chat").slice(1);
  setTab(hash);
  loadCorpus();
  loadExamples();
  pingHealth();
})();
