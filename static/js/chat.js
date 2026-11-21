// VerifAI conversational Ask page
(function () {
  const thread = document.getElementById('chatThread');
  const form = document.getElementById('chatForm');
  const input = document.getElementById('chatInput');
  const sendBtn = document.getElementById('chatSend');
  const newChatBtn = document.getElementById('newChatBtn');
  if (!thread || !form || !input) return;

  const readJSON = (id, fallback) => {
    try { return JSON.parse(document.getElementById(id).textContent); } catch (e) { return fallback; }
  };
  const history = readJSON('chatData', []);
  const fullName = readJSON('chatUser', '') || '';
  const HONORIFICS = ['dr', 'dr.', 'mr', 'mr.', 'mrs', 'mrs.', 'ms', 'ms.', 'miss', 'prof', 'prof.', 'sir', 'madam'];
  const nameParts = fullName.split(/\s+/).filter(Boolean);
  const firstName = (nameParts.filter(p => !HONORIFICS.includes(p.toLowerCase()))[0]) || nameParts[0] || '';
  let busy = false;

  // ---------- safe, minimal markdown ----------
  function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function inline(s) {
    return s
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, '$1<em>$2</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\[(\d{1,2})\]/g, '<sup class="chat-cite">$1</sup>');
  }
  function renderMarkdown(text) {
    const lines = escapeHTML(text || '').split(/\r?\n/);
    let html = '', list = null, para = [];
    const flushPara = () => { if (para.length) { html += '<p>' + inline(para.join('<br>')) + '</p>'; para = []; } };
    const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
    for (const raw of lines) {
      const line = raw.trim();
      const ul = line.match(/^[-*•]\s+(.*)$/);
      const ol = line.match(/^\d+[.)]\s+(.*)$/);
      const h = line.match(/^#{1,4}\s+(.*)$/);
      if (ul || ol) {
        flushPara();
        const tag = ul ? 'ul' : 'ol';
        if (list !== tag) { closeList(); html += `<${tag}>`; list = tag; }
        html += '<li>' + inline((ul || ol)[1]) + '</li>';
      } else if (h) {
        flushPara(); closeList();
        html += '<p class="chat-subhead">' + inline(h[1]) + '</p>';
      } else if (!line) {
        flushPara(); closeList();
      } else {
        closeList(); para.push(line);
      }
    }
    flushPara(); closeList();
    return html;
  }

  // ---------- rendering ----------
  function timeGreeting() {
    const h = new Date().getHours();
    return h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening';
  }

  function scrollToBottom() { thread.scrollTop = thread.scrollHeight; }

  function bubble(role, html, extraClass) {
    const row = document.createElement('div');
    row.className = 'chat-row ' + (role === 'user' ? 'from-user' : 'from-bot') + (extraClass ? ' ' + extraClass : '');
    if (role !== 'user') {
      const av = document.createElement('div');
      av.className = 'chat-avatar';
      av.innerHTML = '<img src="/static/img/verifai-mark.svg" alt="">';
      row.appendChild(av);
    }
    const b = document.createElement('div');
    b.className = 'chat-bubble';
    b.innerHTML = html;
    row.appendChild(b);
    thread.appendChild(row);
    scrollToBottom();
    return b;
  }

  function statusBadge(status) {
    const map = {
      'Verified': 'badge-verified', 'Partially Verified': 'badge-partial',
      'Conflicting Evidence': 'badge-conflicting', 'General Knowledge': 'chat-badge-knowledge',
    };
    return `<span class="${map[status] || 'badge-insufficient'}">${escapeHTML(status)}</span>`;
  }

  function renderBotMessage(m) {
    const meta = m.meta || {};
    let html = '';
    if (meta.interpreted_as) {
      html += `<div class="chat-interpreted"><i class="fas fa-link"></i> Following up on: <em>${escapeHTML(meta.interpreted_as)}</em></div>`;
    }
    html += `<div class="chat-text">${renderMarkdown(m.content)}</div>`;

    if (m.kind === 'answer' || m.kind === 'knowledge') {
      html += '<div class="chat-meta">';
      if (meta.status) html += statusBadge(meta.status);
      if (m.kind === 'answer' && meta.confidence) {
        html += `<span class="chat-confidence">Confidence ${(meta.confidence * 100).toFixed(0)}%</span>`;
      }
      if (m.kind === 'knowledge') {
        html += '<span class="chat-knowledge-note"><i class="fas fa-circle-info"></i> General knowledge, not from indexed sources</span>';
      }
      html += '</div>';

      if (meta.sources && meta.sources.length) {
        html += '<div class="chat-sources"><div class="chat-sources-title">Sources</div><ol>';
        for (const s of meta.sources) {
          const label = escapeHTML(s.title || 'Source') +
            (s.journal ? ` <span class="chat-source-journal">· ${escapeHTML(s.journal)}${s.year ? ' ' + escapeHTML(s.year) : ''}</span>` : '');
          const url = /^(https?:\/\/|\/)/.test(s.url || '') ? s.url : '';
          const ext = url.startsWith('http') ? ' target="_blank" rel="noopener"' : '';
          html += '<li>' + (url ? `<a href="${escapeHTML(url)}"${ext}>${label}</a>` : label) + '</li>';
        }
        html += '</ol></div>';
      }
      if (m.session_id) {
        html += `<div class="chat-actions">
          <a href="/user/comparison/${m.session_id}"><i class="fas fa-chart-simple"></i> Answer report</a>
          <a href="/user/proposed-result/${m.session_id}"><i class="fas fa-shield-halved"></i> Evidence &amp; verification</a>
          <a href="/user/feedback/${m.session_id}"><i class="fas fa-star"></i> Rate answer</a>
        </div>`;
      }
    }
    const extra = (m.kind === 'out_of_scope' || m.kind === 'not_covered') ? 'is-out-of-scope' : (m.kind === 'error' || m.kind === 'unavailable') ? 'is-error' : '';
    bubble('assistant', html, extra);
  }

  function renderUserMessage(text) {
    bubble('user', `<div class="chat-text">${renderMarkdown(text)}</div>`);
  }

  function renderWelcome() {
    const name = firstName ? `, ${escapeHTML(firstName)}` : '';
    bubble('assistant', `<div class="chat-text">
      <p><strong>${timeGreeting()}${name}.</strong> I'm <strong>VerifAI</strong>, your evidence-grounded clinical assistant.</p>
      <p>Ask me about diseases, symptoms, diagnosis, treatments, drugs or genetics. I'll answer from indexed medical
      literature and show you the evidence. I only answer from VerifAI's knowledge base and your uploaded datasets, and I'll tell you when something isn't covered.</p>
      <p>Feel free to ask follow-ups, too. I'll remember what we've been talking about. What would you like to know?</p>
    </div>`, 'is-welcome');
  }

  // ---------- typing indicator ----------
  // The label shows real progress sent by the server ("Searching…", "Writing…", "Verifying…").
  function showTyping() {
    const b = bubble('assistant', '<div class="chat-typing"><span></span><span></span><span></span></div><div class="chat-typing-text">Understanding your message…</div>', 'is-typing');
    const row = b.closest('.chat-row');
    return () => { if (row && row.isConnected) row.remove(); };
  }

  // ---------- sending ----------
  function setBusy(v) {
    busy = v;
    input.disabled = v;
    sendBtn.disabled = v;
    if (!v) input.focus();
  }

  async function send(text) {
    text = (text || '').trim();
    if (!text || busy) return;
    renderUserMessage(text);
    input.value = '';
    autoSize();
    setBusy(true);
    const stopTyping = showTyping();
    let live = null, liveText = '', pending = false, done = false;
    const typingLabel = () => thread.querySelector('.is-typing .chat-typing-text');

    function paintLive() {
      pending = false;
      if (live) { live.innerHTML = `<div class="chat-text">${renderMarkdown(liveText)}<span class="chat-caret"></span></div>`; scrollToBottom(); }
    }
    function handle(evt) {
      if (evt.type === 'status') {
        const el = typingLabel();
        if (el) el.textContent = evt.text;
      } else if (evt.type === 'token') {
        if (!live) { stopTyping(); live = bubble('assistant', '', 'is-streaming'); }
        liveText += evt.text;
        if (!pending) { pending = true; requestAnimationFrame(paintLive); }
      } else if (evt.type === 'final') {
        done = true; stopTyping();
        if (live) live.closest('.chat-row').remove();
        renderBotMessage(evt.message);
      } else if (evt.type === 'error') {
        done = true; stopTyping();
        if (live) live.closest('.chat-row').remove();
        renderBotMessage({ kind: 'error', content: evt.text });
      }
    }

    try {
      const res = await fetch('/user/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        handle({ type: 'error', text: data.error || 'Sorry, I hit a problem answering that. Please try again.' });
        return;
      }
      // Newline-delimited JSON events: status -> tokens -> final
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done: finished } = await reader.read();
        if (finished) break;
        buffer += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (line) { try { handle(JSON.parse(line)); } catch (e) { /* ignore partial line */ } }
        }
      }
      if (buffer.trim()) { try { handle(JSON.parse(buffer)); } catch (e) { /* ignore */ } }
      if (!done) handle({ type: 'error', text: 'The answer was interrupted. Please send your message again.' });
    } catch (e) {
      if (!done) handle({ type: 'error', text: 'The connection was interrupted before I could answer (the server may have restarted). Please send your message again.' });
    } finally {
      setBusy(false);
    }
  }

  function autoSize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  }

  form.addEventListener('submit', e => { e.preventDefault(); send(input.value); });
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input.value); }
  });
  input.addEventListener('input', autoSize);

  document.querySelectorAll('.chat-example').forEach(btn =>
    btn.addEventListener('click', () => send(btn.dataset.q)));

  if (newChatBtn) {
    newChatBtn.addEventListener('click', async () => {
      if (busy) return;
      try { await fetch('/user/chat/new', { method: 'POST' }); } catch (e) { /* ignore */ }
      thread.innerHTML = '';
      renderWelcome();
      input.focus();
    });
  }

  // ---------- initial state ----------
  renderWelcome();
  for (const m of history) {
    if (m.role === 'user') renderUserMessage(m.content);
    else renderBotMessage(m);
  }
  autoSize();
  input.focus();
  // Arriving from a suggested question (?q=...): ask it straight away.
  if (input.value.trim()) {
    const q = input.value;
    if (window.history && history.replaceState) history.replaceState(null, '', window.location.pathname);
    send(q);
  }
})();
