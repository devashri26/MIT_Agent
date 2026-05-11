INSPECTOR_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MITAOE Routed Retrieval Inspector</title>
  <style>
    :root { color-scheme: light; font-family: Arial, sans-serif; }
    body { margin: 0; background: #f7f7f4; color: #1d1d1b; }
    header { padding: 18px 24px; background: #1f3a5f; color: white; }
    header h1 { margin: 0 0 4px 0; font-size: 18px; }
    header .sub { font-size: 12px; opacity: 0.8; }
    main { max-width: 1200px; margin: 0 auto; padding: 20px; }
    form { display: grid; grid-template-columns: 1fr 120px 90px 110px; gap: 10px; margin-bottom: 14px; }
    input, button { font: inherit; padding: 10px 12px; border: 1px solid #b9b9b0; border-radius: 6px; }
    button { background: #1f3a5f; color: white; cursor: pointer; border-color: #1f3a5f; }
    .route { background: white; border: 1px solid #d9d9d2; border-radius: 8px; padding: 12px 14px; margin-bottom: 16px; font-size: 13px; }
    .route .row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
    .result { background: white; border: 1px solid #d9d9d2; border-radius: 8px; margin: 14px 0; padding: 16px; }
    .result h2 { margin: 0 0 6px 0; font-size: 15px; }
    .meta { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 12px; }
    .pill { background: #eef1f4; border-radius: 999px; padding: 3px 9px; font-size: 12px; }
    .pill.score { background: #1f3a5f; color: white; }
    .pill.priority { background: #2e7d32; color: white; }
    .pill.flag { background: #c0392b; color: white; }
    .pill.boost { background: #f39c12; color: white; }
    .pill.match { background: #8e44ad; color: white; }
    .pill.fallback { background: #c0392b; color: white; }
    .pill.component { background: #6c5ce7; color: white; }
    .pill.contamination { background: #e84393; color: white; }
    .pill.mixed { background: #d35400; color: white; }
    .pill.source { background: #2c3e50; color: white; }
    .pill.rerank { background: #16a085; color: white; }
    .pill.answer { background: #27ae60; color: white; }
    .pill.reject { background: #95a5a6; color: white; }
    .pill.abstain { background: #c0392b; color: white; }
    .pill.confidence { background: #2980b9; color: white; }
    .breadcrumb { font-size: 12px; color: #5c5c55; margin-bottom: 6px; }
    .breadcrumb b { color: #1f3a5f; }
    select { font: inherit; padding: 10px 12px; border: 1px solid #b9b9b0; border-radius: 6px; background: white; }
    .context-panel { background: #fffbe6; border: 1px solid #f1c40f; border-radius: 8px; padding: 14px; margin-top: 18px; }
    .context-panel h3 { margin: 0 0 8px 0; font-size: 14px; }
    .context-block { background: white; border-left: 3px solid #16a085; padding: 8px 12px; margin: 8px 0; font-size: 13px; }
    .rejected-section { margin-top: 14px; }
    .rejected-section summary { cursor: pointer; font-size: 13px; color: #5c5c55; }
    .rejected-item { background: #f4f4f0; border-radius: 6px; padding: 6px 10px; margin: 4px 0; font-size: 12px; }
    .chat-panel { background: #eef7ff; border: 1px solid #3498db; border-radius: 8px; padding: 14px; margin-top: 18px; }
    .chat-panel h3 { margin: 0 0 8px 0; font-size: 14px; }
    .chat-msg { background: white; border-radius: 6px; padding: 8px 12px; margin: 6px 0; font-size: 13px; }
    .chat-msg.user { border-left: 3px solid #3498db; }
    .chat-msg.assistant { border-left: 3px solid #16a085; }
    .chat-msg.abstain { border-left: 3px solid #c0392b; background: #fff5f3; }
    .chat-meta { font-size: 11px; color: #5c5c55; margin-top: 4px; }
    .chat-citations { margin-top: 6px; font-size: 12px; }
    .chat-citations a { display: inline-block; margin-right: 8px; background: #16a085; color: white; padding: 2px 7px; border-radius: 10px; text-decoration: none; }
    pre { white-space: pre-wrap; line-height: 1.45; font-size: 13px; background: #fafaf8; padding: 8px; border-radius: 4px; }
    details { margin-top: 10px; }
    .why { background: #fcfaf3; border-left: 3px solid #f39c12; padding: 10px 12px; margin-top: 10px; font-size: 13px; }
    .why .row { display: flex; flex-wrap: wrap; gap: 10px; }
    .why span { color: #5c5c55; }
    .why b { color: #1d1d1b; }
    .muted { color: #5c5c55; }
    a { color: #174b7a; word-break: break-all; }
  </style>
</head>
<body>
  <header>
    <h1>MITAOE Routed Retrieval Inspector</h1>
    <div class="sub">Intent routing · metadata filter · weighted BM25 ranking · explanations</div>
  </header>
  <main>
    <form id="searchForm">
      <input id="query" placeholder="Try: What is MCA eligibility?" required />
      <select id="mode">
        <option value="reranked" selected>reranked</option>
        <option value="hybrid">hybrid</option>
        <option value="bm25">bm25</option>
        <option value="dense">dense</option>
      </select>
      <input id="topK" type="number" value="5" min="1" max="20" />
      <button>Search</button>
    </form>
    <label class="muted" style="display:block;margin-bottom:10px;font-size:13px;">
      <input id="includeComponents" type="checkbox" /> include reusable components
      &nbsp;&nbsp;
      <button id="buildContextBtn" type="button" style="margin-left:10px;padding:4px 10px;font-size:12px;">Build context</button>
      <button id="chatBtn" type="button" style="margin-left:6px;padding:4px 10px;font-size:12px;">Ask (chat)</button>
      <button id="resetChatBtn" type="button" style="margin-left:6px;padding:4px 10px;font-size:12px;">Reset chat</button>
    </label>
    <div id="route"></div>
    <div id="status" class="muted"></div>
    <div id="chat"></div>
    <div id="context"></div>
    <section id="results"></section>
    <section id="rejected"></section>
  </main>
  <script>
    const form = document.getElementById('searchForm');
    const results = document.getElementById('results');
    const rejectedEl = document.getElementById('rejected');
    const contextEl = document.getElementById('context');
    const chatEl = document.getElementById('chat');
    const status = document.getElementById('status');
    const route = document.getElementById('route');
    const buildContextBtn = document.getElementById('buildContextBtn');
    const chatBtn = document.getElementById('chatBtn');
    const resetChatBtn = document.getElementById('resetChatBtn');
    let chatMessages = [];
    let sessionId = localStorage.getItem('chat_session_id') || null;

    function renderChat() {
      if (!chatMessages.length) { chatEl.innerHTML = ''; return; }
      chatEl.innerHTML = `
        <div class="chat-panel">
          <h3>Chat (session ${sessionId || '—'})</h3>
          ${chatMessages.map(m => {
            if (m.role === 'user') {
              return `<div class="chat-msg user"><b>You:</b> ${escapeHtml(m.content)}</div>`;
            }
            const a = m.answer;
            const abstainClass = a.abstained ? 'abstain' : 'assistant';
            const conf = a.confidence || {};
            const meta = [
              `conf <b>${conf.answer_confidence ?? 0}</b>`,
              `grounding <b>${conf.grounding_confidence ?? 0}</b>`,
              `halluc.risk <b>${conf.hallucination_risk ?? 0}</b>`,
              `${a.provider || '—'}/${a.model || '—'}`,
            ].join(' · ');
            const abstainPill = a.abstained
              ? `<span class="pill abstain">abstained: ${escapeHtml(a.abstention_reason || '')}</span>` : '';
            const rewritePill = m.rewritten_query
              ? `<span class="pill">rewrote → ${escapeHtml(m.rewritten_query)}</span>` : '';
            const cites = (a.citations || []).map((c, i) =>
              `<a href="${escapeHtml(c.source_url)}" target="_blank">[${c.index}] ${escapeHtml((c.title||'').slice(0,30))}</a>`
            ).join('');
            const warnings = (a.grounding_warnings || []).length
              ? `<div class="chat-meta">⚠ ${a.grounding_warnings.join('; ')}</div>` : '';
            const unsupported = (a.hallucination && a.hallucination.unsupported_claims || []).length
              ? `<div class="chat-meta">unsupported: ${(a.hallucination.unsupported_claims).join(' / ')}</div>` : '';
            return `<div class="chat-msg ${abstainClass}">
              <b>MITAOE:</b> ${escapeHtml(a.answer)}
              <div class="chat-meta">${meta} ${abstainPill} ${rewritePill}</div>
              ${cites ? `<div class="chat-citations">${cites}</div>` : ''}
              ${warnings}
              ${unsupported}
            </div>`;
          }).join('')}
        </div>
      `;
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const query = document.getElementById('query').value;
      const topK = document.getElementById('topK').value;
      const mode = document.getElementById('mode').value;
      const includeComponents = document.getElementById('includeComponents').checked;
      status.textContent = 'Searching...';
      results.innerHTML = '';
      rejectedEl.innerHTML = '';
      contextEl.innerHTML = '';
      route.innerHTML = '';
      const endpoint = mode === 'reranked' ? '/retrieval/reranked/search'
                     : mode === 'hybrid' ? '/retrieval/hybrid/search'
                     : mode === 'dense' ? '/retrieval/dense/search'
                     : '/retrieval/search';
      const url = `${endpoint}?query=${encodeURIComponent(query)}&top_k=${topK}&include_components=${includeComponents}`;
      const response = await fetch(url);
      const data = await response.json();
      status.textContent = `${data.results.length} results for "${data.query}"`;

      const fallbackPill = data.filter_fallback_used
        ? '<span class="pill fallback">filter fallback used</span>' : '';
      const componentsPill = data.components_excluded > 0
        ? `<span class="pill component">excluded ${data.components_excluded} components</span>` : '';
      route.innerHTML = `
        <div class="route">
          <div><b>Mode:</b> ${escapeHtml(mode)} &nbsp; <b>Intent:</b> ${escapeHtml(data.intent)} ${fallbackPill} ${componentsPill}</div>
          <div class="row">
            <span class="pill">page types: ${escapeHtml((data.allowed_page_types || []).join(', ') || '—')}</span>
            <span class="pill">section types: ${escapeHtml((data.allowed_section_types || []).join(', ') || '—')}</span>
          </div>
          <div class="row">
            <span class="pill">expanded terms: ${escapeHtml((data.expanded_terms || []).join(', ') || '—')}</span>
          </div>
        </div>
      `;

      results.innerHTML = data.results.map((item) => {
        const exp = item.explanation || {};
        const flagPills = (item.quality_flags || [])
          .map((f) => `<span class="pill flag">${escapeHtml(f)}</span>`)
          .join('');
        const matchedTerms = (exp.matched_terms || []).join(', ') || '—';
        const boost = exp.metadata_boost ? `<span class="pill boost">boost: ${escapeHtml(exp.metadata_boost)}</span>` : '';
        const sectionMatch = exp.section_match ? `<span class="pill match">section: ${escapeHtml(exp.section_match)}</span>` : '';
        const componentPill = item.is_reusable_component
          ? `<span class="pill component">${escapeHtml(item.component_type || 'reusable')}</span>` : '';
        const contaminationPill = item.cross_domain_contamination
          ? `<span class="pill contamination">contaminated: ${escapeHtml((item.contamination_sources || []).join(', '))}</span>` : '';
        const mixedPill = item.mixed_topic
          ? `<span class="pill mixed">mixed: ${escapeHtml((item.dominant_topics || []).join(', '))}</span>` : '';
        const breadcrumb = (item.section_path || []).length
          ? `<div class="breadcrumb">${(item.section_path || []).map((seg, i, arr) => i === arr.length - 1 ? `<b>${escapeHtml(seg)}</b>` : escapeHtml(seg)).join(' › ')}</div>` : '';
        const sources = item.retrieval_source || [];
        const sourcePill = sources.length
          ? `<span class="pill source">src: ${escapeHtml(sources.join('+'))}</span>` : '';
        const rankInfo = (item.bm25_rank || item.dense_rank)
          ? `<span class="pill">bm25_rank ${item.bm25_rank || '—'} · dense_rank ${item.dense_rank || '—'}</span>` : '';
        const rerankPill = (item.rerank_score && item.rerank_score > 0)
          ? `<span class="pill rerank">rerank ${item.rerank_score}</span>` : '';
        const answerPill = (item.answerability_score && item.answerability_score > 0)
          ? `<span class="pill answer">answerability ${item.answerability_score}</span>` : '';
        return `
          <article class="result">
            <h2>#${item.rank} ${escapeHtml(item.title)}</h2>
            ${breadcrumb}
            <a href="${item.url}" target="_blank">${escapeHtml(item.url)}</a>
            <div class="meta">
              <span class="pill score">final ${item.score}</span>
              <span class="pill priority">priority ${item.retrieval_priority}</span>
              <span class="pill">${escapeHtml(item.page_type)}</span>
              <span class="pill">${escapeHtml(item.section_type)}</span>
              <span class="pill">${escapeHtml(item.section_heading)}</span>
              <span class="pill">${item.token_count} tokens</span>
              ${boost}
              ${sectionMatch}
              ${sourcePill}
              ${rankInfo}
              ${rerankPill}
              ${answerPill}
              ${componentPill}
              ${contaminationPill}
              ${mixedPill}
              ${flagPills}
            </div>
            <div class="why">
              <div class="row">
                <div><span>matched terms:</span> <b>${escapeHtml(matchedTerms)}</b></div>
                <div><span>bm25:</span> <b>${exp.bm25_score ?? 0}</b> <span>(norm</span> <b>${exp.bm25_normalized ?? 0}</b><span>)</span></div>
                <div><span>priority:</span> <b>${exp.retrieval_priority ?? 0}</b></div>
                <div><span>section bonus:</span> <b>${exp.section_match_bonus ?? 0}</b></div>
                <div><span>page_type matched intent:</span> <b>${exp.page_type_match ? 'yes' : 'no'}</b></div>
              </div>
            </div>
            <pre>${escapeHtml(item.text)}</pre>
            <details>
              <summary>Raw chunk metadata</summary>
              <pre>${escapeHtml(JSON.stringify(item.metadata, null, 2))}</pre>
            </details>
          </article>
        `;
      }).join('');

      const rejected = data.rejected || [];
      if (rejected.length) {
        rejectedEl.innerHTML = `
          <details class="rejected-section" open>
            <summary>Rejected by reranker (${rejected.length})</summary>
            ${rejected.map(item => `
              <div class="rejected-item">
                #${item.rank || '—'} <b>${escapeHtml(item.title || '')}</b>
                — <span class="pill reject">${escapeHtml(item.rejection_reason || 'rejected')}</span>
                <span class="pill rerank">rerank ${item.rerank_score}</span>
                <span class="pill answer">answerability ${item.answerability_score}</span>
                <span class="pill">${escapeHtml(item.page_type)} / ${escapeHtml(item.section_type)}</span>
              </div>
            `).join('')}
          </details>
        `;
      }
    });

    buildContextBtn.addEventListener('click', async () => {
      const query = document.getElementById('query').value;
      const topK = document.getElementById('topK').value;
      const includeComponents = document.getElementById('includeComponents').checked;
      if (!query) { contextEl.textContent = 'Enter a query first.'; return; }
      contextEl.innerHTML = '<div class="muted">Building context...</div>';
      const response = await fetch('/context/build', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          query, top_k: Number(topK), candidate_pool: 20,
          token_budget: 2000, include_components: includeComponents,
        }),
      });
      const ctx = await response.json();
      const warnings = (ctx.grounding_warnings || []).length
        ? `<div><b>Warnings:</b> ${(ctx.grounding_warnings).join(', ')}</div>` : '';
      const blocks = (ctx.context_blocks || []).map((b, i) => `
        <div class="context-block">
          <b>[${i + 1}] ${escapeHtml(b.title)}</b>
          <div class="breadcrumb">${(b.section_path || []).join(' › ')}</div>
          <div class="muted">${b.token_count} tokens · rerank ${b.rerank_score} · answerability ${b.answerability_score}</div>
          <a href="${b.source_url}" target="_blank">${escapeHtml(b.source_url)}</a>
        </div>
      `).join('');
      const dropped = (ctx.dropped_blocks || []).length
        ? `<details><summary>${ctx.dropped_blocks.length} block(s) dropped during assembly</summary>
             ${ctx.dropped_blocks.map(d => `<div class="rejected-item">${escapeHtml(d.chunk_id)} — ${escapeHtml(d.reason)}</div>`).join('')}
           </details>` : '';
      contextEl.innerHTML = `
        <div class="context-panel">
          <h3>Grounded context (confidence ${ctx.grounding_confidence}, ${ctx.total_tokens}/${ctx.token_budget} tokens, ${ctx.distinct_section_types} section types)</h3>
          ${warnings}
          ${blocks}
          ${dropped}
          <details><summary>Show LLM prompt</summary><pre>${escapeHtml(ctx.prompt)}</pre></details>
        </div>
      `;
    });

    chatBtn.addEventListener('click', async () => {
      const query = document.getElementById('query').value;
      if (!query) { chatEl.innerHTML = '<div class="muted">Enter a query first.</div>'; return; }
      chatMessages.push({ role: 'user', content: query });
      renderChat();
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          query, session_id: sessionId, top_k: 5, candidate_pool: 20,
          token_budget: 2000, include_components: false, run_judge: true,
        }),
      });
      const data = await response.json();
      sessionId = data.session_id;
      localStorage.setItem('chat_session_id', sessionId);
      chatMessages.push({
        role: 'assistant',
        answer: data.answer,
        rewritten_query: data.rewritten_query,
      });
      renderChat();
    });

    resetChatBtn.addEventListener('click', async () => {
      if (sessionId) await fetch(`/conversation/${sessionId}`, { method: 'DELETE' });
      sessionId = null;
      localStorage.removeItem('chat_session_id');
      chatMessages = [];
      renderChat();
    });

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
    }
  </script>
</body>
</html>
"""
