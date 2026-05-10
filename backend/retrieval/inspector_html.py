INSPECTOR_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MITAOE BM25 Retrieval Inspector</title>
  <style>
    :root { color-scheme: light; font-family: Arial, sans-serif; }
    body { margin: 0; background: #f7f7f4; color: #1d1d1b; }
    header { padding: 18px 24px; background: #1f3a5f; color: white; }
    main { max-width: 1180px; margin: 0 auto; padding: 20px; }
    form { display: grid; grid-template-columns: 1fr 90px 110px; gap: 10px; margin-bottom: 18px; }
    input, button { font: inherit; padding: 10px 12px; border: 1px solid #b9b9b0; border-radius: 6px; }
    button { background: #1f3a5f; color: white; cursor: pointer; border-color: #1f3a5f; }
    .result { background: white; border: 1px solid #d9d9d2; border-radius: 8px; margin: 14px 0; padding: 16px; }
    .meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px; }
    .pill { background: #eef1f4; border-radius: 999px; padding: 4px 9px; font-size: 12px; }
    pre { white-space: pre-wrap; line-height: 1.45; font-size: 14px; }
    details { margin-top: 10px; }
    .muted { color: #5c5c55; }
    a { color: #174b7a; }
  </style>
</head>
<body>
  <header><h1>MITAOE BM25 Retrieval Inspector</h1></header>
  <main>
    <form id="searchForm">
      <input id="query" placeholder="Try: What is MCA eligibility?" required />
      <input id="topK" type="number" value="5" min="1" max="20" />
      <button>Search</button>
    </form>
    <div id="status" class="muted"></div>
    <section id="results"></section>
  </main>
  <script>
    const form = document.getElementById('searchForm');
    const results = document.getElementById('results');
    const status = document.getElementById('status');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const query = document.getElementById('query').value;
      const topK = document.getElementById('topK').value;
      status.textContent = 'Searching...';
      results.innerHTML = '';
      const response = await fetch(`/retrieval/search?query=${encodeURIComponent(query)}&top_k=${topK}`);
      const data = await response.json();
      status.textContent = `${data.results.length} results for "${data.query}"`;
      results.innerHTML = data.results.map((item) => `
        <article class="result">
          <h2>#${item.rank} ${escapeHtml(item.title)}</h2>
          <a href="${item.url}" target="_blank">${item.url}</a>
          <div class="meta">
            <span class="pill">BM25 ${item.score}</span>
            <span class="pill">${item.page_type}</span>
            <span class="pill">${item.content_type}</span>
            <span class="pill">${item.token_count} tokens</span>
            <span class="pill">quality ${item.quality_score}</span>
            <span class="pill">${escapeHtml(item.section_heading)}</span>
          </div>
          <pre>${escapeHtml(item.text)}</pre>
          <details>
            <summary>Metadata / warnings</summary>
            <pre>${escapeHtml(JSON.stringify(item.metadata, null, 2))}</pre>
          </details>
        </article>
      `).join('');
    });
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
    }
  </script>
</body>
</html>
"""

