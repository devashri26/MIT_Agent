from backend.ingestion.cleaners.html_cleaner import HtmlCleaner


def test_html_cleaner_removes_boilerplate_and_preserves_content() -> None:
    html = """
    <html>
      <body>
        <nav>Home About Admissions</nav>
        <main>
          <h1>Admissions</h1>
          <p>Applications are open.</p>
          <ul><li>Submit marksheet</li></ul>
        </main>
        <footer>Copyright</footer>
        <script>alert("x")</script>
      </body>
    </html>
    """

    cleaned, malformed = HtmlCleaner().clean(html)

    assert "Admissions" in cleaned
    assert "Applications are open" in cleaned
    assert "Submit marksheet" in cleaned
    assert "Copyright" not in cleaned
    assert "alert" not in cleaned
    assert malformed is False

