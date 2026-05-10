from io import StringIO

from backend.ingestion.loaders.csv_loader import CsvLoader


def test_csv_loader_maps_expected_columns() -> None:
    csv_data = StringIO("url,title,html,text\nhttps://example.edu,Admissions,<h1>Apply</h1>,Fallback\n")
    events = list(CsvLoader(chunk_size=1).iter_rows(csv_data))

    assert len(events) == 1
    assert events[0].row is not None
    assert events[0].row.url == "https://example.edu"
    assert events[0].row.title == "Admissions"


def test_csv_loader_tolerates_missing_optional_columns() -> None:
    csv_data = StringIO("crawl/loadedUrl,markdown\nhttps://example.edu/page,Only text\n")
    events = list(CsvLoader(chunk_size=1).iter_rows(csv_data))

    assert events[0].row is not None
    assert events[0].row.url == "https://example.edu/page"
    assert events[0].row.text == "Only text"


def test_csv_loader_reports_unusable_rows() -> None:
    csv_data = StringIO("title,html\nNo URL,<p>content</p>\n")
    events = list(CsvLoader(chunk_size=1).iter_rows(csv_data))

    assert events[0].is_failed
    assert events[0].error == "missing source url"

