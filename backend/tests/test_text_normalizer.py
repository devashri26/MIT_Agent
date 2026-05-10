from backend.ingestion.normalizers.text_normalizer import TextNormalizer


def test_text_normalizer_collapses_whitespace_and_unicode() -> None:
    text = "Admissions\u00a0 \tOpen\r\n\r\n\r\nApply   now"

    assert TextNormalizer().normalize(text) == "Admissions Open\n\nApply now"

