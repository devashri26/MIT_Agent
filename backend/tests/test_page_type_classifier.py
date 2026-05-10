from backend.ingestion.classifiers.page_type_classifier import PageTypeClassifier


def test_page_type_classifier_prefers_url_rules() -> None:
    classifier = PageTypeClassifier()

    assert classifier.classify("https://example.edu/blog/article", "Exam Tips", [], "exam")[0] == "Blog"
    assert classifier.classify("https://example.edu/about", "Faculty Books", [], "faculty")[0] == "Faculty"
    assert classifier.classify("https://example.edu/computer-engineering", "Placements", [], "placements")[0] == "Department"


def test_page_type_classifier_detects_directory_index() -> None:
    classifier = PageTypeClassifier()

    page_type, confidence, warnings = classifier.classify(
        "https://example.edu/files/",
        "Index of /files",
        [],
        "Index of /files Name Last modified Size Description",
    )

    assert page_type == "DirectoryIndex"
    assert confidence == 1.0
    assert warnings == []
