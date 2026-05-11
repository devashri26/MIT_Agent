from backend.normalization.boilerplate_registry import (
    ReusableComponentRegistry,
    fingerprint_paragraph,
    split_paragraphs,
)
from backend.normalization.component_detector import build_registry
from backend.normalization.widget_suppressor import (
    assess_chunk_components,
    classify_component_type,
    classify_registry,
)


def test_fingerprint_stable_across_whitespace() -> None:
    a = fingerprint_paragraph("Apply Now to MITAOE")
    b = fingerprint_paragraph("apply now   to    mitaoe")
    c = fingerprint_paragraph("apply now\nto\nmitaoe")
    assert a == b == c


def test_split_paragraphs_respects_min_chars() -> None:
    text = "A short.\n\nThis paragraph is long enough to clear the threshold for sure.\n\nshort"
    paragraphs = split_paragraphs(text, min_chars=40)
    assert len(paragraphs) == 1
    assert "long enough" in paragraphs[0]


def test_registry_counts_documents() -> None:
    registry = ReusableComponentRegistry()
    fp = fingerprint_paragraph("Apply Now to MITAOE for admissions")
    registry.register(fp, "doc1", "Apply Now to MITAOE for admissions")
    registry.register(fp, "doc2", "Apply Now to MITAOE for admissions")
    registry.register(fp, "doc2", "Apply Now to MITAOE for admissions")
    assert registry.document_count(fp) == 2


def test_classify_component_type() -> None:
    assert classify_component_type("Apply now to begin your admission journey") == "admissions_cta"
    assert classify_component_type("Frequently Asked Questions") == "faq_widget"
    assert classify_component_type("Copyright 2024 MITAOE. All rights reserved.") == "footer"
    assert classify_component_type("MCA eligibility requires a relevant degree") is None


def test_classify_registry_only_marks_reused() -> None:
    chunks = [
        {"document_id": f"doc{i}", "text": "Apply now to MITAOE for admissions to the best programs offered."}
        for i in range(6)
    ] + [
        {"document_id": "unique", "text": "MCA eligibility requires a relevant undergraduate degree and entrance exam."}
    ]
    registry = build_registry(chunks)
    counts = classify_registry(registry, min_doc_count=5)
    assert counts.get("admissions_cta") == 1
    assert "admissions_cta" in counts


def test_assess_chunk_marks_reusable_when_majority_component() -> None:
    registry = ReusableComponentRegistry()
    cta = "Apply now to MITAOE for admissions to the best programs offered."
    # Pre-classify this fingerprint
    fp = fingerprint_paragraph(cta)
    registry.register(fp, "doc1", cta)
    registry.set_component_type(fp, "admissions_cta")

    chunk_text = f"{cta}\n\n{cta}\n\nA tiny tail."
    is_reusable, dominant, types = assess_chunk_components(chunk_text, registry)
    assert is_reusable is True
    assert dominant == "admissions_cta"


def test_assess_chunk_not_reusable_when_minor_component() -> None:
    registry = ReusableComponentRegistry()
    cta = "Apply now to MITAOE for admissions to the best programs offered."
    fp = fingerprint_paragraph(cta)
    registry.register(fp, "doc1", cta)
    registry.set_component_type(fp, "admissions_cta")

    long_unique = (
        "MCA eligibility requires a relevant undergraduate degree and an entrance "
        "examination score above the cutoff defined by the academic council each year."
    )
    chunk_text = f"{long_unique}\n\n{long_unique}\n\n{cta}"
    is_reusable, dominant, types = assess_chunk_components(chunk_text, registry)
    assert is_reusable is False
    assert "admissions_cta" in types
