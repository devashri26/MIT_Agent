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
    is_preserved_content,
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


def test_fee_table_is_preserved_not_boilerplate() -> None:
    """Regression: BTech fee tables share text across all program admission pages.
    Previously they were flagged 'boilerplate' and excluded from retrieval, so fee
    queries couldn't be answered. is_preserved_content() must spare them."""
    fee_text = (
        "SR.NO TYPES OF FEES OPEN (IN RS.) OBC / EBC/EWS (IN RS.) "
        "1. Tuition Fees 1, 52, 173.00 76, 087.00 "
        "2. Development Fees 22, 827.00 22, 827.00 "
        "3. University Fees 737.00 737.00"
    )
    assert is_preserved_content(fee_text) is True
    assert classify_component_type(fee_text) is None


def test_eligibility_block_is_preserved() -> None:
    text = "Eligibility criteria for MCA: minimum 50 percent in graduation and entrance examination."
    assert is_preserved_content(text) is True
    assert classify_component_type(text) is None


def test_marketing_text_is_not_preserved() -> None:
    text = "Apply now to MITAOE the best engineering college. Click here. Download brochure."
    assert is_preserved_content(text) is False
    assert classify_component_type(text) == "admissions_cta"


def test_classify_registry_spares_high_value_content() -> None:
    """An academic-content fingerprint repeated across many docs must NOT be tagged
    reusable — the preserve list overrides the cross-doc frequency rule."""
    chunks = [
        {
            "document_id": f"doc{i}",
            "text": "Tuition Fees 1,52,173 Development Fees 22,827 University Fees 737",
        }
        for i in range(8)
    ]
    registry = build_registry(chunks)
    counts = classify_registry(registry, min_doc_count=5)
    # No component type assigned at all
    assert counts == {}
    fp = fingerprint_paragraph(chunks[0]["text"])
    assert registry.get_component_type(fp) is None


def test_chunk_with_boilerplate_surround_but_fee_content_is_spared() -> None:
    """Regression: chunks where the academic content (fee table) is wrapped in boilerplate
    that dominates by character mass must STILL not be flagged reusable. The earlier
    fix at the registry level wasn't enough — the chunk-level check needs to spare any
    chunk containing preserved content anywhere."""
    registry = ReusableComponentRegistry()
    # Boilerplate paragraph that appears across many docs:
    boilerplate = (
        "All candidates are advised to submit their applications to the Admission "
        "Authorities notified by the Government of Maharashtra on or before the last "
        "date of application unless specified differently elsewhere."
    )
    bp_fp = fingerprint_paragraph(boilerplate)
    registry.register(bp_fp, "doc1", boilerplate)
    registry.set_component_type(bp_fp, "boilerplate")

    fee_block = (
        "SR.NO TYPES OF FEES OPEN (IN RS.) "
        "1. Tuition Fees 1, 52, 173.00 "
        "2. Development Fees 22, 827.00 "
        "3. University Fees 737.00 Total 1, 75, 737.00"
    )
    # Two boilerplate paragraphs surrounding one fee paragraph → boilerplate dominates by mass.
    chunk_text = f"{boilerplate}\n\n{boilerplate}\n\n{fee_block}"
    is_reusable, dominant, types = assess_chunk_components(chunk_text, registry)
    assert is_reusable is False, "fee-bearing chunk must not be marked reusable"
    # boilerplate paragraphs should still be reported in `types` for explainability
    assert "boilerplate" in types
