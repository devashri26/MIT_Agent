from typing import Any

from pydantic import BaseModel, Field


ALLOWED_PAGE_TYPES = [
    "Admissions",
    "Programs",
    "Faculty",
    "Placements",
    "Club",
    "Events",
    "Curriculum",
    "Research",
    "Facilities",
    "Notices",
    "Blog",
    "General",
]

ALLOWED_SECTION_TYPES = [
    "eligibility",
    "fees",
    "placements",
    "faculty",
    "research",
    "facilities",
    "curriculum",
    "admissions",
    "hostel",
    "internships",
    "clubs",
    "events",
    "contact",
    "faq",
    "overview",
    "statistics",
    "syllabus",
    "general",
]

ALLOWED_QUALITY_FLAGS = [
    "low_content",
    "duplicate",
    "event_page",
    "cta_heavy",
    "boilerplate_heavy",
    "weak_classification",
    "thin_content",
    "non_canonical",
    "reusable_component",
    "cross_domain_contamination",
    "mixed_topic",
]


class NormalizedChunk(BaseModel):
    chunk_id: str
    document_id: str
    url: str
    canonical_url: str
    title: str
    department: str | None = None
    section_heading: str
    chunk_index: int
    text: str
    token_count: int
    content_type: str
    quality_score: float
    chunk_hash: str

    page_type: str
    page_type_confidence: float
    section_type: str
    retrieval_priority: float
    quality_flags: list[str] = Field(default_factory=list)
    is_canonical: bool = True

    section_path: list[str] = Field(default_factory=list)
    is_reusable_component: bool = False
    component_type: str | None = None
    mixed_topic: bool = False
    dominant_topics: list[str] = Field(default_factory=list)
    cross_domain_contamination: bool = False
    contamination_sources: list[str] = Field(default_factory=list)

    embedding_eligible: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)


class CorpusNormalizationReport(BaseModel):
    total_chunks: int = 0
    page_type_distribution: dict[str, int] = Field(default_factory=dict)
    section_type_distribution: dict[str, int] = Field(default_factory=dict)
    quality_flag_distribution: dict[str, int] = Field(default_factory=dict)
    canonical_pages: int = 0
    non_canonical_pages: int = 0
    weak_classifications: int = 0


class SemanticNormalizationReport(BaseModel):
    total_chunks: int = 0
    reusable_components_detected: int = 0
    component_type_distribution: dict[str, int] = Field(default_factory=dict)
    contaminated_chunks: int = 0
    contamination_source_distribution: dict[str, int] = Field(default_factory=dict)
    mixed_topic_chunks: int = 0
    hierarchy_sections_extracted: int = 0
    generic_overview_sections_replaced: int = 0
    semantic_sections_created: int = 0
