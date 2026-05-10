from pathlib import Path

import orjson

from backend.retrieval.bm25_service import BM25RetrievalService


BENCHMARK_QUERIES = [
    "What is MCA eligibility?",
    "Who is faculty coordinator of IEEE?",
    "What are hostel facilities?",
    "Mechanical placement statistics",
    "BTech AI curriculum",
    "Computer Engineering course structure",
    "Information Technology syllabus",
    "Chemical engineering faculty profile",
    "Civil engineering placements",
    "What is admission process?",
    "What entrance exams are accepted?",
    "What is the fee structure?",
    "Hostel rules and accommodation",
    "Library facilities",
    "Research publications computer engineering",
    "IEEE student branch coordinator",
    "NSS activities",
    "Entrepreneurship development cell",
    "Mechanical engineering labs",
    "AI ML department curriculum",
    "Data Science department",
    "Placement highest package",
    "Internship policy",
    "Academic calendar",
    "Exam timetable",
    "PhD admission eligibility",
    "MTech programs",
    "BTech information technology",
    "Faculty qualification mechanical",
    "Civil engineering infrastructure",
    "Scholarship details",
    "International internship",
    "Anti ragging committee",
    "Grievance redressal",
    "Student clubs",
    "Cultural events",
    "Sports facilities",
    "Alumni association",
    "Training and placement cell",
    "Recruiters visiting campus",
    "Computer engineering HOD",
    "Mechanical engineering HOD",
    "Chemical engineering HOD",
    "Electronics engineering HOD",
    "Information technology HOD",
    "Board of studies computer engineering",
    "Curriculum semester 1",
    "Honours minor degree",
    "Industry 4.0 blog",
    "Alandi engineering college",
]


def main() -> None:
    service = BM25RetrievalService()
    output = []
    for query in BENCHMARK_QUERIES:
        response = service.search(query, top_k=5)
        output.append(response.model_dump(mode="json"))
    report_path = Path("reports/bm25_benchmark_results.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(orjson.dumps(output, option=orjson.OPT_INDENT_2))
    print(f"Wrote {len(output)} benchmark queries to {report_path}")


if __name__ == "__main__":
    main()
