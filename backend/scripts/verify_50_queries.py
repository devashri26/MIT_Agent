"""Run 50 sample user questions through retrieval+rerank+context and report which ones
surface the expected information. No LLM calls — pure retrieval validation."""

from __future__ import annotations

from backend.context.context_builder import build_grounded_context
from backend.retrieval.reranked_retrieval import RerankedRetrievalService


# Each entry: (question, must_contain_any_of_these_keywords).
# A question PASSES if any context block contains any of its expected keywords (case-insensitive).
CASES: list[tuple[str, list[str]]] = [
    # Admissions
    ("What courses are offered at MIT AOE?", ["btech", "mtech", "computer engineering", "mechanical", "civil"]),
    ("What is the admission process for BTech?", ["mht-cet", "mht cet", "cap", "centralised admission", "centralized admission", "merit basis", "intake", "branch intake", "btech", "b.tech"]),
    ("What documents are required for admission?", ["leaving certificate", "marksheet", "domicile", "aadhaar", "scorecard"]),
    ("Is hostel facility available?", ["hostel", "accommodation"]),
    ("What is the eligibility for BTech admission?", ["eligibility", "physics", "chemistry", "mathematics", "pcm", "12th"]),
    ("Does MIT AOE provide scholarships?", ["scholarship", "ebc", "tfws", "financial aid"]),
    ("Can I get direct admission?", ["direct second year", "lateral entry", "management quota", "institutional"]),
    ("What is the college intake capacity?", ["intake", "seats", "sanctioned"]),
    ("What are the admission dates?", ["admission", "cet cell", "schedule"]),
    ("Where is MIT AOE located?", ["alandi", "pune"]),
    # Fees
    ("What is the fee structure for BTech?", ["22827", "152173", "1, 52, 173", "tuition fees"]),
    ("Is there any fee concession available?", ["ebc", "tfws", "scholarship", "concession", "reservation"]),
    ("What is the hostel fee?", ["hostel", "accommodation", "fee"]),
    ("Can fees be paid in installments?", ["installment", "payment", "refund"]),
    ("What payment methods are accepted?", ["online", "demand draft", "bank", "payment", "dd"]),
    # Placements
    ("What is the placement percentage at MIT AOE?", ["placement", "570", "placed", "lpa"]),
    ("Which companies visit the campus?", ["accenture", "cognizant", "tcs", "infosys", "wipro", "ibm", "kpit", "zensar", "persistent"]),
    ("What is the highest package offered?", ["28 lpa", "highest package", "ctc"]),
    ("Does the college provide placement training?", ["training", "mock interview", "soft skill", "aptitude", "crpc"]),
    ("Are internships provided?", ["internship"]),
    ("Which branch has the best placements?", ["placement", "computer", "it"]),
    # Spot Round
    ("What is a spot round?", ["spot round", "institutional round", "vacant seat"]),
    ("How can I apply for spot rounds?", ["spot round", "institutional round", "vacant seat", "apply"]),
    ("What documents are needed for spot round admission?", ["spot round", "documents", "cet"]),
    ("Are spot round admissions based on merit?", ["spot round", "merit", "vacant"]),
    ("When are spot rounds conducted?", ["spot round", "after cap", "vacancies"]),
    # Campus & Facilities
    ("Does MIT AOE have Wi-Fi campus?", ["wi-fi", "wifi", "internet", "campus"]),
    ("Is transportation facility available?", ["bus", "transport", "shuttle"]),
    ("Does the college have a library?", ["library", "book", "journal", "digital"]),
    ("Are sports facilities available?", ["sports", "gymnasium", "playground", "ground"]),
    ("Does the college organize technical events?", ["hackathon", "workshop", "fest", "symposium", "competition"]),
    ("Is there a coding club or technical community?", ["codechef", "club", "ieee", "coding"]),
    # Academics
    ("What is the academic calendar?", ["academic calendar", "semester", "schedule", "exam"]),
    ("How are internal marks calculated?", ["internal", "marks", "attendance", "evaluation", "assessment"]),
    ("Does MIT AOE follow autonomous curriculum?", ["autonomous", "curriculum", "syllabus"]),
    ("Are there industry projects?", ["project", "industry", "capstone", "internship"]),
    ("Does the college support research activities?", ["research", "publication", "patent"]),
    # Faculty
    ("Who is the HOD of Computer Engineering?", ["computer engineering", "hod", "head", "professor"]),
    ("What are the faculty research areas?", ["research", "machine learning", "ai", "cloud", "iot", "cybersecurity", "data science"]),
    ("How can I contact faculty members?", ["@mitaoe", "contact", "email"]),
    # General
    ("Tell me about MIT AOE.", ["mitaoe", "autonomous", "engineering", "alandi", "pune"]),
    ("What are the college timings?", ["timing", "hours", "schedule"]),
    ("How can I contact the admission office?", ["admission", "contact", "email", "phone"]),
    ("Does the college provide entrepreneurship support?", ["entrepreneur", "edc", "startup", "innovation"]),
    ("Is ragging prohibited on campus?", ["anti ragging", "anti-ragging", "ragging"]),
    ("Can alumni help with placements?", ["alumni"]),
    ("Does the college conduct workshops?", ["workshop", "seminar", "training", "expert"]),
    ("What coding languages are taught?", ["python", "java", "c++", "javascript", "programming"]),
    ("Does the college support hackathons?", ["hackathon", "hack"]),
    ("Can students work on real-world projects?", ["project", "industry", "real-world", "capstone"]),
]


def main() -> None:
    svc = RerankedRetrievalService(candidate_pool=20)
    pass_count = 0
    fails: list[tuple[int, str, list[str], list[str]]] = []

    for i, (question, keywords) in enumerate(CASES, start=1):
        r = svc.search(question, top_k=7)
        ctx = build_grounded_context(query=question, intent=r.intent, reranked=r.results, token_budget=5000)
        hit_kw: list[str] = []
        for block in ctx.context_blocks:
            text_lower = (block.text + " " + block.title).lower()
            for kw in keywords:
                if kw.lower() in text_lower and kw not in hit_kw:
                    hit_kw.append(kw)
        if hit_kw:
            pass_count += 1
            mark = "✅"
        else:
            mark = "❌"
            fails.append((i, question, keywords, [r.results[0].title[:50] if r.results else "(no results)"]))
        print(f"{mark} Q{i:02d}  {question}")
        if hit_kw:
            print(f"      found: {hit_kw[:3]}")
        else:
            print(f"      intent={r.intent}  top: {r.results[0].title[:60] if r.results else '—'}")

    print()
    print(f"=== {pass_count}/{len(CASES)} pass  ({len(fails)} fail) ===")
    if fails:
        print()
        print("FAILED QUESTIONS:")
        for i, q, kw, top in fails:
            print(f"  Q{i:02d}  {q}")
            print(f"        expected any of: {kw}")
            print(f"        top retrieved: {top}")


if __name__ == "__main__":
    main()
