import re
from langsmith import traceable

HOSPITAL_KEYWORDS = {
    "doctor", "doctors", "timing", "timings", "available", "availability",
    "hospital", "location", "address", "specialist", "speciality", "specialty",
    "department", "cardiologist", "neurologist", "dermatologist", "orthopedic",
    "ent", "physician", "surgeon", "consult", "appointment", "op", "branch"
}

PROBLEM_KEYWORDS = {
    "pain", "fever", "cough", "cold", "headache", "dizzy", "dizziness", "vomiting",
    "weak", "weakness", "sore throat", "chest pain", "stomach pain", "rash",
    "breathing", "breath", "nausea", "diarrhea", "injury", "swelling", "symptom",
    "allergy", "infection", "hurt"
}

@traceable(name="Query Classification", run_type="chain")
def classify_query(query: str) -> str:
    q = query.lower().strip()

    if any(word in q for word in HOSPITAL_KEYWORDS):
        return "HOSPITAL_INFO"

    if any(word in q for word in PROBLEM_KEYWORDS):
        return "PROBLEM"

    # heuristic for first-person symptom statements
    if re.search(r"\b(i have|i am having|i feel|my|suffering from)\b", q):
        return "PROBLEM"

    return "GENERAL"