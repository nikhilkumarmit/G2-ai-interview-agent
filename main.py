import os, json, time, argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DATA = Path("data")
DATA.mkdir(exist_ok=True)


def ask_ai(prompt):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content


def read_file(path):
    if path.endswith(".pdf"):
        return "\n".join(p.extract_text() or ""
                         for p in PdfReader(path).pages)
    return Path(path).read_text()


def run(resume_path, jd_path, limit=8, minutes=12):
    resume, jd = read_file(resume_path), read_file(jd_path)

    info = ask_ai(f"""
    Analyse this resume and job description.
    Return JSON with candidate_name, skills, projects,
    target_role and five interview_questions.

    Resume: {resume[:8000]}
    Job description: {jd[:5000]}
    """)
    info = json.loads(info.strip().removeprefix("```json").removesuffix("```"))

    print("Candidate:", info.get("candidate_name", "Unknown"))
    print("Skills:", info.get("skills", []))
    print("Role:", info.get("target_role", "Unknown"))

    answers, transcript = [], []
    start = time.time()

    for question in info["interview_questions"][:limit]:
        if (time.time() - start) / 60 >= minutes:
            break

        print("\nAI:", question)
        answer = input("You (or /end): ").strip()
        if answer.lower() == "/end":
            break

        answers.append({"question": question, "answer": answer})
        transcript.append({"question": question, "answer": answer})

    report = ask_ai(f"""
    Give interview feedback with scores, strengths,
    improvements and an overall note. Use evidence from answers.
    Do not make hiring decisions.

    Answers: {json.dumps(answers)}
    """)

    path = DATA / f"report_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(report, encoding="utf-8")
    print("\nFeedback report saved:", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True)
    p.add_argument("--jd", required=True)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--minutes", type=int, default=12)
    a = p.parse_args()
    run(a.resume, a.jd, a.limit, a.minutes)
