import random

# In a real implementation, you would use Spacy/NLTK and OpenAI/local LLMs here.
# For now, we simulate the NLP-based ATS scoring and Skill Gap analysis.

def analyze_resume_text(resume_text: str, target_career: str):
    """
    Simulates extracting text from a resume, analyzing it against a target career,
    and generating an ATS score and skill gap analysis.
    """
    # Mock ATS Score (e.g., 0-100)
    base_score = 65.0
    random_factor = random.uniform(-10.0, 25.0)
    ats_score = min(100.0, max(0.0, base_score + random_factor))

    # Mock Skill Gap Analysis
    skills_found = ["Python", "Communication"]
    skills_missing = []
    
    if "Data" in target_career:
        skills_missing = ["SQL", "Machine Learning", "Pandas"]
    elif "Software" in target_career:
        skills_missing = ["Data Structures", "System Design", "Docker"]
    else:
        skills_missing = ["Domain Specific Knowledge", "Project Management"]

    return {
        "ats_score": round(ats_score, 2),
        "skills_found": skills_found,
        "skills_missing": skills_missing,
        "recommendation": "Try adding more keywords related to your target role."
    }

def generate_interview_prep(target_career: str, missing_skills: list):
    """
    Simulates AI-driven interview preparation generation.
    """
    questions = [
        "Tell me about a time you had to learn a new technology quickly.",
        "How do you handle disagreements within a team?"
    ]
    
    if missing_skills:
        skill = missing_skills[0]
        questions.append(f"How would you approach a project requiring {skill} if you have limited experience?")
        
    if "Data" in target_career:
        questions.append("Explain the bias-variance tradeoff.")
    elif "Software" in target_career:
        questions.append("How would you design a scalable URL shortener?")

    return {
        "interview_questions": questions,
        "tips": [
            "Use the STAR method (Situation, Task, Action, Result) for behavioral questions.",
            "Don't be afraid to ask clarifying questions before diving into technical problems."
        ]
    }
