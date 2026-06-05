import re

CAREER_SKILLS = {
    "Data Scientist": ["python", "sql", "machine learning", "pandas", "scikit-learn", "statistics", "data modeling", "aws", "azure", "deep learning", "nlp", "tensorflow", "pytorch"],
    "Software Engineer": ["java", "python", "javascript", "c++", "system design", "data structures", "algorithms", "docker", "kubernetes", "ci/cd", "aws", "api"],
    "Data Analyst": ["sql", "python", "excel", "tableau", "power bi", "statistics", "data visualization", "pandas", "r"],
    "UX Designer": ["figma", "sketch", "wireframing", "prototyping", "user research", "ui", "interaction design", "adobe xd"],
    "Digital Marketer": ["seo", "sem", "content strategy", "google analytics", "ads", "social media", "campaign management"],
    "Business Analyst": ["excel", "sql", "tableau", "power bi", "stakeholder management", "requirements gathering", "agile", "jira"],
    "Cybersecurity": ["network security", "linux", "firewalls", "vulnerability assessment", "penetration testing", "siem", "incident response"],
    "Default": ["communication", "problem solving", "teamwork", "project management", "leadership", "agile"]
}

def analyze_resume_text(resume_text: str, target_career: str):
    """
    Extracts text from a resume, analyzes it against a target career's required skills,
    and generates an ATS score and skill gap analysis.
    """
    text_lower = resume_text.lower()
    
    # Match target career to predefined skills
    target_key = "Default"
    for career in CAREER_SKILLS.keys():
        if career.lower() in target_career.lower() or target_career.lower() in career.lower():
            target_key = career
            break
            
    required_skills = CAREER_SKILLS[target_key]
    
    skills_found = []
    skills_missing = []
    
    for skill in required_skills:
        # Use regex for whole word match
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            # For better formatting, capitalize appropriately
            formatted_skill = skill.title() if len(skill) > 3 else skill.upper()
            skills_found.append(formatted_skill)
        else:
            formatted_skill = skill.title() if len(skill) > 3 else skill.upper()
            skills_missing.append(formatted_skill)
            
    # Calculate ATS Score
    if required_skills:
        ats_score = (len(skills_found) / len(required_skills)) * 100.0
    else:
        ats_score = 0.0
        
    # Generate Recommendation and Overall Analysis
    if ats_score >= 80:
        recommendation = "Excellent match! Your resume is highly tailored to this role."
        overall_analysis = (
            f"Based on our AI analysis, your resume is an exceptionally strong fit for a {target_career} role. "
            f"You have successfully highlighted core competencies like {', '.join(skills_found[:3])}. "
            "To further optimize your profile, consider adding minor missing skills or quantifying your achievements."
        )
    elif ats_score >= 50:
        recommendation = "Good match, but you could add more relevant keywords to improve your chances."
        overall_analysis = (
            f"Your profile shows a solid foundation for {target_career}, demonstrating proficiency in {', '.join(skills_found[:2]) if skills_found else 'some key areas'}. "
            f"However, to pass strict ATS filters, you should explicitly include experience with {', '.join(skills_missing[:2]) if skills_missing else 'industry standard tools'}."
        )
    else:
        recommendation = "Low match. Try adding more keywords related to your target role to pass ATS."
        overall_analysis = (
            f"Currently, your resume lacks several critical keywords expected for a {target_career} position. "
            f"ATS systems often filter out resumes missing core skills like {', '.join(skills_missing[:3]) if skills_missing else 'required technical skills'}. "
            "Consider restructuring your bullet points to clearly reflect these technologies."
        )

    return {
        "ats_score": round(ats_score, 2),
        "skills_found": skills_found,
        "skills_missing": skills_missing,
        "recommendation": recommendation,
        "overall_analysis": overall_analysis
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
        roadmap_url = "https://roadmap.sh/ai-data-scientist"
    elif "Software" in target_career:
        questions.append("How would you design a scalable URL shortener?")
        roadmap_url = "https://roadmap.sh/backend"
    elif "UX" in target_career or "Design" in target_career:
        questions.append("How do you balance user needs with business goals?")
        roadmap_url = "https://roadmap.sh/ux-design"
    elif "Cyber" in target_career or "Security" in target_career:
        questions.append("How would you secure a newly deployed web application?")
        roadmap_url = "https://roadmap.sh/cyber-security"
    else:
        roadmap_url = "https://roadmap.sh/frontend" # default fallback

    return {
        "interview_questions": questions,
        "tips": [
            "Use the STAR method (Situation, Task, Action, Result) for behavioral questions.",
            "Don't be afraid to ask clarifying questions before diving into technical problems."
        ],
        "roadmap_url": roadmap_url
    }
