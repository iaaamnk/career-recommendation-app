"""
NLP-based Resume Analysis Module.

Uses TF-IDF vectorization and cosine similarity for ATS scoring,
and NLP-driven interview question generation based on skill gap context.
"""
import re
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------
# Career skill corpus — each career maps to a comprehensive text
# description used for TF-IDF similarity matching.
# ----------------------------------------------------------------
CAREER_SKILL_CORPUS = {
    "Data Scientist": (
        "python sql machine learning pandas scikit-learn statistics data modeling "
        "aws azure deep learning nlp tensorflow pytorch data analysis visualization "
        "hypothesis testing regression classification clustering feature engineering "
        "jupyter notebook data wrangling exploratory data analysis big data hadoop spark"
    ),
    "Software Engineer": (
        "java python javascript c++ system design data structures algorithms docker "
        "kubernetes ci/cd aws api rest microservices git testing unit integration "
        "agile scrum object oriented design patterns software architecture spring boot "
        "cloud computing linux debugging performance optimization"
    ),
    "Data Analyst": (
        "sql python excel tableau power bi statistics data visualization pandas r "
        "data warehousing etl reporting dashboards analytics business intelligence "
        "data cleaning pivot tables google analytics data modeling descriptive statistics"
    ),
    "UX Designer": (
        "figma sketch wireframing prototyping user research ui interaction design "
        "adobe xd design systems usability testing information architecture responsive "
        "design typography color theory user journey persona accessibility heuristics "
        "visual design layout"
    ),
    "Digital Marketer": (
        "seo sem content strategy google analytics ads social media campaign management "
        "email marketing conversion optimization content marketing copywriting "
        "facebook ads google ads ppc brand awareness audience targeting marketing automation"
    ),
    "Business Analyst": (
        "excel sql tableau power bi stakeholder management requirements gathering agile "
        "jira process modeling data interpretation business process project management "
        "user stories acceptance criteria gap analysis documentation workflows"
    ),
    "Cybersecurity Analyst": (
        "network security linux firewalls vulnerability assessment penetration testing "
        "siem incident response security monitoring ethical hacking cryptography "
        "compliance risk assessment threat intelligence malware analysis ids ips "
        "security auditing python scripting"
    ),
    "AI Researcher": (
        "python deep learning tensorflow pytorch reinforcement learning mathematics "
        "statistics research methods computer vision nlp transformers neural networks "
        "gans autoencoders attention mechanisms optimization gradient descent "
        "published papers research publication"
    ),
    "Machine Learning Engineer": (
        "python scikit-learn mlops tensorflow pytorch model deployment feature engineering "
        "data pipelines aws docker kubernetes model serving a/b testing experiment tracking "
        "hyperparameter tuning cross validation ensemble methods"
    ),
    "DevOps Engineer": (
        "docker kubernetes ci/cd aws terraform ansible jenkins linux shell scripting "
        "infrastructure as code monitoring logging cloud architecture networking "
        "git github actions prometheus grafana"
    ),
    "Project Manager": (
        "project management communication agile scrum jira risk management budgeting "
        "stakeholder management leadership team management planning resource allocation "
        "timeline scheduling waterfall kanban reporting"
    ),
    "Financial Analyst": (
        "financial modeling excel econometrics valuation financial analysis bloomberg "
        "terminal data analysis accounting budgeting forecasting ratio analysis "
        "investment analysis portfolio management risk assessment"
    ),
    "Default": (
        "communication problem solving teamwork project management leadership agile "
        "time management adaptability critical thinking collaboration"
    ),
}

# Keyword lists for skill gap detail (kept for backward compatibility in output)
CAREER_SKILLS_LIST = {
    "Data Scientist": ["python", "sql", "machine learning", "pandas", "scikit-learn", "statistics", "data modeling", "aws", "azure", "deep learning", "nlp", "tensorflow", "pytorch"],
    "Software Engineer": ["java", "python", "javascript", "c++", "system design", "data structures", "algorithms", "docker", "kubernetes", "ci/cd", "aws", "api"],
    "Data Analyst": ["sql", "python", "excel", "tableau", "power bi", "statistics", "data visualization", "pandas", "r"],
    "UX Designer": ["figma", "sketch", "wireframing", "prototyping", "user research", "ui", "interaction design", "adobe xd"],
    "Digital Marketer": ["seo", "sem", "content strategy", "google analytics", "ads", "social media", "campaign management"],
    "Business Analyst": ["excel", "sql", "tableau", "power bi", "stakeholder management", "requirements gathering", "agile", "jira"],
    "Cybersecurity Analyst": ["network security", "linux", "firewalls", "vulnerability assessment", "penetration testing", "siem", "incident response"],
    "AI Researcher": ["python", "deep learning", "tensorflow", "pytorch", "research methods", "mathematics", "nlp", "computer vision"],
    "Machine Learning Engineer": ["python", "scikit-learn", "mlops", "tensorflow", "docker", "model deployment", "feature engineering"],
    "DevOps Engineer": ["docker", "kubernetes", "ci/cd", "aws", "terraform", "linux", "jenkins", "ansible"],
    "Project Manager": ["project management", "communication", "agile", "scrum", "jira", "risk management", "leadership"],
    "Financial Analyst": ["financial modeling", "excel", "valuation", "financial analysis", "data analysis", "accounting"],
    "Default": ["communication", "problem solving", "teamwork", "project management", "leadership", "agile"],
}


def _match_career_key(target_career: str, mapping: dict) -> str:
    """Find the best matching key in a mapping for the given career string."""
    target_lower = target_career.lower()
    for key in mapping:
        if key.lower() in target_lower or target_lower in key.lower():
            return key
    return "Default"


def _build_tfidf_scorer():
    """Build a TF-IDF vectorizer fitted on all career corpus documents."""
    careers = list(CAREER_SKILL_CORPUS.keys())
    corpus = [CAREER_SKILL_CORPUS[c] for c in careers]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=500,
        lowercase=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix, careers


# Pre-build the TF-IDF model at module load
_vectorizer, _career_tfidf_matrix, _career_labels = _build_tfidf_scorer()


def analyze_resume_text(resume_text: str, target_career: str):
    """
    Analyzes resume text against a target career using TF-IDF vectorization
    and cosine similarity for ATS scoring, with detailed skill gap analysis.
    """
    text_lower = resume_text.lower()

    # --- TF-IDF Cosine Similarity ATS Score ---
    target_key = _match_career_key(target_career, CAREER_SKILL_CORPUS)
    target_idx = _career_labels.index(target_key)

    # Vectorize the resume text using the same fitted vectorizer
    resume_tfidf = _vectorizer.transform([text_lower])

    # Compute cosine similarity between resume and target career
    similarity = cosine_similarity(resume_tfidf, _career_tfidf_matrix[target_idx])[0][0]

    # Also compute similarity to all careers for context
    all_similarities = cosine_similarity(resume_tfidf, _career_tfidf_matrix)[0]
    top_career_indices = np.argsort(all_similarities)[::-1][:3]
    top_matching_careers = [
        {"career": _career_labels[i], "similarity": round(float(all_similarities[i]) * 100, 1)}
        for i in top_career_indices
    ]

    # Scale similarity to a 0-100 ATS score (cosine similarity typically 0-0.6 for text)
    ats_score = min(100.0, round(similarity * 150, 2))  # Scale factor for realistic scores

    # --- Skill Gap Analysis (keyword-level detail) ---
    skill_key = _match_career_key(target_career, CAREER_SKILLS_LIST)
    required_skills = CAREER_SKILLS_LIST[skill_key]

    skills_found = []
    skills_missing = []

    for skill in required_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            formatted = skill.title() if len(skill) > 3 else skill.upper()
            skills_found.append(formatted)
        else:
            formatted = skill.title() if len(skill) > 3 else skill.upper()
            skills_missing.append(formatted)

    # --- TF-IDF Feature Importance (top resume keywords) ---
    feature_names = _vectorizer.get_feature_names_out()
    resume_tfidf_dense = resume_tfidf.toarray()[0]
    top_term_indices = np.argsort(resume_tfidf_dense)[::-1][:10]
    top_resume_keywords = [
        feature_names[i] for i in top_term_indices if resume_tfidf_dense[i] > 0
    ]

    # --- Generate Recommendation ---
    if ats_score >= 80:
        recommendation = "Excellent match! Your resume is highly tailored to this role."
        overall_analysis = (
            f"Based on TF-IDF analysis, your resume achieves a {ats_score:.0f}% semantic match "
            f"for a {target_career} role. Key strengths include proficiency in "
            f"{', '.join(skills_found[:3]) if skills_found else 'relevant areas'}. "
            f"Your resume's top keyword signals: {', '.join(top_resume_keywords[:5])}."
        )
    elif ats_score >= 50:
        recommendation = "Good match, but adding more relevant keywords would improve your chances."
        overall_analysis = (
            f"Your profile shows a {ats_score:.0f}% semantic alignment with {target_career}, "
            f"with demonstrated skills in {', '.join(skills_found[:2]) if skills_found else 'some areas'}. "
            f"To improve ATS pass rate, incorporate keywords like "
            f"{', '.join(skills_missing[:3]) if skills_missing else 'industry-standard tools'}."
        )
    else:
        recommendation = "Low match. Restructure your resume with more role-specific keywords."
        overall_analysis = (
            f"TF-IDF analysis indicates a {ats_score:.0f}% semantic match — below the typical "
            f"ATS threshold for {target_career}. Critical missing signals include "
            f"{', '.join(skills_missing[:3]) if skills_missing else 'core technical skills'}. "
            f"Consider restructuring your resume to explicitly mention these technologies."
        )

    return {
        "ats_score": round(ats_score, 2),
        "skills_found": skills_found,
        "skills_missing": skills_missing,
        "recommendation": recommendation,
        "overall_analysis": overall_analysis,
        "top_resume_keywords": top_resume_keywords,
        "top_matching_careers": top_matching_careers,
        "similarity_method": "TF-IDF Cosine Similarity",
    }


# ----------------------------------------------------------------
# NLP-Driven Interview Preparation
# ----------------------------------------------------------------

# Template pools for dynamic question generation
_BEHAVIORAL_TEMPLATES = [
    "Describe a situation where you had to learn {skill} under a tight deadline. What was your approach?",
    "Tell me about a project where {skill} played a critical role. What challenges did you face?",
    "How would you explain {skill} to a non-technical stakeholder?",
    "Give an example of how you applied {skill} to solve a real-world problem.",
    "What steps would you take to get up to speed on {skill} for a new project?",
]

_TECHNICAL_TEMPLATES = {
    "Data Scientist": [
        "Explain the bias-variance tradeoff and how it affects model selection.",
        "How would you handle a dataset with significant class imbalance?",
        "Walk me through your approach to feature engineering for a {domain} problem.",
        "What metrics would you use to evaluate a {model_type} model and why?",
        "Describe how you would set up an A/B test to validate a model improvement.",
    ],
    "Software Engineer": [
        "How would you design a scalable URL shortener?",
        "Explain the trade-offs between SQL and NoSQL databases for a {domain} application.",
        "Walk me through how you would debug a production performance bottleneck.",
        "How do you ensure code quality in a fast-moving team?",
        "Describe your approach to writing maintainable and testable code.",
    ],
    "UX Designer": [
        "How do you balance user needs with business goals in your design process?",
        "Describe your approach to conducting usability testing.",
        "How would you redesign a feature that has low user engagement?",
        "Walk me through your process for creating a design system from scratch.",
    ],
    "Cybersecurity Analyst": [
        "How would you secure a newly deployed web application?",
        "Describe your incident response process for a suspected data breach.",
        "What tools and techniques do you use for vulnerability assessment?",
        "How do you stay current with emerging security threats?",
    ],
    "Default": [
        "How do you prioritize competing tasks with tight deadlines?",
        "Describe your approach to cross-functional collaboration.",
        "How do you handle ambiguous requirements in a project?",
        "Tell me about a time you had to adapt quickly to a changing situation.",
    ],
}

_DOMAIN_CONTEXTS = ["healthcare", "fintech", "e-commerce", "social media", "logistics", "education"]
_MODEL_TYPES = ["classification", "regression", "clustering", "recommendation"]

# Roadmap URL mapping
_ROADMAP_URLS = {
    "Data": "https://roadmap.sh/ai-data-scientist",
    "Software": "https://roadmap.sh/backend",
    "UX": "https://roadmap.sh/ux-design",
    "Design": "https://roadmap.sh/ux-design",
    "Cyber": "https://roadmap.sh/cyber-security",
    "Security": "https://roadmap.sh/cyber-security",
    "DevOps": "https://roadmap.sh/devops",
    "Cloud": "https://roadmap.sh/devops",
    "Machine Learning": "https://roadmap.sh/mlops",
    "AI": "https://roadmap.sh/ai-data-scientist",
    "Frontend": "https://roadmap.sh/frontend",
    "Backend": "https://roadmap.sh/backend",
    "Mobile": "https://roadmap.sh/android",
}


def generate_interview_prep(target_career: str, missing_skills: list):
    """
    Generates NLP-driven interview preparation content based on the target
    career and identified skill gaps. Uses template interpolation with
    contextual variation for dynamic, relevant questions.
    """
    random.seed(hash(target_career + str(missing_skills)) % (2**31))

    questions = []

    # 1. Generate skill-gap-aware behavioral questions
    if missing_skills:
        for skill in missing_skills[:2]:
            template = random.choice(_BEHAVIORAL_TEMPLATES)
            questions.append(template.format(skill=skill))

    # 2. Add career-specific technical questions
    tech_key = _match_career_key(target_career, _TECHNICAL_TEMPLATES)
    tech_questions = _TECHNICAL_TEMPLATES[tech_key]

    # Interpolate domain/model context into templates
    selected_tech = random.sample(tech_questions, min(3, len(tech_questions)))
    for q in selected_tech:
        q = q.format(
            domain=random.choice(_DOMAIN_CONTEXTS),
            model_type=random.choice(_MODEL_TYPES),
        )
        questions.append(q)

    # 3. Determine roadmap URL
    roadmap_url = "https://roadmap.sh/frontend"  # default
    for keyword, url in _ROADMAP_URLS.items():
        if keyword.lower() in target_career.lower():
            roadmap_url = url
            break

    return {
        "interview_questions": questions,
        "tips": [
            "Use the STAR method (Situation, Task, Action, Result) for behavioral questions.",
            "Don't be afraid to ask clarifying questions before diving into technical problems.",
            f"Focus on demonstrating transferable skills if you lack direct {target_career} experience.",
            "Prepare 2-3 concrete examples of projects where you applied relevant skills.",
        ],
        "roadmap_url": roadmap_url,
        "generation_method": "NLP-driven contextual template interpolation",
    }
