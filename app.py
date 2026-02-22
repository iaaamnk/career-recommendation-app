import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from flask import Flask, request, jsonify, render_template
import warnings
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-before-going-to-production')

# ---------------- CONFIG ----------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
DATA_FILE_PATH = "AI-based Career Recommendation System.csv"

# ---------------- GLOBALS ----------------
rf_model = None
edu_le = None
skill_mlb = None
interest_mlb = None
target_le = None
riasec_scaler = None
feature_scaler = None
feature_names = None
kmeans_model = None
cluster_career_map = None

# ---------------- CONSTANTS ----------------
CAREER_GROUP_MAPPING = {
    "Data Scientist": "Data Analytics & Science",
    "Data Analyst": "Data Analytics & Science",
    "Biostatistician": "Data Analytics & Science",
    "Research Analyst": "Data Analytics & Science",
    "Data Engineer": "Data Analytics & Science",
    "AI Researcher": "Artificial Intelligence & Research",
    "AI Specialist": "Artificial Intelligence & Research",
    "Deep Learning Engineer": "Artificial Intelligence & Research",
    "NLP Engineer": "Artificial Intelligence & Research",
    "Machine Learning Engineer": "Artificial Intelligence & Research",
    "Research Scientist": "Artificial Intelligence & Research",
    "Software Engineer": "Software Development",
    "Software Developer": "Software Development",
    "Backend Developer": "Software Development",
    "Front-end Developer": "Software Development",
    "Full Stack Developer": "Software Development",
    "Mobile Developer": "Software Development",
    "UX Designer": "Design & UX",
    "UX Researcher": "Design & UX",
    "Graphic Designer": "Design & UX",
    "Digital Marketer": "Digital Marketing & Content",
    "Marketing Manager": "Digital Marketing & Content",
    "Content Strategist": "Digital Marketing & Content",
    "Financial Analyst": "Business & Finance",
    "Business Analyst": "Business & Finance",
    "Project Manager": "Business & Finance",
    "Embedded Systems Engineer": "Specialized Engineering & Infra",
    "Automation Engineer": "Specialized Engineering & Infra",
    "DevOps Engineer": "Specialized Engineering & Infra",
    "Cloud Engineer": "Specialized Engineering & Infra",
    "Cybersecurity Analyst": "Specialized Engineering & Infra",
    "Cybersecurity Specialist": "Specialized Engineering & Infra",
}

CAREER_PROFILES = {
    "Data Analytics & Science": {"R": 6, "I": 9, "A": 3, "S": 4, "E": 5, "C": 8},
    "Artificial Intelligence & Research": {"R": 7, "I": 10, "A": 2, "S": 3, "E": 4, "C": 7},
    "Software Development": {"R": 8, "I": 7, "A": 4, "S": 4, "E": 5, "C": 7},
    "Design & UX": {"R": 3, "I": 5, "A": 9, "S": 7, "E": 6, "C": 3},
    "Digital Marketing & Content": {"R": 4, "I": 5, "A": 7, "S": 8, "E": 9, "C": 6},
    "Business & Finance": {"R": 5, "I": 7, "A": 4, "S": 8, "E": 8, "C": 8},
    "Specialized Engineering & Infra": {"R": 9, "I": 7, "A": 3, "S": 4, "E": 6, "C": 8},
    "Default": {"R": 5, "I": 5, "A": 5, "S": 5, "E": 5, "C": 5},
}

CAREER_ROADMAPS = {
    "Data Analytics & Science": {
        "Required skills": ["Python (Pandas, Scikit-learn)", "Statistical Analysis", "SQL", "Cloud Basics (AWS/Azure)", "Data Modeling"],
        "Courses (Free)": ["Kaggle Learn Micro-Courses", "freeCodeCamp Data Science", "Google's Data Analytics Professional Certificate (via Coursera Audit)"],
        "Courses (Paid)": ["Simplilearn Data Scientist Master's Program", "MITx Micromasters in Statistics and Data Science"],
        "Certifications": ["Microsoft Certified: Azure Data Analyst Associate", "IBM Data Science Professional Certificate"],
        "Learning path": "Beginner: Python, SQL. Intermediate: Statistics, Data Visualization, ETL/ELT. Advanced: ML Basics, Cloud Data Services.",
        "Salary range in India": "₹6,00,000 - ₹20,00,000 per annum",
        "Future market demand": "High (5/5). Foundational role across all industries."
    },
    "Artificial Intelligence & Research": {
        "Required skills": ["Deep Learning Frameworks (TensorFlow/PyTorch)", "Advanced Mathematics", "MLOps", "Model Deployment", "Research Publication"],
        "Courses (Free)": ["Andrew Ng's Deep Learning Specialization (Coursera Audit)", "fast.ai Practical Deep Learning for Coders"],
        "Courses (Paid)": ["Udacity AI Programmer Nanodegree", "Stanford CS229 Machine Learning"],
        "Certifications": ["AWS Certified Machine Learning - Specialty", "Google Professional Machine Learning Engineer"],
        "Learning path": "Beginner: Core ML/DL. Intermediate: CNNs, RNNs, Transformers, Specific Domain (NLP/Vision). Advanced: MLOps, A/B Testing, Research Contribution.",
        "Salary range in India": "₹10,00,000 - ₹35,00,000+ per annum",
        "Future market demand": "Extremely High (5/5). Top-tier technical innovation role."
    },
    "Software Development": {
        "Required skills": ["DSA (Algorithms/Data Structures)", "System Design", "Specific Language (Java/Python/JS)", "CI/CD", "Testing (Unit/Integration)"],
        "Courses (Free)": ["Harvard CS50", "freeCodeCamp Development Tracks", "MIT OpenCourseware: Algorithms"],
        "Courses (Paid)": ["Scaler Academy", "Udemy System Design Interview Prep"],
        "Certifications": ["AWS Certified Developer - Associate", "Professional Cloud Developer (Google)"],
        "Learning path": "Beginner: Language Fundamentals, DSA. Intermediate: Database Management, API Development. Advanced: Microservices, Cloud Integration, Architecture Patterns.",
        "Salary range in India": "₹8,00,000 - ₹30,00,000+ per annum",
        "Future market demand": "Very High (5/5). The most stable technical backbone role.",
    },
    "Design & UX": {
        "Required skills": ["Prototyping (Figma/Sketch)", "User Research", "Wireframing", "Interaction Design", "Design Systems"],
        "Courses (Free)": ["Google UX Design Professional Certificate (Coursera Audit)", "Design+Code tutorials"],
        "Courses (Paid)": ["Interaction Design Foundation (IxDF) courses", "General Assembly UX Design Bootcamp"],
        "Certifications": ["Nielsen Norman Group UX Certification", "Google UX Design Professional Certificate"],
        "Learning path": "Beginner: Design Principles, Figma, Psychology. Intermediate: User research, usability testing. Advanced: Accessibility, Design Systems, Motion.",
        "Salary range in India": "₹6,00,000 - ₹18,00,000 per annum",
        "Future market demand": "High (4/5). Critical for product success and user retention.",
    },
    "Digital Marketing & Content": {
        "Required skills": ["SEO/SEM", "Content Strategy", "Digital Marketing (Ads)", "Analytics (Google/Adobe)", "Communication"],
        "Courses (Free)": ["Google Digital Garage", "HubSpot Academy"],
        "Courses (Paid)": ["Digital Marketing Nanodegree (Udacity)", "DMI Pro"],
        "Certifications": ["Google Ads Certification", "Meta Blueprint Certification", "Adobe Certified Expert - Analytics"],
        "Learning path": "Beginner: Marketing fundamentals, platform knowledge. Intermediate: Strategy, paid campaigns, analytics. Advanced: Branding, team leadership, ROI optimization.",
        "Salary range in India": "₹5,00,000 - ₹15,00,000 per annum",
        "Future market demand": "High (4/5). Essential for business visibility and growth."
    },
    "Business & Finance": {
        "Required skills": ["Financial Modeling", "Valuation", "Advanced Excel", "Data Interpretation", "Stakeholder Management"],
        "Courses (Free)": ["Corporate Finance Institute (CFI) Free Courses", "EdX Financial Analysis courses"],
        "Courses (Paid)": ["CFI Financial Modeling & Valuation Analyst (FMVA)", "Wall Street Prep (WSP) Training"],
        "Certifications": ["Chartered Financial Analyst (CFA)", "Certified Associate in Project Management (CAPM)"],
        "Learning path": "Beginner: Accounting, Excel, valuation. Intermediate: Financial modeling, project planning. Advanced: Portfolio Management, Risk Analysis, Strategic Consulting.",
        "Salary range in India": "₹7,00,000 - ₹20,00,000 per annum",
        "Future market demand": "Moderate-High (3/5). Stable demand for strategic roles."
    },
    "Specialized Engineering & Infra": {
        "Required skills": ["Linux/Shell Scripting", "Cloud Architecture (IaaS, PaaS)", "Infrastructure as Code (Terraform/Ansible)", "Security Protocols", "Networking"],
        "Courses (Free)": ["freeCodeCamp DevOps/Cybersecurity Tracks", "The Linux Foundation courses"],
        "Courses (Paid)": ["Certified Ethical Hacker (CEH) courses", "Cloud provider professional certifications"],
        "Certifications": ["AWS Certified Solutions Architect", "CompTIA Security+", "Certified Ethical Hacker (CEH)"],
        "Learning path": "Beginner: OS/Networking fundamentals. Intermediate: Cloud services, CI/CD, Scripting. Advanced: Security Auditing, Large-scale infrastructure management, Disaster Recovery.",
        "Salary range in India": "₹9,00,000 - ₹28,00,000+ per annum",
        "Future market demand": "Very High (5/5). Critical for stable, secure, and scalable systems."
    },
    "Default": {
        "Required skills": ["General communication", "Problem-solving", "Adaptability"],
        "Courses (Free)": ["Start with foundational skills"],
        "Courses (Paid)": ["Consult a career counselor"],
        "Certifications": ["Not applicable"],
        "Learning path": "Explore foundational training and internship opportunities.",
        "Salary range in India": "Varies widely",
        "Future market demand": "Requires specific focus"
    }
}

# ---------------- HELPERS ----------------
def get_imputed_riasec(career, code):
    base = CAREER_PROFILES.get(career, CAREER_PROFILES["Default"]).get(code, 5)
    return max(0, min(10, base + random.randint(-1, 1)))

def unsupervised_recommendation(X_input):
    cluster_id = kmeans_model.predict(X_input)[0]
    return int(cluster_id), cluster_career_map.get(cluster_id, "Unknown")

def get_top_alternatives(probas, predicted_index, top_n=3):
    sorted_indices = np.argsort(probas)[::-1]
    alternatives = []
    count = 0
    for idx in sorted_indices:
        if idx != predicted_index and count < top_n:
            career = target_le.inverse_transform([idx])[0]
            score = probas[idx]
            alternatives.append((career, score))
            count += 1
    return alternatives

def generate_career_roadmap(career_name):
    return CAREER_ROADMAPS.get(career_name, CAREER_ROADMAPS["Default"])

# ---------------- TRAINING ----------------
def load_and_train_model():
    global rf_model, edu_le, skill_mlb, interest_mlb, target_le
    global riasec_scaler, feature_scaler, feature_names
    global kmeans_model, cluster_career_map

    try:
        df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE_PATH} not found.")
        return

    # Clean list columns
    df['Skills'] = df['Skills'].fillna('').apply(lambda x: [i.strip() for i in str(x).split(';') if i])
    df['Interests'] = df['Interests'].fillna('').apply(lambda x: [i.strip() for i in str(x).split(';') if i])

    # Map careers
    df['Recommended_Career'] = df['Recommended_Career'].map(CAREER_GROUP_MAPPING).fillna('Default')

    # Impute RIASEC
    riasec_codes = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    for code in riasec_codes:
        df[code] = df['Recommended_Career'].apply(lambda c: get_imputed_riasec(c, code[0]))

    # Encoders
    edu_le = LabelEncoder()
    skill_mlb = MultiLabelBinarizer()
    interest_mlb = MultiLabelBinarizer()
    target_le = LabelEncoder()
    riasec_scaler = StandardScaler()
    feature_scaler = StandardScaler()

    # Preprocessing
    df['Education_Cleaned'] = df['Education'].str.replace("'", "")
    df['Education_Encoded'] = edu_le.fit_transform(df['Education_Cleaned'])

    skill_df = pd.DataFrame(skill_mlb.fit_transform(df['Skills']),
                            columns=[f"Skill_{c}" for c in skill_mlb.classes_])
    interest_df = pd.DataFrame(interest_mlb.fit_transform(df['Interests']),
                               columns=[f"Interest_{c}" for c in interest_mlb.classes_])

    riasec_df = pd.DataFrame(
        riasec_scaler.fit_transform(df[riasec_codes]),
        columns=[f"RIASEC_{c}" for c in riasec_codes]
    )

    X = pd.concat([df[['Age']], riasec_df, skill_df, interest_df], axis=1)
    X['Age_Scaled'] = feature_scaler.fit_transform(X[['Age']])
    X = X.drop(columns=['Age'])

    feature_names = X.columns.tolist()
    X_values = X.values
    y = target_le.fit_transform(df['Recommended_Career'])

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, class_weight='balanced')
    rf_model.fit(X_values, y)

    # Train KMeans
    kmeans_model = KMeans(n_clusters=len(target_le.classes_), random_state=RANDOM_SEED, n_init=10)
    clusters = kmeans_model.fit_predict(X_values)

    cluster_career_map = {}
    df['Cluster'] = clusters
    for c in np.unique(clusters):
        # Find the most frequent career in this cluster
        cluster_data = df[df['Cluster'] == c]
        if not cluster_data.empty:
            label = cluster_data['Recommended_Career'].mode()[0]
            cluster_career_map[c] = label
        else:
            cluster_career_map[c] = "Unknown"

    print("Model trained | Silhouette Score:", round(silhouette_score(X_values, clusters), 4))

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/how_it_works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": rf_model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if rf_model is None:
        return jsonify({"error": "Model not trained."}), 500

    data = request.json
    
    age = data.get('age')
    education = data.get('education')
    skills = data.get('skills', [])
    interests = data.get('interests', [])
    riasec = data.get('riasec_scores', [])

    if not all([age, education, riasec]) or len(riasec) != 6:
        return jsonify({"error": "Invalid input data"}), 400

    input_df = pd.DataFrame({
        "Age": [age],
        "Education_Cleaned": [education.replace("'", "")],
        "Skills": [skills],
        "Interests": [interests],
        "Realistic": [riasec[0]],
        "Investigative": [riasec[1]],
        "Artistic": [riasec[2]],
        "Social": [riasec[3]],
        "Enterprising": [riasec[4]],
        "Conventional": [riasec[5]]
    })

    # Transform features
    try:
        # Note: We don't really use education encoded in the X feature set in the user's snippet logic?
        # Checking Training X: X = pd.concat([df[['Age']], riasec_df, skill_df, interest_df], axis=1)
        # It seems Education_Encoded was computed but NOT used in the X feature set in the provided training code snippet.
        # "X = pd.concat([df[['Age']], riasec_df, skill_df, interest_df], axis=1)"
        # But in the original code it WAS used: "X_base = df[['Age', 'Education_Encoded']]"
        # The user's NEW snippet text says: "X = pd.concat([df[['Age']], riasec_df, skill_df, interest_df], axis=1)"
        # So Education is EXCLUDED from the features in the new model. I will follow the user's snippet.
        pass 
    except Exception as e:
        pass

    skill_df = pd.DataFrame(skill_mlb.transform(input_df['Skills']),
                            columns=[f"Skill_{c}" for c in skill_mlb.classes_])
    interest_df = pd.DataFrame(interest_mlb.transform(input_df['Interests']),
                               columns=[f"Interest_{c}" for c in interest_mlb.classes_])
    
    riasec_cols = ['Realistic','Investigative','Artistic','Social','Enterprising','Conventional']
    riasec_df = pd.DataFrame(
        riasec_scaler.transform(input_df[riasec_cols]),
        columns=[f"RIASEC_{c}" for c in riasec_cols]
    )

    X_new = pd.concat([riasec_df, skill_df, interest_df], axis=1)
    # The user snippet had X_new = pd.concat([riasec_df...]). values for Age were not there initially.
    # Then X_new['Age_Scaled'] = ...
    # This works because X_new is a DataFrame here.
    
    X_new['Age_Scaled'] = feature_scaler.transform(input_df[['Age']])
    
    # Realign columns to match training
    X_new = X_new.reindex(columns=feature_names, fill_value=0).values

    # Predict
    probas = rf_model.predict_proba(X_new)[0]
    idx = np.argmax(probas)
    recommended_career = target_le.inverse_transform([idx])[0]
    confidence = round(probas[idx], 4)

    # Unsupervised
    cluster_id, unsup_career = unsupervised_recommendation(X_new)

    # Detailed Response Generation for Frontend
    top_alternatives = get_top_alternatives(probas, idx, top_n=3)
    top_3_careers = [{"career": alt[0], "score": round(alt[1], 4)} for alt in top_alternatives]
    
    riasec_dict = dict(zip(riasec_cols, riasec))
    top_riasec = sorted(riasec_dict.items(), key=lambda item: item[1], reverse=True)[:2]
    
    explanation = (
        f"The system recommends **{recommended_career}** ({confidence*100:.1f}% Match). "
        f"Unsupervised analysis also suggests **{unsup_career}** (Cluster {cluster_id}). "
        f"Your profile aligns with **{top_riasec[0][0]}** and **{top_riasec[1][0]}** traits."
    )
    
    career_roadmap = generate_career_roadmap(recommended_career)

    return jsonify({
        "Recommended_Career": recommended_career,
        "Recommendation_Score": confidence,
        "Confidence": confidence, # Keeping both for compatibility
        "Unsupervised_Cluster": cluster_id,
        "Unsupervised_Recommendation": unsup_career,
        "Top_3_Careers": top_3_careers,
        "Explanation": explanation,
        "Career_Roadmap": career_roadmap
    })

if __name__ == "__main__":
    load_and_train_model()
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, port=5000)
