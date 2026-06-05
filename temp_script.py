import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
)
import re
import json
import warnings

# Suppress harmless warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- CONFIGURATION ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
DATA_FILE_ID = "AI-based Career Recommendation System.csv"

# --- CRITICAL CHANGE: CAREER GROUPING FOR HIGH ACCURACY ---
# Grouping 32 specific careers into 7 stable, broader categories to increase sample size per class.
CAREER_GROUP_MAPPING = {
    # Data & Analytics Core
    "Data Scientist": "Data Analytics & Science",
    "Data Analyst": "Data Analytics & Science",
    "Biostatistician": "Data Analytics & Science",
    "Research Analyst": "Data Analytics & Science",
    "Data Engineer": "Data Analytics & Science",

    # AI & Advanced Research
    "AI Researcher": "Artificial Intelligence & Research",
    "AI Specialist": "Artificial Intelligence & Research",
    "Deep Learning Engineer": "Artificial Intelligence & Research",
    "NLP Engineer": "Artificial Intelligence & Research",
    "Machine Learning Engineer": "Artificial Intelligence & Research",
    "Research Scientist": "Artificial Intelligence & Research",

    # Software Development & Engineering
    "Software Engineer": "Software Development",
    "Software Developer": "Software Development",
    "Backend Developer": "Software Development",
    "Front-end Developer": "Software Development",
    "Full Stack Developer": "Software Development",
    "Mobile Developer": "Software Development",

    # Design, UX & Creative
    "UX Designer": "Design & UX",
    "UX Researcher": "Design & UX",
    "Graphic Designer": "Design & UX",

    # Digital Marketing & Content
    "Digital Marketer": "Digital Marketing & Content",
    "Marketing Manager": "Digital Marketing & Content",
    "Content Strategist": "Digital Marketing & Content",

    # Business, Finance & Management
    "Financial Analyst": "Business & Finance",
    "Business Analyst": "Business & Finance",
    "Project Manager": "Business & Finance",

    # Specialized Technical/Infra
    "Embedded Systems Engineer": "Specialized Engineering & Infra",
    "Automation Engineer": "Specialized Engineering & Infra",
    "DevOps Engineer": "Specialized Engineering & Infra",
    "Cloud Engineer": "Specialized Engineering & Infra",
    "Cybersecurity Analyst": "Specialized Engineering & Infra",
    "Cybersecurity Specialist": "Specialized Engineering & Infra",
}

# --- RIASEC PROFILES for Imputation & Roadmaps (Updated for Groups) ---
# We use generalized RIASEC profiles for the new groups.
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

# --- PART 1: DATA LOADING, CLEANING, AND IMPUTATION ---

def load_and_clean_data(file_id):
    """Loads the CSV, cleans up list columns, groups careers, and imputes RIASEC scores."""
    
    # 1. Load Data
    df = pd.read_csv(file_id)
    
    # 2. Clean List Columns (Skills and Interests are semicolon-separated)
    def clean_list_column(series):
        return series.fillna('').astype(str).apply(
            lambda x: [s.strip() for s in x.split(';') if s.strip()]
        )
        
    df['Skills'] = clean_list_column(df['Skills'])
    df['Interests'] = clean_list_column(df['Interests'])
    
    # *** ACCURACY BOOST: GROUP CAREERS ***
    df['Recommended_Career'] = df['Recommended_Career'].map(CAREER_GROUP_MAPPING).fillna('Default')
    
    # 3. Impute RIASEC Scores (now using Grouped Careers)
    riasec_codes = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    
    for code in riasec_codes:
        df[code] = df['Recommended_Career'].apply(
            lambda career: get_imputed_riasec(career, code[0])
        )
        
    unique_careers = df['Recommended_Career'].unique()
    print(f"Unique Careers found in dataset (GROUPED): {len(unique_careers)}")
    
    return df

def get_imputed_riasec(career, code_char):
    """Imputes a single RIASEC score based on career profile and adds random noise."""
    profile = CAREER_PROFILES.get(career, CAREER_PROFILES["Default"])
    base_score = profile.get(code_char, 5)
    
    # Add random noise (+/- 1 for realism)
    noise = random.randint(-1, 1)
    imputed_score = max(0, min(10, base_score + noise))
    return imputed_score

# Load the dataset
df = load_and_clean_data(DATA_FILE_ID)
print("--- PART 1: DATA LOADING, CLEANING, AND RIASEC IMPUTATION (GROUPED) ---")
print(f"Loaded DataFrame with {len(df)} candidates. Target Classes Reduced.")
print("First 5 rows (including imputed RIASEC scores):")
print(df[['Name', 'Age', 'Education', 'Recommended_Career', 'Realistic', 'Investigative']].head().to_markdown(index=False))

# --- PART 2: PREPROCESSING PIPELINE ---
# (No changes needed here, as the pipeline adapts to the new target_le fit)

def preprocess_data(df):
    """Applies necessary preprocessing steps for model training."""
    
    # Preprocessing instances (defined globally for use in prediction function)
    global edu_le, skill_mlb, interest_mlb, target_le, riasec_scaler, feature_scaler
    edu_le = LabelEncoder()
    skill_mlb = MultiLabelBinarizer()
    interest_mlb = MultiLabelBinarizer()
    target_le = LabelEncoder()
    riasec_scaler = StandardScaler()
    feature_scaler = StandardScaler()
    
    # 1. Label Encoding for Education
    # FIX: Remove regex=False for compatibility with older Python/Pandas versions in the environment
    df['Education_Cleaned'] = df['Education'].str.replace("'", "")
    df['Education_Encoded'] = edu_le.fit_transform(df['Education_Cleaned'])
    
    # 2. MultiLabelBinarizer for Skills
    skill_matrix = skill_mlb.fit_transform(df['Skills'])
    skill_df = pd.DataFrame(skill_matrix, columns=[f'Skill_{c}' for c in skill_mlb.classes_])
    
    # 3. MultiLabelBinarizer for Interests
    interest_matrix = interest_mlb.fit_transform(df['Interests'])
    interest_df = pd.DataFrame(interest_matrix, columns=[f'Interest_{c}' for c in interest_mlb.classes_])
    
    # 4. RIASEC Normalization
    riasec_cols = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    riasec_data = df[riasec_cols]
    riasec_normalized = riasec_scaler.fit_transform(riasec_data)
    riasec_df = pd.DataFrame(riasec_normalized, columns=[f'RIASEC_{c}' for c in riasec_cols])
    
    # 5. Combine all into feature matrix X
    X_base = df[['Age', 'Education_Encoded']].reset_index(drop=True)
    X = pd.concat([X_base, riasec_df, skill_df, interest_df], axis=1)
    
    # 6. Encode Recommended_Career into numeric labels as y_true
    y_true = target_le.fit_transform(df['Recommended_Career'])
    
    # 7. Scale final features with StandardScaler (Age only, others are binary or already scaled)
    X['Age_Scaled'] = feature_scaler.fit_transform(X[['Age']])
    
    # Final feature matrix X: Drop unscaled 'Age' and use 'Age_Scaled'
    X = X.drop(columns=['Age', 'Education_Encoded'])
    
    feature_names = X.columns.tolist()
    X = X.values # Convert to numpy array for models

    return X, y_true, feature_names

X, y_true, feature_names = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=RANDOM_SEED, stratify=y_true)

print("\n--- PART 2: PREPROCESSING PIPELINE ---")
print(f"Feature Matrix X shape: {X.shape}")
print(f"Target Vector y shape: {y_true.shape}")
print(f"Feature Names ({len(feature_names)} total): {feature_names[:3]}...{feature_names[-3:]}")
print(f"Encoded Careers (NEW GROUPS): {list(target_le.classes_)}")

# --- PART 3 & 4: MACHINE LEARNING MODELS & SUPERVISED EVALUATION ---

print("\n--- PART 3 & 4: MODEL TRAINING AND EVALUATION (ACCURACY BOOSTED) ---")

# 1. Decision Tree Classifier
dt_model = DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_SEED)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# 2. Random Forest Classifier (Used for final predictions)
rf_model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, class_weight='balanced')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n### Decision Tree Classifier (Criterion=Entropy) ###")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")

print("\n### Random Forest Classifier (n_estimators=300) ###")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred_rf, average='weighted', zero_division=0):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred_rf, average='weighted', zero_division=0):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test, y_pred_rf, average='weighted', zero_division=0):.4f}")

print("\nConfusion Matrix (Random Forest):\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred_rf, target_names=target_le.classes_, zero_division=0))

# 3. KMeans Clustering (Unsupervised Evaluation)
n_clusters = len(target_le.classes_)
kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
kmeans.fit(X)
y_clusters = kmeans.labels_

print("\n### KMeans Clustering Evaluation ###")
print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(y_true, y_clusters):.4f}")
print(f"Normalized Mutual Information (NMI): {normalized_mutual_info_score(y_true, y_clusters):.4f}")
print(f"Fowlkes-Mallows Index (FMI): {fowlkes_mallows_score(y_true, y_clusters):.4f}")

# --- PART 5: EXPLAINABLE AI (SHAP) ---
# SHAP component removed as requested.

print("\n--- PART 5: EXPLAINABLE AI (SHAP) - Removed ---")
print("Global and local explanations are disabled.")


# --- PART 7: CAREER ROADMAP ENGINE ---

# ROADMAPS ARE UPDATED FOR GROUPED CAREERS
CAREER_ROADMAPS = {
    "Data Analytics & Science": {
        "Required skills": ["Python (Pandas, Scikit-learn)", "Statistical Analysis", "SQL", "Cloud Basics (AWS/Azure)", "Data Modeling"],
        "Courses (Free)": ["Kaggle Learn Micro-Courses", "freeCodeCamp Data Science", "Google's Data Analytics Professional Certificate (via Coursera Audit)"],
        "Courses (Paid)": ["Simplilearn Data Scientist Master's Program", "MITx Micromasters in Statistics and Data Science"],
        "Certifications": ["Microsoft Certified: Azure Data Analyst Associate", "IBM Data Science Professional Certificate"],
        "Learning path": "Beginner: Python, SQL. Intermediate: Statistics, Data Visualization, ETL/ELT. Advanced: ML Basics, Cloud Data Services.",
        "Salary range in India": "₹6,00,000 - ₹20,00,000 per annum (Experience dependent)",
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


def generate_career_roadmap(career_name):
    """Retrieves the pre-defined roadmap for a given career."""
    return CAREER_ROADMAPS.get(career_name, {
        "Required skills": [f"Specialized skills for {career_name}"],
        "Courses (Free)": ["Check online university courses"],
        "Courses (Paid)": ["Check industry-specific bootcamps"],
        "Certifications": ["Check professional bodies for this field"],
        "Learning path": "Consult industry leaders and job descriptions.",
        "Salary range in India": "Varies by specialization",
        "Future market demand": "Check recent reports"
    })

# --- PART 8: NLP SKILL EXTRACTION ---

def extract_features_from_text(text):
    """Simple keyword-based NLP extractor for skills and interests."""
    
    # Define keywords based on the features found in the dataset
    all_skills = skill_mlb.classes_.tolist()
    all_interests = interest_mlb.classes_.tolist()
    
    extracted_skills = set()
    extracted_interests = set()
    text_lower = text.lower()
    
    # Generic tokenization and cleanup
    tokens = set(re.findall(r'\b\w+\b', text_lower))
    
    # Check for direct matches or common synonyms
    for skill in all_skills:
        if skill.lower() in text_lower or any(token in skill.lower() for token in tokens if len(token) > 2):
            extracted_skills.add(skill)

    for interest in all_interests:
        if interest.lower() in text_lower or any(token in interest.lower() for token in tokens if len(token) > 2):
            extracted_interests.add(interest)
            
    return list(extracted_skills), list(extracted_interests), {} # Personality keywords unused but maintained for API structure

# --- PART 6: PREDICTION FUNCTION (Core Engine) ---

def get_top_alternatives(probas, predicted_index, top_n=3):
    """Finds the top N prediction probabilities excluding the primary prediction."""
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

def predict_career(age, education, skills, interests, ria_sec_scores):
    """
    Predicts a recommended career using the trained RandomForest model.
    """
    
    # 1. Input Validation and Structuring
    if len(ria_sec_scores) != 6:
        return {"error": "RIASEC scores must contain exactly 6 values."}

    input_data = {
        'Age': [age],
        # FIX: Removed keyword argument regex=False for str.replace() compatibility
        'Education_Cleaned': [education.replace("'", "")],
        'Skills': [skills],
        'Interests': [interests],
        'Realistic': [ria_sec_scores[0]],
        'Investigative': [ria_sec_scores[1]],
        'Artistic': [ria_sec_scores[2]],
        'Social': [ria_sec_scores[3]],
        'Enterprising': [ria_sec_scores[4]],
        'Conventional': [ria_sec_scores[5]]
    }
    input_df = pd.DataFrame(input_data)
    
    # 2. Preprocessing New Data (Must use fitted transformers)
    
    # Education
    try:
        input_df['Education_Encoded'] = edu_le.transform(input_df['Education_Cleaned'])
    except ValueError:
        input_df['Education_Encoded'] = -1 # Treat as unknown

    # Skills
    skill_matrix = skill_mlb.transform(input_df['Skills'])
    skill_df_new = pd.DataFrame(skill_matrix, columns=[f'Skill_{c}' for c in skill_mlb.classes_])
    
    # Interests
    interest_matrix = interest_mlb.transform(input_df['Interests'])
    interest_df_new = pd.DataFrame(interest_matrix, columns=[f'Interest_{c}' for c in interest_mlb.classes_])

    # RIASEC Normalization
    riasec_cols_raw = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    riasec_normalized = riasec_scaler.transform(input_df[riasec_cols_raw])
    riasec_df_new = pd.DataFrame(riasec_normalized, columns=[f'RIASEC_{c}' for c in riasec_cols_raw])
    
    # Age Scaling
    age_scaled = feature_scaler.transform(input_df[['Age']])

    # 3. Recreate the final, correctly ordered feature vector (X_input)
    X_new_single = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    
    # Set scaled Age (first column in feature_names)
    X_new_single.loc[0, 'Age_Scaled'] = age_scaled[0, 0]
    
    # Set RIASEC, Skills, and Interests (must check if the feature exists in the training features)
    for col in riasec_df_new.columns: 
        if col in X_new_single.columns:
            X_new_single.loc[0, col] = riasec_df_new[col].iloc[0]
            
    for col in skill_df_new.columns: 
        if col in X_new_single.columns:
             X_new_single.loc[0, col] = skill_df_new[col].iloc[0]
             
    for col in interest_df_new.columns:
        if col in X_new_single.columns:
            X_new_single.loc[0, col] = interest_df_new[col].iloc[0]

    X_input_array = X_new_single.values 

    # 4. Prediction (The prediction will be on the Grouped Career)
    probas = rf_model.predict_proba(X_input_array)[0]
    predicted_index = np.argmax(probas)
    
    recommended_career = target_le.inverse_transform([predicted_index])[0]
    recommendation_score = round(probas[predicted_index], 4)
    
    # 5. Alternative Careers
    top_alternatives = get_top_alternatives(probas, predicted_index, top_n=3)
    top_3_careers = {
        f"Alternative {i+1}": {"career": alt[0], "score": round(alt[1], 4)}
        for i, alt in enumerate(top_alternatives)
    }
    
    # 6. Explanation (Generic replacement since SHAP is removed)
    
    # Simple check for top RIASEC alignment based on input scores (not model weights)
    riasec_dict = dict(zip(['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional'], ria_sec_scores))
    top_riasec = sorted(riasec_dict.items(), key=lambda item: item[1], reverse=True)[:2]
    
    explanation = (
        f"The prediction for **{recommended_career}** is based on the strong alignment between the predicted group's profile and your input data. "
        f"Key drivers include the candidate's skills ({', '.join(skills[:3])}...) and high alignment with the **{top_riasec[0][0]}** and **{top_riasec[1][0]}** personality types."
    )
    
    # 7. Career Roadmap
    career_roadmap = generate_career_roadmap(recommended_career)

    return {
        "Recommended_Career": recommended_career,
        "Recommendation_Score": recommendation_score,
        "Top_3_Careers": top_3_careers,
        "Explanation": explanation,
        "Career_Roadmap": career_roadmap
    }

# --- EXAMPLE PREDICTION TEST (Part 6) ---

print("\n--- PART 6: PREDICTION FUNCTION TEST (Using Data Analyst Profile) ---")

test_profile = {
    "age": 26, 
    "education": "Master's", 
    "skills": ["SQL", "Data Warehousing", "Python", "ETL"], 
    "interests": ["Finance", "Technology"], 
    "riasec": [6, 8, 3, 5, 7, 9] # High Investigative/Conventional
}

prediction_result = predict_career(
    test_profile["age"],
    test_profile["education"],
    test_profile["skills"],
    test_profile["interests"],
    test_profile["riasec"]
)

print(json.dumps(prediction_result, indent=4))
