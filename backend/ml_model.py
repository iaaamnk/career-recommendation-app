import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- CONFIG ----------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
DATA_FILE_PATH = "../AI-based Career Recommendation System.csv"

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

def load_and_train_model():
    global rf_model, edu_le, skill_mlb, interest_mlb, target_le
    global riasec_scaler, feature_scaler, feature_names
    global kmeans_model, cluster_career_map

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, DATA_FILE_PATH)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return False

    df = pd.read_csv(csv_path)

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
        cluster_data = df[df['Cluster'] == c]
        if not cluster_data.empty:
            label = cluster_data['Recommended_Career'].mode()[0]
            cluster_career_map[c] = label
        else:
            cluster_career_map[c] = "Unknown"

    print("Model trained successfully.")
    return True

def predict_career(age: int, education: str, skills: list, interests: list, riasec: list):
    if rf_model is None:
        raise ValueError("Model not loaded.")

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
    X_new['Age_Scaled'] = feature_scaler.transform(input_df[['Age']])
    
    # Realign columns
    X_new = X_new.reindex(columns=feature_names, fill_value=0).values

    probas = rf_model.predict_proba(X_new)[0]
    idx = np.argmax(probas)
    recommended_career = target_le.inverse_transform([idx])[0]
    confidence = float(probas[idx])

    cluster_id, unsup_career = unsupervised_recommendation(X_new)

    top_alternatives = get_top_alternatives(probas, idx, top_n=3)
    top_3_careers = [{"career": alt[0], "score": float(alt[1])} for alt in top_alternatives]
    
    return {
        "Recommended_Career": recommended_career,
        "Recommendation_Score": confidence,
        "Unsupervised_Cluster": int(cluster_id),
        "Unsupervised_Recommendation": unsup_career,
        "Top_3_Careers": top_3_careers
    }
