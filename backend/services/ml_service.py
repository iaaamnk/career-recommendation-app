import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

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

class CareerPredictorService:
    def __init__(self):
        self.rf_model = None
        self.edu_le = None
        self.skill_mlb = None
        self.interest_mlb = None
        self.target_le = None
        self.riasec_scaler = None
        self.feature_scaler = None
        self.feature_names = None
        self.kmeans_model = None
        self.cluster_career_map = {}
        self.is_loaded = False

    @staticmethod
    def _get_imputed_riasec(career, code):
        base = CAREER_PROFILES.get(career, CAREER_PROFILES["Default"]).get(code, 5)
        return max(0, min(10, base + random.randint(-1, 1)))

    def load_and_train(self, csv_file_path=None):
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        if not csv_file_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            csv_file_path = os.path.join(base_dir, "AI-based Career Recommendation System.csv")

        if not os.path.exists(csv_file_path):
            print(f"Error: dataset CSV not found at {csv_file_path}")
            return False

        df = pd.read_csv(csv_file_path)

        df['Skills'] = df['Skills'].fillna('').apply(lambda x: [i.strip() for i in str(x).split(';') if i])
        df['Interests'] = df['Interests'].fillna('').apply(lambda x: [i.strip() for i in str(x).split(';') if i])
        df['Recommended_Career'] = df['Recommended_Career'].map(CAREER_GROUP_MAPPING).fillna('Default')

        riasec_codes = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
        for code in riasec_codes:
            df[code] = df['Recommended_Career'].apply(lambda c: self._get_imputed_riasec(c, code[0]))

        self.edu_le = LabelEncoder()
        self.skill_mlb = MultiLabelBinarizer()
        self.interest_mlb = MultiLabelBinarizer()
        self.target_le = LabelEncoder()
        self.riasec_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()

        df['Education_Cleaned'] = df['Education'].str.replace("'", "")
        df['Education_Encoded'] = self.edu_le.fit_transform(df['Education_Cleaned'])

        skill_df = pd.DataFrame(self.skill_mlb.fit_transform(df['Skills']),
                                columns=[f"Skill_{c}" for c in self.skill_mlb.classes_])
        interest_df = pd.DataFrame(self.interest_mlb.fit_transform(df['Interests']),
                                   columns=[f"Interest_{c}" for c in self.interest_mlb.classes_])

        riasec_df = pd.DataFrame(
            self.riasec_scaler.fit_transform(df[riasec_codes]),
            columns=[f"RIASEC_{c}" for c in riasec_codes]
        )

        X = pd.concat([df[['Age']], riasec_df, skill_df, interest_df], axis=1)
        X['Age_Scaled'] = self.feature_scaler.fit_transform(X[['Age']])
        X = X.drop(columns=['Age'])

        self.feature_names = X.columns.tolist()
        X_values = X.values
        y = self.target_le.fit_transform(df['Recommended_Career'])

        # Train main model
        self.rf_model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, class_weight='balanced')
        self.rf_model.fit(X_values, y)

        # Train KMeans
        self.kmeans_model = KMeans(n_clusters=len(self.target_le.classes_), random_state=RANDOM_SEED, n_init=10)
        clusters = self.kmeans_model.fit_predict(X_values)

        self.cluster_career_map = {}
        df['Cluster'] = clusters
        for c in np.unique(clusters):
            cluster_data = df[df['Cluster'] == c]
            if not cluster_data.empty:
                label = cluster_data['Recommended_Career'].mode()[0]
                self.cluster_career_map[c] = label
            else:
                self.cluster_career_map[c] = "Unknown"

        self.is_loaded = True
        print("CareerPredictorService trained and ready.")
        return True

    def predict(self, age: int, education: str, skills: list, interests: list, riasec: list):
        if not self.is_loaded or self.rf_model is None:
            raise ValueError("ML Predictor Service is not initialized.")

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

        skill_df = pd.DataFrame(self.skill_mlb.transform(input_df['Skills']),
                                columns=[f"Skill_{c}" for c in self.skill_mlb.classes_])
        interest_df = pd.DataFrame(self.interest_mlb.transform(input_df['Interests']),
                                   columns=[f"Interest_{c}" for c in self.interest_mlb.classes_])

        riasec_cols = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
        riasec_df = pd.DataFrame(
            self.riasec_scaler.transform(input_df[riasec_cols]),
            columns=[f"RIASEC_{c}" for c in riasec_cols]
        )

        X_new = pd.concat([riasec_df, skill_df, interest_df], axis=1)
        X_new['Age_Scaled'] = self.feature_scaler.transform(input_df[['Age']])
        X_new = X_new.reindex(columns=self.feature_names, fill_value=0).values

        probas = self.rf_model.predict_proba(X_new)[0]
        idx = int(np.argmax(probas))
        recommended_career = self.target_le.inverse_transform([idx])[0]
        confidence = float(probas[idx])

        cluster_id = int(self.kmeans_model.predict(X_new)[0])
        unsup_career = self.cluster_career_map.get(cluster_id, "Unknown")

        top_alternatives = self._get_top_alternatives(probas, idx, top_n=3)

        return {
            "Recommended_Career": recommended_career,
            "Recommendation_Score": confidence,
            "Unsupervised_Cluster": cluster_id,
            "Unsupervised_Recommendation": unsup_career,
            "Top_3_Careers": top_alternatives
        }

    def _get_top_alternatives(self, probas, predicted_index, top_n=3):
        sorted_indices = np.argsort(probas)[::-1]
        alternatives = []
        count = 0
        for idx in sorted_indices:
            if idx != predicted_index and count < top_n:
                career = self.target_le.inverse_transform([idx])[0]
                score = float(probas[idx])
                alternatives.append({"career": career, "score": score})
                count += 1
        return alternatives

ml_service = CareerPredictorService()
