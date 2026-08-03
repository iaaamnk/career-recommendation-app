# 🎓 PathFinder AI - Technical Interview Cheat Sheet

A concise, high-impact guide designed to help you confidently present and explain the **AI-Based Career Recommendation System & ATS Resume Scanner** in technical interviews.

---

## 🚀 1. The 30-Second Elevator Pitch

> *"PathFinder AI is a full-stack career trajectory platform that uses Machine Learning and Natural Language Processing to guide candidates. It takes user age, skills, interests, and Holland RIASEC personality scores to predict optimal career paths using Random Forest & KMeans clustering. Additionally, it features a TF-IDF powered NLP engine that scans resumes against target roles to calculate an ATS compatibility score, identify skill gaps, and generate customized interview questions."*

---

## 🏛️ 2. Clean Architecture Overview

The system follows a **Clean, Layered Modular Architecture**:

```
 ┌──────────────────────────────────────────────────────────┐
 │               Flutter Web & Cross-Platform UI            │
 └────────────────────────────┬─────────────────────────────┘
                              │ REST HTTP / JSON
 ┌────────────────────────────▼─────────────────────────────┐
 │                Flask App & Modular Blueprints            │
 │     (/health, /api/recommend, /api/resume/analyze, etc)  │
 └────────────────────────────┬─────────────────────────────┘
                              │
 ┌────────────────────────────┼─────────────────────────────┐
 │    Business Services Layer │  Authentication Middleware  │
 │  - CareerPredictorService  │  - Supabase JWT             │
 │  - ResumeAnalyzerService   │  - Firebase Admin           │
 └─────────────┬──────────────┴─────────────────────────────┘
               │
 ┌─────────────▼────────────────────────────────────────────┐
 │        Flask-SQLAlchemy Database ORM (SQLite/PostgreSQL) │
 └──────────────────────────────────────────────────────────┘
```

### Key Layers:
1. **Presentation Layer (Frontend)**: Flutter app utilizing Riverpod state management and dynamic HTTP network handling with demo fallback capabilities.
2. **API Routing Layer (`routes/`)**: Flask Blueprints (`health`, `recommendation`, `resume`, `interview`, `history`) separating concerns cleanly.
3. **Services Layer (`services/`)**: Encapsulates ML and NLP logic into reusable, thread-safe service classes (`CareerPredictorService`, `ResumeAnalyzerService`).
4. **Data Layer (`models.py`)**: SQLAlchemy ORM models (`User`, `Assessment`, `Resume`) managing database persistence.

---

## 🧠 3. Core ML & NLP Algorithms Explained

### A. Machine Learning: Career Recommendation (`services/ml_service.py`)
- **Supervised Model**: **Random Forest Classifier** (`n_estimators=300`)
  - *Why Random Forest?* High accuracy (99.5%), robust against overfitting, handles mixed categorical and scaled numerical data effortlessly.
- **Unsupervised Model**: **KMeans Clustering**
  - Group candidates into clusters to discover hidden similarities and suggest alternative career options.
- **Feature Pipeline**:
  - `Age` -> Standard Scaler
  - `Education` -> Label Encoder
  - `Skills` & `Interests` -> Multi-Label Binarizer (One-Hot Encoded)
  - `RIASEC Scores` -> Imputed & Standard Scaled based on Holland Occupational Codes.

### B. Natural Language Processing: ATS Resume Analysis (`services/nlp_service.py`)
- **Text Vectorization**: **TF-IDF (Term Frequency-Inverse Document Frequency)**
  - Converts raw resume text and career skill descriptions into 500-dimensional numerical vectors.
- **Similarity Metric**: **Cosine Similarity**
  - Measures the cosine angle between the resume TF-IDF vector and the target career vector to compute a 0–100% **ATS Match Score**.
- **Skill Gap & Interview Generation**:
  - Regex pattern matching isolates missing technical keywords.
  - Template interpolation generates tailored behavioral and technical interview questions based on missing skills.

---

## 📊 4. Key Database Schema

| Table Name | Model Class | Key Fields | Purpose |
| :--- | :--- | :--- | :--- |
| **`users`** | `User` | `id`, `email`, `supabase_uid`, `firebase_uid`, `created_at` | Manages authenticated and guest users |
| **`assessments`** | `Assessment` | `id`, `user_id`, `skills`, `riasec_scores`, `recommended_career`, `recommendation_score` | Stores career quiz results and predictions |
| **`resumes`** | `Resume` | `id`, `user_id`, `ats_score`, `skill_gap_analysis`, `created_at` | Stores ATS scan results and skill gap analysis |

---

## 💬 5. Frequently Asked Interview Questions & Answers

### Q1: Why did you choose Flask for the backend over Django?
> *"Flask is lightweight, modular, and explicit. Since our core requirements center around RESTful AI/ML microservices rather than server-rendered templates or heavy built-in Django admin panels, Flask allows us to maintain a clean application factory pattern with low overhead and faster startup times."*

### Q2: How does the career recommendation engine handle new or sparse user inputs?
> *"We use Multi-Label Binarization for missing or varying skill lists, and default feature alignment (`reindex`) so unmentioned skills are safely zero-filled. For RIASEC personality scores, if a user hasn't completed full psychological profiling, we impute scores using baseline occupational profiles."*

### Q3: How is the ATS score calculated?
> *"We fit a TF-IDF vectorizer over curated career skill corpuses. When a user uploads a resume, we compute the Cosine Similarity between the TF-IDF vector of their resume and that of the target career. The cosine similarity (range 0 to 1) is scaled into an intuitive 0–100 ATS match score."*

### Q4: How do you handle authentication in a multi-provider setup?
> *"Our `@require_auth` decorator checks Bearer tokens against Supabase JWT verification and Firebase Admin ID token verification sequentially. In development or demo mode, it seamlessly provides a guest fallback so the system remains 100% testable without external service lock-in."*

---

## ⚡ 6. Quick Terminal Commands to Remember

```bash
# Run Flask Backend locally
cd backend
venv/bin/python app.py

# Run Backend Unit Tests
cd backend
venv/bin/python -m unittest discover -s tests

# Run Flutter Web Frontend
cd frontend_web
flutter run -d chrome
```
