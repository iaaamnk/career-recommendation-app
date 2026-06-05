from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from database import engine, get_db, Base
import models
import ml_model
from auth import verify_firebase_token

# Create DB tables (wrapped in try-except so it doesn't crash on boot if Hive is unavailable)
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Warning: Could not connect to database on startup: {e}")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PathFinder API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attempt to load model on startup
@app.on_event("startup")
def load_ml_model():
    ml_model.load_and_train_model()

class AssessmentRequest(BaseModel):
    age: int
    education: str
    skills: List[str]
    interests: List[str]
    riasec_scores: List[float]

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": ml_model.rf_model is not None}

@app.post("/api/recommend")
def recommend_career(request: AssessmentRequest, db: Session = Depends(get_db), user: models.User = Depends(verify_firebase_token)):
    # 2. Get predictions
    try:
        prediction = ml_model.predict_career(
            age=request.age,
            education=request.education,
            skills=request.skills,
            interests=request.interests,
            riasec=request.riasec_scores
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # 3. Save to database
    db_assessment = models.Assessment(
        user_id=user.id,
        age=request.age,
        education=request.education,
        skills=request.skills,
        interests=request.interests,
        riasec_scores=request.riasec_scores,
        recommended_career=prediction["Recommended_Career"],
        recommendation_score=prediction["Recommendation_Score"],
        unsupervised_cluster=prediction["Unsupervised_Cluster"],
        unsupervised_career=prediction["Unsupervised_Recommendation"],
        top_alternatives=prediction["Top_3_Careers"]
    )
    db.add(db_assessment)
    db.commit()
    db.refresh(db_assessment)

    return {
        "assessment_id": db_assessment.id,
        "prediction": prediction
    }

class ResumeAnalyzeRequest(BaseModel):
    resume_text: str
    target_career: str

@app.post("/api/resume/analyze")
def analyze_resume(request: ResumeAnalyzeRequest, db: Session = Depends(get_db), user: models.User = Depends(verify_firebase_token)):
    import nlp_model
    
    analysis = nlp_model.analyze_resume_text(request.resume_text, request.target_career)
    interview_prep = nlp_model.generate_interview_prep(request.target_career, analysis["skills_missing"])
    
    db_resume = models.Resume(
        user_id=user.id,
        file_path="simulated_path.pdf",
        ats_score=analysis["ats_score"],
        skill_gap_analysis={"missing": analysis["skills_missing"], "found": analysis["skills_found"]}
    )
    db.add(db_resume)
    db.commit()
    db.refresh(db_resume)
    
    return {
        "resume_id": db_resume.id,
        "analysis": analysis,
        "interview_prep": interview_prep
    }

class InterviewPrepRequest(BaseModel):
    target_career: str
    missing_skills: List[str]

@app.post("/api/interview/prep")
def get_interview_prep(request: InterviewPrepRequest):
    import nlp_model
    prep = nlp_model.generate_interview_prep(request.target_career, request.missing_skills)
    return prep
