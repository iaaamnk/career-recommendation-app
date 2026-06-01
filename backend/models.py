from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String, unique=True, index=True, nullable=True)
    email = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    assessments = relationship("Assessment", back_populates="user")
    resumes = relationship("Resume", back_populates="user")

class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Input Features
    age = Column(Integer)
    education = Column(String)
    skills = Column(JSON)  # List of skills
    interests = Column(JSON) # List of interests
    riasec_scores = Column(JSON) # [R, I, A, S, E, C]
    
    # ML Results
    recommended_career = Column(String)
    recommendation_score = Column(Float)
    unsupervised_cluster = Column(Integer)
    unsupervised_career = Column(String)
    top_alternatives = Column(JSON) # List of dicts
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="assessments")

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    file_path = Column(String)
    ats_score = Column(Float, nullable=True)
    skill_gap_analysis = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="resumes")
