from datetime import datetime, timezone
from extensions import db

def _utc_now():
    return datetime.now(timezone.utc)

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    supabase_uid = db.Column(db.String(255), unique=True, nullable=True, index=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=_utc_now)

    assessments = db.relationship('Assessment', backref='user', lazy=True, cascade="all, delete-orphan")
    resumes = db.relationship('Resume', backref='user', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "supabase_uid": self.supabase_uid,
            "email": self.email,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Assessment(db.Model):
    __tablename__ = "assessments"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    education = db.Column(db.String(255), nullable=False)
    skills = db.Column(db.JSON, nullable=False)
    interests = db.Column(db.JSON, nullable=False)
    riasec_scores = db.Column(db.JSON, nullable=False)
    recommended_career = db.Column(db.String(255), nullable=False)
    recommendation_score = db.Column(db.Float, nullable=False)
    unsupervised_cluster = db.Column(db.Integer, nullable=False)
    unsupervised_career = db.Column(db.String(255), nullable=False)
    top_alternatives = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=_utc_now)

    def to_dict(self):
        return {
            "id": self.id,
            "recommended_career": self.recommended_career,
            "recommendation_score": self.recommendation_score,
            "unsupervised_cluster": self.unsupervised_cluster,
            "unsupervised_career": self.unsupervised_career,
            "top_alternatives": self.top_alternatives,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Resume(db.Model):
    __tablename__ = "resumes"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    ats_score = db.Column(db.Float, nullable=True)
    skill_gap_analysis = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=_utc_now)

    def to_dict(self):
        return {
            "id": self.id,
            "ats_score": self.ats_score,
            "skill_gap_analysis": self.skill_gap_analysis,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
