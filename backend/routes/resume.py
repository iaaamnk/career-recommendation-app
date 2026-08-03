from flask import Blueprint, request, jsonify, g
from extensions import db
from models import Resume
from auth import require_auth
from services.nlp_service import nlp_service
from utils.error_handlers import ValidationError, APIError

resume_bp = Blueprint('resume', __name__)

@resume_bp.route('/api/resume/analyze', methods=['POST'])
@require_auth
def analyze_resume():
    user = g.user
    data = request.get_json() or {}

    resume_text = data.get("resume_text", "")
    target_career = data.get("target_career", "")

    if not resume_text or not target_career:
        raise ValidationError("resume_text and target_career are required")

    try:
        analysis = nlp_service.analyze_resume_text(resume_text, target_career)
        interview_prep = nlp_service.generate_interview_prep(target_career, analysis["skills_missing"])
    except Exception as e:
        raise APIError(f"Analysis error: {str(e)}", status_code=500)

    resume = Resume(
        user_id=user.id,
        file_path="simulated_path.pdf",
        ats_score=analysis["ats_score"],
        skill_gap_analysis={"missing": analysis["skills_missing"], "found": analysis["skills_found"]}
    )

    db.session.add(resume)
    db.session.commit()

    return jsonify({
        "resume_id": resume.id,
        "analysis": analysis,
        "interview_prep": interview_prep
    }), 200
