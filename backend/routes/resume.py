import uuid
from flask import Blueprint, request, jsonify, g
from auth import require_auth
from services.nlp_service import nlp_service
from services.db_service import db_service
from utils.error_handlers import ValidationError, APIError

resume_bp = Blueprint('resume', __name__)

@resume_bp.route('/api/resume/analyze', methods=['POST'], strict_slashes=False)
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

    resume_id = str(uuid.uuid4())
    
    # Save to database
    db_service.insert_resume_analysis(user["id"], resume_id, analysis, interview_prep)

    return jsonify({
        "resume_id": resume_id,
        "analysis": analysis,
        "interview_prep": interview_prep
    }), 200

