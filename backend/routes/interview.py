from flask import Blueprint, request, jsonify
from services.nlp_service import nlp_service
from utils.error_handlers import ValidationError, APIError

interview_bp = Blueprint('interview', __name__)

@interview_bp.route('/api/interview/prep', methods=['POST'])
def get_interview_prep():
    data = request.get_json() or {}
    target_career = data.get("target_career", "")
    missing_skills = list(data.get("missing_skills", []))

    if not target_career:
        raise ValidationError("target_career is required")

    try:
        prep = nlp_service.generate_interview_prep(target_career, missing_skills)
        return jsonify(prep), 200
    except Exception as e:
        raise APIError(f"Interview prep error: {str(e)}", status_code=500)
