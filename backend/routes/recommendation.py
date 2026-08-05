import uuid
from flask import Blueprint, request, jsonify, g
from auth import require_auth
from services.ml_service import ml_service
from services.db_service import db_service
from utils.error_handlers import ValidationError, APIError

recommendation_bp = Blueprint('recommendation', __name__)

@recommendation_bp.route('/api/recommend', methods=['POST'], strict_slashes=False)
@require_auth
def recommend_career():
    user = g.user
    data = request.get_json() or {}

    try:
        age = int(data.get("age", 24))
        education = str(data.get("education", ""))
        skills = list(data.get("skills", []))
        interests = list(data.get("interests", []))
        riasec_scores = [float(s) for s in data.get("riasec_scores", [])]
    except Exception as e:
        raise ValidationError(f"Invalid request data: {str(e)}")

    try:
        prediction = ml_service.predict(
            age=age,
            education=education,
            skills=skills,
            interests=interests,
            riasec=riasec_scores
        )
    except Exception as e:
        raise APIError(f"Prediction error: {str(e)}", status_code=500)

    assessment_id = str(uuid.uuid4())
    
    # Save to database
    db_service.insert_assessment(user["id"], assessment_id, prediction)

    return jsonify({
        "assessment_id": assessment_id,
        "prediction": prediction
    }), 200

