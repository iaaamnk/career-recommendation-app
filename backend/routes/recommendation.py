from flask import Blueprint, request, jsonify, g
from extensions import db
from models import Assessment
from auth import require_auth
from services.ml_service import ml_service
from utils.error_handlers import ValidationError, APIError

recommendation_bp = Blueprint('recommendation', __name__)

@recommendation_bp.route('/api/recommend', methods=['POST'])
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

    assessment = Assessment(
        user_id=user.id,
        age=age,
        education=education,
        skills=skills,
        interests=interests,
        riasec_scores=riasec_scores,
        recommended_career=prediction["Recommended_Career"],
        recommendation_score=prediction["Recommendation_Score"],
        unsupervised_cluster=prediction["Unsupervised_Cluster"],
        unsupervised_career=prediction["Unsupervised_Recommendation"],
        top_alternatives=prediction["Top_3_Careers"]
    )

    db.session.add(assessment)
    db.session.commit()

    return jsonify({
        "assessment_id": assessment.id,
        "prediction": prediction
    }), 200
