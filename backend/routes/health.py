from flask import Blueprint, jsonify
from services.ml_service import ml_service

health_bp = Blueprint('health', __name__)

@health_bp.route('/', methods=['GET'], strict_slashes=False)
def index():
    return jsonify({
        "status": "ok",
        "message": "PathFinder AI API Service is running.",
        "model_loaded": ml_service.is_loaded and ml_service.rf_model is not None,
        "endpoints": {
            "health": "/health",
            "recommend": "/api/recommend",
            "resume_analyze": "/api/resume/analyze",
            "interview_prep": "/api/interview/prep",
            "history": "/api/history"
        }
    }), 200

@health_bp.route('/health', methods=['GET'], strict_slashes=False)
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": ml_service.is_loaded and ml_service.rf_model is not None
    }), 200
