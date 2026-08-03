from flask import Blueprint, jsonify
from services.ml_service import ml_service

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": ml_service.is_loaded and ml_service.rf_model is not None
    }), 200
