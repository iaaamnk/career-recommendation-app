from flask import Blueprint, jsonify, g
from models import Assessment, Resume
from auth import require_auth

history_bp = Blueprint('history', __name__)

@history_bp.route('/api/history', methods=['GET'])
@require_auth
def get_user_history():
    user = g.user

    assessments = Assessment.query.filter_by(user_id=user.id).order_by(Assessment.created_at.desc()).all()
    resumes = Resume.query.filter_by(user_id=user.id).order_by(Resume.created_at.desc()).all()

    return jsonify({
        "assessments": [a.to_dict() for a in assessments],
        "resumes": [r.to_dict() for r in resumes]
    }), 200
