from flask import Blueprint, jsonify, g
from auth import require_auth

history_bp = Blueprint('history', __name__)

@history_bp.route('/api/history', methods=['GET'], strict_slashes=False)
@require_auth
def get_user_history():
    user = g.user

    return jsonify({
        "assessments": [],
        "resumes": []
    }), 200

