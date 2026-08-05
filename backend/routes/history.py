from flask import Blueprint, jsonify, g
from auth import require_auth
from services.db_service import db_service

history_bp = Blueprint('history', __name__)

@history_bp.route('/api/history', methods=['GET'], strict_slashes=False)
@require_auth
def get_user_history():
    user = g.user
    
    history_data = db_service.get_user_history(user["id"])

    return jsonify(history_data), 200

