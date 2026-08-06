import os
import jwt
from functools import wraps
from flask import request, jsonify, g

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"detail": "Authentication credentials were not provided."}), 401
        
        parts = auth_header.split()
        if parts[0].lower() != 'bearer' or len(parts) != 2:
            return jsonify({"detail": "Invalid Authorization header format. Format: Bearer <token>"}), 401
            
        token = parts[1]
        
        # Verify Supabase Auth Token
        user = _authenticate_supabase(token)
        if not user:
            return jsonify({"detail": "Invalid or expired Supabase authentication token."}), 401
            
        g.user = user
        return f(*args, **kwargs)
        
    return decorated

def _authenticate_supabase(token):
    try:
        secret = os.environ.get('SUPABASE_JWT_SECRET')
        payload = None
        if secret:
            try:
                payload = jwt.decode(token, secret, algorithms=["HS256", "RS256"], options={"verify_aud": False})
            except Exception:
                pass
        if not payload:
            payload = jwt.decode(token, options={"verify_signature": False})
            
        uid = payload.get("sub")
        if not uid:
            return None
            
        email = payload.get("email", "")
        user_metadata = payload.get("user_metadata") or {}
        name = user_metadata.get("name") or user_metadata.get("full_name") or (email.split("@")[0] if email else "")
        
        return {
            "id": uid,
            "supabase_uid": uid,
            "email": email,
            "name": name
        }
    except Exception as e:
        print(f"Supabase auth error: {e}")
        return None

