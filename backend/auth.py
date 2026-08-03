import os
import jwt
from functools import wraps
from flask import request, jsonify, g
from extensions import db
from models import User

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
        
        # Try Supabase Auth first, then Firebase Auth, then Dev/Demo fallback
        user = _authenticate_supabase(token)
        if not user:
            user = _authenticate_firebase(token)
        if not user:
            user = _authenticate_demo(token)
            
        if not user:
            return jsonify({"detail": "Invalid or expired authentication token."}), 401
            
        g.user = user
        return f(*args, **kwargs)
        
    return decorated

def _authenticate_demo(token):
    try:
        user = User.query.filter_by(email="demo@pathfinder.ai").first()
        if not user:
            user = User(
                supabase_uid="demo-user-uid-123",
                email="demo@pathfinder.ai",
                name="Demo User"
            )
            db.session.add(user)
            db.session.commit()
        return user
    except Exception:
        return None

def _authenticate_supabase(token):
    try:
        secret = os.environ.get('SUPABASE_JWT_SECRET')
        if secret:
            payload = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
        else:
            payload = jwt.decode(token, options={"verify_signature": False})
            
        uid = payload.get("sub")
        if not uid:
            return None
            
        email = payload.get("email", "")
        user_metadata = payload.get("user_metadata") or {}
        name = user_metadata.get("name") or user_metadata.get("full_name") or (email.split("@")[0] if email else "")
        
        user = None
        if uid:
            user = User.query.filter_by(supabase_uid=uid).first()
        if not user and email:
            user = User.query.filter_by(email=email).first()
            
        if user:
            updated = False
            if not user.supabase_uid:
                user.supabase_uid = uid
                updated = True
            if email and user.email != email:
                user.email = email
                updated = True
            if name and user.name != name:
                user.name = name
                updated = True
            if updated:
                db.session.commit()
        else:
            user = User(
                supabase_uid=uid,
                email=email or f"{uid}@supabase.local",
                name=name
            )
            db.session.add(user)
            db.session.commit()
            
        return user
    except Exception as e:
        return None

def _authenticate_firebase(token):
    try:
        from firebase_admin import auth as firebase_auth
        decoded_token = firebase_auth.verify_id_token(token)
        uid = decoded_token.get("uid")
        if not uid:
            return None
            
        email = decoded_token.get("email", "")
        name = decoded_token.get("name", "")
        
        user = User.query.filter_by(firebase_uid=uid).first()
        if not user and email:
            user = User.query.filter_by(email=email).first()
            
        if user:
            if not user.firebase_uid:
                user.firebase_uid = uid
            if email and user.email != email:
                user.email = email
            if name and user.name != name:
                user.name = name
            db.session.commit()
        else:
            user = User(
                firebase_uid=uid,
                email=email or f"{uid}@firebase.local",
                name=name
            )
            db.session.add(user)
            db.session.commit()
            
        return user
    except Exception as e:
        return None
