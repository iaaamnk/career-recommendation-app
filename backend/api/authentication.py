import os
import jwt
from rest_framework import authentication
from rest_framework import exceptions
from .models import User

class SupabaseAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return None
        
        parts = auth_header.split()
        if parts[0].lower() != 'bearer' or len(parts) != 2:
            return None
            
        token = parts[1]
        try:
            secret = os.environ.get('SUPABASE_JWT_SECRET')
            if secret:
                payload = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
            else:
                # In development mode without secret, decode unverified
                payload = jwt.decode(token, options={"verify_signature": False})
                
            uid = payload.get("sub")
            if not uid:
                raise exceptions.AuthenticationFailed("Invalid token payload: missing 'sub' claim")
                
            email = payload.get("email", "")
            user_metadata = payload.get("user_metadata") or {}
            name = user_metadata.get("name") or user_metadata.get("full_name") or (email.split("@")[0] if email else "")
            
            user = None
            if uid:
                user = User.objects.filter(supabase_uid=uid).first()
            if not user and email:
                user = User.objects.filter(email=email).first()
                
            if user:
                if not user.supabase_uid:
                    user.supabase_uid = uid
                if email and user.email != email:
                    user.email = email
                if name and user.name != name:
                    user.name = name
                user.save()
            else:
                user = User.objects.create(
                    supabase_uid=uid,
                    email=email or f"{uid}@supabase.local",
                    name=name
                )
                
            return (user, token)
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed("Supabase token has expired")
        except jwt.InvalidTokenError as e:
            raise exceptions.AuthenticationFailed(f"Invalid Supabase token: {str(e)}")
        except Exception as e:
            raise exceptions.AuthenticationFailed(f"Authentication failed: {str(e)}")

class FirebaseAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return None
        
        parts = auth_header.split()
        if parts[0].lower() != 'bearer' or len(parts) != 2:
            return None
            
        token = parts[1]
        try:
            from firebase_admin import auth as firebase_auth
            decoded_token = firebase_auth.verify_id_token(token)
            uid = decoded_token.get("uid")
            email = decoded_token.get("email")
            name = decoded_token.get("name")
            
            user, created = User.objects.get_or_create(
                firebase_uid=uid,
                defaults={'email': email or '', 'name': name or ''}
            )
            if not created and (user.email != email or user.name != name):
                user.email = email or ''
                user.name = name or ''
                user.save()
                
            return (user, token)
        except Exception as e:
            raise exceptions.AuthenticationFailed(f"Invalid authentication credentials: {str(e)}")

