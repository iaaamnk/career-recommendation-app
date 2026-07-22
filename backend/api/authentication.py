from rest_framework import authentication
from rest_framework import exceptions
from firebase_admin import auth as firebase_auth
from .models import User

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
