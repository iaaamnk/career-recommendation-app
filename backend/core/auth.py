import os
import json
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from .models import User
from django.conf import settings

# Initialize Firebase Admin
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cred_path = os.path.join(current_dir, "serviceAccountKey.json")

try:
    if not firebase_admin._apps:
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
        elif "FIREBASE_SERVICE_ACCOUNT_KEY" in os.environ:
            cert_dict = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT_KEY"])
            cred = credentials.Certificate(cert_dict)
        else:
            raise Exception("No Firebase credentials found in serviceAccountKey.json or FIREBASE_SERVICE_ACCOUNT_KEY")
        firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Warning: Failed to initialize Firebase Admin: {e}")

class FirebaseAuthentication(BaseAuthentication):
    def authenticate(self, request):
        auth_header = request.META.get("HTTP_AUTHORIZATION")
        if not auth_header:
            return None

        parts = auth_header.split()
        if parts[0].lower() != "bearer":
            return None
        elif len(parts) == 1:
            raise AuthenticationFailed("Invalid token header. No credentials provided.")
        elif len(parts) > 2:
            raise AuthenticationFailed("Invalid token header. Token string should not contain spaces.")

        token = parts[1]

        try:
            decoded_token = firebase_auth.verify_id_token(token)
        except Exception as e:
            raise AuthenticationFailed(f"Invalid authentication credentials: {str(e)}")

        uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        name = decoded_token.get("name")

        # Find or create user
        user, created = User.objects.get_or_create(
            firebase_uid=uid,
            defaults={"email": email, "name": name}
        )
        
        return (user, token)
