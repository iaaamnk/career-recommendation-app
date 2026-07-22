import os
import sys
import json
from django.apps import AppConfig
import firebase_admin
from firebase_admin import credentials

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        # Prevent training and initialization twice during Django management commands
        if 'manage.py' in sys.argv and any(cmd in sys.argv for cmd in ['migrate', 'makemigrations', 'check', 'shell', 'collectstatic']):
            return

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cred_path = os.path.join(base_dir, "serviceAccountKey.json")

        # Initialize Firebase Admin
        try:
            if not firebase_admin._apps:
                if os.path.exists(cred_path):
                    cred = credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(cred)
                elif "FIREBASE_SERVICE_ACCOUNT_KEY" in os.environ:
                    cert_dict = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT_KEY"])
                    cred = credentials.Certificate(cert_dict)
                    firebase_admin.initialize_app(cred)
                else:
                    print("Warning: No Firebase credentials found in serviceAccountKey.json or FIREBASE_SERVICE_ACCOUNT_KEY")
        except Exception as e:
            print(f"Warning: Failed to initialize Firebase Admin: {e}")

        # Initialize/Load ML Model
        try:
            import ml_model
            print("Loading and training ML model on startup...")
            ml_model.load_and_train_model()
        except Exception as e:
            print(f"Error loading ML model on startup: {e}")
