import unittest
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from extensions import db
from models import User

class TestFlaskAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app(config_name='testing')
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()
            self.test_user = User(
                supabase_uid="test-user-uid-123",
                email="testuser@example.com",
                name="Test User"
            )
            db.session.add(self.test_user)
            db.session.commit()

            import jwt
            self.auth_headers = {
                "Authorization": f"Bearer {jwt.encode({'sub': 'test-user-uid-123', 'email': 'testuser@example.com'}, 'secret', algorithm='HS256')}"
            }

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_health_check(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "ok")
        self.assertIn("model_loaded", data)

    def test_interview_prep(self):
        payload = {
            "target_career": "Data Scientist",
            "missing_skills": ["TensorFlow", "Docker"]
        }
        response = self.client.post(
            '/api/interview/prep',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("interview_questions", data)
        self.assertIn("tips", data)
        self.assertIn("roadmap_url", data)

    def test_interview_prep_missing_target(self):
        response = self.client.post(
            '/api/interview/prep',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_recommend_career(self):
        payload = {
            "age": 25,
            "education": "Bachelor's",
            "skills": ["Python", "SQL", "Pandas"],
            "interests": ["Data Analysis", "Machine Learning"],
            "riasec_scores": [7, 9, 4, 5, 6, 8]
        }
        response = self.client.post(
            '/api/recommend',
            data=json.dumps(payload),
            content_type='application/json',
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("assessment_id", data)
        self.assertIn("prediction", data)
        self.assertIn("Recommended_Career", data["prediction"])

    def test_analyze_resume(self):
        payload = {
            "resume_text": "Experienced Python data analyst proficient in SQL, Pandas, Tableau, and Machine Learning.",
            "target_career": "Data Analyst"
        }
        response = self.client.post(
            '/api/resume/analyze',
            data=json.dumps(payload),
            content_type='application/json',
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("resume_id", data)
        self.assertIn("analysis", data)
        self.assertIn("interview_prep", data)

    def test_get_history(self):
        response = self.client.get(
            '/api/history',
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("assessments", data)
        self.assertIn("resumes", data)

if __name__ == '__main__':
    unittest.main()
