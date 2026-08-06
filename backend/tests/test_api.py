import unittest
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

class TestFlaskAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app(config_name='testing')
        self.client = self.app.test_client()

        import jwt
        self.auth_headers = {
            "Authorization": f"Bearer {jwt.encode({'sub': 'test-user-uid-123', 'email': 'testuser@example.com'}, 'secret', algorithm='HS256')}"
        }

    def tearDown(self):
        pass

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

    def test_user_history_isolation(self):
        import jwt
        token_a = jwt.encode({'sub': 'user-a-uid-123', 'email': 'usera@example.com'}, 'secret', algorithm='HS256')
        token_b = jwt.encode({'sub': 'user-b-uid-456', 'email': 'userb@example.com'}, 'secret', algorithm='HS256')
        headers_user_a = {"Authorization": f"Bearer {token_a}"}
        headers_user_b = {"Authorization": f"Bearer {token_b}"}

        # User A performs resume analysis
        res_a = self.client.post(
            '/api/resume/analyze',
            data=json.dumps({
                "resume_text": "User A Python developer resume",
                "target_career": "Software Engineer"
            }),
            content_type='application/json',
            headers=headers_user_a
        )
        self.assertEqual(res_a.status_code, 200)

        # User B fetches history
        hist_b = self.client.get('/api/history', headers=headers_user_b)
        self.assertEqual(hist_b.status_code, 200)
        data_b = json.loads(hist_b.data)
        # User B should NOT see User A's resume analysis
        self.assertEqual(len(data_b["resumes"]), 0)

        # User A fetches history
        hist_a = self.client.get('/api/history', headers=headers_user_a)
        self.assertEqual(hist_a.status_code, 200)
        data_a = json.loads(hist_a.data)
        # User A SHOULD see User A's resume analysis
        self.assertEqual(len(data_a["resumes"]), 1)

if __name__ == '__main__':
    unittest.main()
