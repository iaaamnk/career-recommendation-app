import os
from supabase import create_client, Client

class DBService:
    def __init__(self):
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_ANON_KEY")
        self.client: Client = None
        
        # In-memory storage for demo user
        self.demo_assessments = []
        self.demo_resumes = []
        
        if self.supabase_url and self.supabase_key:
            self.client = create_client(self.supabase_url, self.supabase_key)
            
    def insert_assessment(self, user_id: str, assessment_id: str, prediction: dict):
        data = {
            "id": assessment_id,
            "user_id": user_id,
            "prediction_data": prediction,
        }
        
        if user_id == "demo-user-id":
            import datetime
            data["created_at"] = datetime.datetime.utcnow().isoformat()
            self.demo_assessments.insert(0, data)
            return [data]
            
        if not self.client:
            return None
            
        try:
            response = self.client.table("assessments").insert(data).execute()
            return response.data
        except Exception as e:
            print(f"Error inserting assessment: {e}")
            return None
            
    def insert_resume_analysis(self, user_id: str, resume_id: str, analysis: dict, interview_prep: dict):
        data = {
            "id": resume_id,
            "user_id": user_id,
            "analysis_data": analysis,
            "interview_prep": interview_prep,
        }
        
        if user_id == "demo-user-id":
            import datetime
            data["created_at"] = datetime.datetime.utcnow().isoformat()
            self.demo_resumes.insert(0, data)
            return [data]
            
        if not self.client:
            return None
            
        try:
            response = self.client.table("resume_analyses").insert(data).execute()
            return response.data
        except Exception as e:
            print(f"Error inserting resume analysis: {e}")
            return None
            
    def get_user_history(self, user_id: str):
        if user_id == "demo-user-id":
            return {
                "assessments": self.demo_assessments,
                "resumes": self.demo_resumes
            }
            
        if not self.client:
            return {"assessments": [], "resumes": []}
            
        assessments = []
        resumes = []
        
        try:
            a_res = self.client.table("assessments").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            assessments = a_res.data
        except Exception as e:
            print(f"Error fetching assessments: {e}")
            
        try:
            r_res = self.client.table("resume_analyses").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            resumes = r_res.data
        except Exception as e:
            print(f"Error fetching resume analyses: {e}")
            
        return {
            "assessments": assessments,
            "resumes": resumes
        }

db_service = DBService()
