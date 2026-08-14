import os
from supabase import create_client, Client

class DBService:
    def __init__(self):
        self.supabase_url = os.environ.get("SUPABASE_URL")
        # Prefer service role key to bypass RLS if available, otherwise anon
        self.supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
        self.client: Client = None
        
        # In-memory storage fallback for all users: user_id -> list
        self.memory_assessments = {}
        self.memory_resumes = {}
        
        if self.supabase_url and self.supabase_key:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
            except Exception as e:
                print(f"Failed to initialize Supabase client: {e}")

    def insert_assessment(self, user_id: str, assessment_id: str, prediction: dict):
        data = {
            "id": assessment_id,
            "user_id": user_id,
            "prediction_data": prediction,
        }
        
        from datetime import datetime, timezone
        data["created_at"] = datetime.now(timezone.utc).isoformat()
        
        success = False
        if self.client:
            try:
                response = self.client.table("assessments").insert(data).execute()
                success = True
            except Exception as e:
                print(f"Error inserting assessment to Supabase: {e}")
        
        # Fallback to in-memory if DB fails or doesn't exist
        if not success:
            if user_id not in self.memory_assessments:
                self.memory_assessments[user_id] = []
            self.memory_assessments[user_id].insert(0, data)
            
        return [data]
            
    def insert_resume_analysis(self, user_id: str, resume_id: str, analysis: dict, interview_prep: dict):
        data = {
            "id": resume_id,
            "user_id": user_id,
            "analysis_data": analysis,
            "interview_prep": interview_prep,
        }
        
        from datetime import datetime, timezone
        data["created_at"] = datetime.now(timezone.utc).isoformat()
        
        success = False
        if self.client:
            try:
                response = self.client.table("resume_analyses").insert(data).execute()
                success = True
            except Exception as e:
                print(f"Error inserting resume analysis to Supabase: {e}")
                
        # Fallback to in-memory if DB fails or doesn't exist
        if not success:
            if user_id not in self.memory_resumes:
                self.memory_resumes[user_id] = []
            self.memory_resumes[user_id].insert(0, data)
            
        return [data]
            
    def get_user_history(self, user_id: str):
        assessments = []
        resumes = []
        
        if self.client:
            try:
                a_res = self.client.table("assessments").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
                assessments = a_res.data or []
            except Exception as e:
                print(f"Error fetching assessments from Supabase: {e}")
                assessments = self.memory_assessments.get(user_id, [])
                
            try:
                r_res = self.client.table("resume_analyses").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
                resumes = r_res.data or []
            except Exception as e:
                print(f"Error fetching resumes from Supabase: {e}")
                resumes = self.memory_resumes.get(user_id, [])
        else:
            assessments = self.memory_assessments.get(user_id, [])
            resumes = self.memory_resumes.get(user_id, [])
            
        return {
            "assessments": assessments,
            "resumes": resumes
        }

db_service = DBService()
