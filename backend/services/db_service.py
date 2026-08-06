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
            self.client = create_client(self.supabase_url, self.supabase_key)
            
    def insert_assessment(self, user_id: str, assessment_id: str, prediction: dict):
        data = {
            "id": assessment_id,
            "user_id": user_id,
            "prediction_data": prediction,
        }
        
        import datetime
        data["created_at"] = datetime.datetime.utcnow().isoformat()
        
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
        
        import datetime
        data["created_at"] = datetime.datetime.utcnow().isoformat()
        
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
                
            try:
                r_res = self.client.table("resume_analyses").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
                resumes = r_res.data or []
            except Exception as e:
                print(f"Error fetching resume analyses from Supabase: {e}")
                
        # Merge with any in-memory fallback data for this user
        mem_assessments = self.memory_assessments.get(user_id, [])
        mem_resumes = self.memory_resumes.get(user_id, [])
        
        # Avoid duplicates if it somehow succeeded in DB but also went to memory
        db_a_ids = {a['id'] for a in assessments}
        db_r_ids = {r['id'] for r in resumes}
        
        for ma in mem_assessments:
            if ma['id'] not in db_a_ids:
                assessments.append(ma)
                
        for mr in mem_resumes:
            if mr['id'] not in db_r_ids:
                resumes.append(mr)
                
        # Sort combined results by created_at descending
        assessments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        resumes.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
        return {
            "assessments": assessments,
            "resumes": resumes
        }

db_service = DBService()
