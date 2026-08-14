import os
import json
import base64
import io
import requests
from pypdf import PdfReader

class GeminiService:
    def __init__(self):
        self.default_api_key = os.environ.get("GEMINI_API_KEY", "")

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract plain text from PDF bytes using pypdf."""
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text.strip()
        except Exception as e:
            print(f"pypdf extraction error: {e}")
            return ""

    def analyze_resume_with_gemini(
        self,
        resume_text: str = "",
        file_bytes: bytes = None,
        mime_type: str = None,
        target_career: str = "Software Engineer",
        user_api_key: str = None
    ) -> dict:
        """
        Analyzes resume text or file (PDF/Image) using Gemini API (gemini-2.5-flash or gemini-1.5-flash).
        Returns a dict containing 'analysis', 'interview_prep', and 'analyzed_by'.
        """
        api_key = user_api_key or self.default_api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return None

        prompt = (
            f"You are an expert HR Recruiter and ATS (Applicant Tracking System) Resume Analyzer.\n"
            f"Thoroughly analyze this resume against the target role: '{target_career}'.\n\n"
            f"Return ONLY a raw valid JSON object with NO markdown code fences (` ```json `), NO trailing text, matching this exact structure:\n"
            f"{{\n"
            f'  "ats_score": 82.5,\n'
            f'  "skills_found": ["Python", "Flask", "SQL"],\n'
            f'  "skills_missing": ["Docker", "Kubernetes", "AWS"],\n'
            f'  "recommendation": "Strong technical background with good foundational skills, but needs cloud deployment experience.",\n'
            f'  "overall_analysis": "Comprehensive review of work history, skill gaps, resume layout, and ATS optimization advice for {target_career}.",\n'
            f'  "top_resume_keywords": ["Python", "REST API", "Database", "Git"],\n'
            f'  "top_matching_careers": [\n'
            f'    {{"career": "{target_career}", "similarity": 82.5}},\n'
            f'    {{"career": "Backend Developer", "similarity": 78.0}},\n'
            f'    {{"career": "Data Engineer", "similarity": 70.0}}\n'
            f'  ],\n'
            f'  "interview_prep": {{\n'
            f'    "interview_questions": [\n'
            f'      "Walk me through a complex API project you built in Python.",\n'
            f'      "How do you approach database schema design and query optimization?",\n'
            f'      "Describe how you handle authentication and security in web services.",\n'
            f'      "Explain a time you had to debug a difficult production issue."\n'
            f'    ],\n'
            f'    "tips": [\n'
            f'      "Highlight concrete metrics and business impact in your work experience.",\n'
            f'      "Prepare to elaborate on missing skills like Docker and AWS.",\n'
            f'      "Use the STAR method for behavioral responses."\n'
            f'    ],\n'
            f'    "roadmap_url": "https://roadmap.sh/backend"\n'
            f'  }}\n'
            f"}}\n"
        )

        parts = []

        if file_bytes and mime_type:
            b64_data = base64.b64encode(file_bytes).decode('utf-8')
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": b64_data
                }
            })

        if resume_text:
            prompt += f"\n\nResume Content:\n{resume_text}"

        parts.append({"text": prompt})

        # Try gemini-2.5-flash first, then gemini-1.5-flash
        models = ["gemini-2.5-flash", "gemini-1.5-flash"]
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": 0.2,
                    "response_mime_type": "application/json"
                }
            }

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    res_data = response.json()
                    candidates = res_data.get("candidates", [])
                    if candidates:
                        text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                        if text_content.startswith("```json"):
                            text_content = text_content[7:]
                        if text_content.startswith("```"):
                            text_content = text_content[3:]
                        if text_content.endswith("```"):
                            text_content = text_content[:-3]
                        text_content = text_content.strip()

                        parsed = json.loads(text_content)
                        interview_prep = parsed.pop("interview_prep", {})
                        
                        return {
                            "analysis": parsed,
                            "interview_prep": interview_prep,
                            "analyzed_by": f"Gemini AI ({model})"
                        }
                else:
                    print(f"Gemini {model} returned HTTP {response.status_code}: {response.text}")
            except Exception as e:
                print(f"Gemini API request error ({model}): {e}")

        return None

gemini_service = GeminiService()
