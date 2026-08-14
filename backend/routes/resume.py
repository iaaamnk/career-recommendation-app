import uuid
import base64
from flask import Blueprint, request, jsonify, g
from auth import require_auth
from services.nlp_service import nlp_service
from services.gemini_service import gemini_service
from services.db_service import db_service
from utils.error_handlers import ValidationError, APIError

resume_bp = Blueprint('resume', __name__)

@resume_bp.route('/api/resume/analyze', methods=['POST'], strict_slashes=False)
@require_auth
def analyze_resume():
    user = g.user
    
    resume_text = ""
    target_career = ""
    user_api_key = request.headers.get("X-Gemini-API-Key", "")
    file_bytes = None
    mime_type = None

    # Handle multipart form upload
    if request.files:
        file_obj = request.files.get("file")
        target_career = request.form.get("target_career", "").strip()
        resume_text = request.form.get("resume_text", "").strip()
        user_api_key = request.form.get("gemini_api_key", user_api_key).strip()

        if file_obj:
            filename = file_obj.filename.lower()
            file_bytes = file_obj.read()
            if filename.endswith(".pdf"):
                mime_type = "application/pdf"
            elif filename.endswith(".png"):
                mime_type = "image/png"
            elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
                mime_type = "image/jpeg"
            elif filename.endswith(".webp"):
                mime_type = "image/webp"

    # Handle JSON request body
    else:
        data = request.get_json() or {}
        resume_text = data.get("resume_text", "").strip()
        target_career = data.get("target_career", "").strip()
        user_api_key = data.get("gemini_api_key", user_api_key).strip()
        b64_file = data.get("file_data", "")
        mime_type = data.get("mime_type", "")

        if b64_file:
            try:
                file_bytes = base64.b64decode(b64_file)
            except Exception:
                raise ValidationError("Invalid base64 encoded file data")

    if not target_career:
        raise ValidationError("target_career is required")

    extracted_pdf_text = ""
    if file_bytes and mime_type == "application/pdf":
        extracted_pdf_text = gemini_service.extract_text_from_pdf(file_bytes)
        if not resume_text and extracted_pdf_text:
            resume_text = extracted_pdf_text

    if not resume_text and not file_bytes:
        raise ValidationError("Please provide resume_text or upload a PDF/Image file.")

    # 1. Try Gemini API
    gemini_result = gemini_service.analyze_resume_with_gemini(
        resume_text=resume_text,
        file_bytes=file_bytes,
        mime_type=mime_type,
        target_career=target_career,
        user_api_key=user_api_key
    )

    if gemini_result:
        analysis = gemini_result["analysis"]
        interview_prep = gemini_result["interview_prep"]
        analyzed_by = gemini_result.get("analyzed_by", "Gemini AI")
    else:
        # 2. Local Fallback (if no Gemini API key or Gemini API call failed)
        if not resume_text and mime_type in ["image/png", "image/jpeg", "image/webp"]:
            raise APIError("Analyzing image resumes requires a Gemini API Key. Please provide a Gemini API Key or paste your resume text / upload a PDF.", status_code=400)
        
        if not resume_text:
            raise ValidationError("Could not extract readable text from the uploaded file. Please paste your resume text.")

        try:
            analysis = nlp_service.analyze_resume_text(resume_text, target_career)
            interview_prep = nlp_service.generate_interview_prep(target_career, analysis.get("skills_missing", []))
            analyzed_by = "Local NLP Engine (TF-IDF Matcher)"
        except Exception as e:
            raise APIError(f"Resume analysis error: {str(e)}", status_code=500)

    analysis["analyzed_by"] = analyzed_by
    if extracted_pdf_text:
        analysis["extracted_text"] = extracted_pdf_text[:500] + ("..." if len(extracted_pdf_text) > 500 else "")

    resume_id = str(uuid.uuid4())
    
    # Save to database
    db_service.insert_resume_analysis(user["id"], resume_id, analysis, interview_prep)

    return jsonify({
        "resume_id": resume_id,
        "analysis": analysis,
        "interview_prep": interview_prep,
        "analyzed_by": analyzed_by,
        "extracted_text": extracted_pdf_text
    }), 200


