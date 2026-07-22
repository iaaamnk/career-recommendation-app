from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from .models import Assessment, Resume
import ml_model
import nlp_model

@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    return Response({
        "status": "ok",
        "model_loaded": ml_model.rf_model is not None
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def recommend_career(request):
    user = request.user
    data = request.data
    
    try:
        age = int(data.get("age", 24))
        education = str(data.get("education", ""))
        skills = list(data.get("skills", []))
        interests = list(data.get("interests", []))
        riasec_scores = [float(s) for s in data.get("riasec_scores", [])]
    except Exception as e:
        return Response({"detail": f"Invalid request data: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        
    try:
        prediction = ml_model.predict_career(
            age=age,
            education=education,
            skills=skills,
            interests=interests,
            riasec=riasec_scores
        )
    except Exception as e:
        return Response({"detail": f"Prediction error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    assessment = Assessment.objects.create(
        user=user,
        age=age,
        education=education,
        skills=skills,
        interests=interests,
        riasec_scores=riasec_scores,
        recommended_career=prediction["Recommended_Career"],
        recommendation_score=prediction["Recommendation_Score"],
        unsupervised_cluster=prediction["Unsupervised_Cluster"],
        unsupervised_career=prediction["Unsupervised_Recommendation"],
        top_alternatives=prediction["Top_3_Careers"]
    )
    
    return Response({
        "assessment_id": assessment.id,
        "prediction": prediction
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_resume(request):
    user = request.user
    data = request.data
    
    resume_text = data.get("resume_text", "")
    target_career = data.get("target_career", "")
    
    if not resume_text or not target_career:
        return Response({"detail": "resume_text and target_career are required"}, status=status.HTTP_400_BAD_REQUEST)
        
    try:
        analysis = nlp_model.analyze_resume_text(resume_text, target_career)
        interview_prep = nlp_model.generate_interview_prep(target_career, analysis["skills_missing"])
    except Exception as e:
        return Response({"detail": f"Analysis error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    resume = Resume.objects.create(
        user=user,
        file_path="simulated_path.pdf",
        ats_score=analysis["ats_score"],
        skill_gap_analysis={"missing": analysis["skills_missing"], "found": analysis["skills_found"]}
    )
    
    return Response({
        "resume_id": resume.id,
        "analysis": analysis,
        "interview_prep": interview_prep
    })

@api_view(['POST'])
@permission_classes([AllowAny])
def get_interview_prep(request):
    data = request.data
    target_career = data.get("target_career", "")
    missing_skills = list(data.get("missing_skills", []))
    
    if not target_career:
        return Response({"detail": "target_career is required"}, status=status.HTTP_400_BAD_REQUEST)
        
    try:
        prep = nlp_model.generate_interview_prep(target_career, missing_skills)
        return Response(prep)
    except Exception as e:
        return Response({"detail": f"Interview prep error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_history(request):
    user = request.user
    
    assessments = Assessment.objects.filter(user=user).order_by('-created_at')
    resumes = Resume.objects.filter(user=user).order_by('-created_at')
    
    assessments_data = []
    for a in assessments:
        assessments_data.append({
            "id": a.id,
            "recommended_career": a.recommended_career,
            "recommendation_score": a.recommendation_score,
            "unsupervised_cluster": a.unsupervised_cluster,
            "unsupervised_career": a.unsupervised_career,
            "top_alternatives": a.top_alternatives,
            "created_at": a.created_at.isoformat()
        })
        
    resumes_data = []
    for r in resumes:
        resumes_data.append({
            "id": r.id,
            "ats_score": r.ats_score,
            "skill_gap_analysis": r.skill_gap_analysis,
            "created_at": r.created_at.isoformat()
        })
        
    return Response({
        "assessments": assessments_data,
        "resumes": resumes_data
    })
