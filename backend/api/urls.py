from django.urls import path
from . import views

urlpatterns = [
    path('health', views.health_check, name='health_check'),
    path('api/recommend', views.recommend_career, name='recommend_career'),
    path('api/resume/analyze', views.analyze_resume, name='analyze_resume'),
    path('api/interview/prep', views.get_interview_prep, name='interview_prep'),
    path('api/history', views.get_user_history, name='get_user_history'),
]
