from django.db import models
from django.utils import timezone

class User(models.Model):
    firebase_uid = models.CharField(max_length=128, unique=True, null=True, blank=True, db_index=True)
    email = models.EmailField(unique=True, db_index=True)
    name = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.email

class Assessment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='assessments')
    
    # Input Features
    age = models.IntegerField()
    education = models.CharField(max_length=255)
    skills = models.JSONField()  # List of skills
    interests = models.JSONField() # List of interests
    riasec_scores = models.JSONField() # [R, I, A, S, E, C]
    
    # ML Results
    recommended_career = models.CharField(max_length=255)
    recommendation_score = models.FloatField()
    unsupervised_cluster = models.IntegerField()
    unsupervised_career = models.CharField(max_length=255)
    top_alternatives = models.JSONField() # List of dicts
    
    created_at = models.DateTimeField(default=timezone.now)

class Resume(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='resumes')
    
    file_path = models.CharField(max_length=512)
    ats_score = models.FloatField(null=True, blank=True)
    skill_gap_analysis = models.JSONField(null=True, blank=True)
    
    created_at = models.DateTimeField(default=timezone.now)
