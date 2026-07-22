from django.db import models

class User(models.Model):
    firebase_uid = models.CharField(max_length=255, unique=True, null=True, blank=True, db_index=True)
    email = models.EmailField(unique=True, db_index=True)
    name = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def is_authenticated(self):
        return True

    class Meta:
        db_table = "users"

class Assessment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='assessments', db_column='user_id')
    age = models.IntegerField()
    education = models.CharField(max_length=255)
    skills = models.JSONField()
    interests = models.JSONField()
    riasec_scores = models.JSONField()
    recommended_career = models.CharField(max_length=255)
    recommendation_score = models.FloatField()
    unsupervised_cluster = models.IntegerField()
    unsupervised_career = models.CharField(max_length=255)
    top_alternatives = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "assessments"

class Resume(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='resumes', db_column='user_id')
    file_path = models.CharField(max_length=255)
    ats_score = models.FloatField(null=True, blank=True)
    skill_gap_analysis = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "resumes"
