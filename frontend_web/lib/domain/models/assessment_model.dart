class CareerAlternative {
  final String career;
  final double score;

  const CareerAlternative({
    required this.career,
    required this.score,
  });

  factory CareerAlternative.fromJson(Map<String, dynamic> json) {
    return CareerAlternative(
      career: json['career'] as String? ?? 'Unknown',
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class AssessmentResult {
  final String recommendedCareer;
  final double recommendationScore;
  final int cluster;
  final List<CareerAlternative> topCareers;

  const AssessmentResult({
    required this.recommendedCareer,
    required this.recommendationScore,
    required this.cluster,
    required this.topCareers,
  });

  factory AssessmentResult.fromJson(Map<String, dynamic> json) {
    final pred = json['prediction'] as Map<String, dynamic>? ?? json;
    final alternativesRaw = pred['Top_3_Careers'] as List? ?? [];
    
    return AssessmentResult(
      recommendedCareer: pred['Recommended_Career'] as String? ?? 'Unknown',
      recommendationScore: (pred['Recommendation_Score'] as num?)?.toDouble() ?? 0.0,
      cluster: (pred['Unsupervised_Cluster'] as num?)?.toInt() ?? 0,
      topCareers: alternativesRaw
          .map((item) => CareerAlternative.fromJson(item as Map<String, dynamic>))
          .toList(),
    );
  }
}

class AssessmentInput {
  final int age;
  final String education;
  final List<String> skills;
  final List<String> interests;
  final List<double> riasecScores;

  const AssessmentInput({
    required this.age,
    required this.education,
    required this.skills,
    required this.interests,
    required this.riasecScores,
  });

  Map<String, dynamic> toJson() {
    return {
      "age": age,
      "education": education,
      "skills": skills,
      "interests": interests,
      "riasec_scores": riasecScores,
    };
  }
}
