class AssessmentHistoryItem {
  final String recommendedCareer;
  final double recommendationScore;
  final String createdAt;
  final int? age;
  final List<String> skills;
  final List<String> interests;
  final List<double> riasecScores;

  const AssessmentHistoryItem({
    required this.recommendedCareer,
    required this.recommendationScore,
    required this.createdAt,
    this.age,
    this.skills = const [],
    this.interests = const [],
    this.riasecScores = const [],
  });

  factory AssessmentHistoryItem.fromJson(Map<String, dynamic> json) {
    final predictionData = json['prediction_data'] as Map<String, dynamic>? ?? {};
    final userInputs = predictionData['user_inputs'] as Map<String, dynamic>? ?? {};
    return AssessmentHistoryItem(
      recommendedCareer: predictionData['Recommended_Career'] as String? ?? 'Unknown',
      recommendationScore: (predictionData['Recommendation_Score'] as num?)?.toDouble() ?? 0.0,
      createdAt: json['created_at'] as String? ?? '',
      age: (userInputs['age'] as num?)?.toInt(),
      skills: (userInputs['skills'] as List?)?.map((e) => e.toString()).toList() ?? [],
      interests: (userInputs['interests'] as List?)?.map((e) => e.toString()).toList() ?? [],
      riasecScores: (userInputs['riasec_scores'] as List?)?.map((e) => (e as num).toDouble()).toList() ?? [],
    );
  }
}

class ResumeHistoryItem {
  final int atsScore;
  final String createdAt;

  const ResumeHistoryItem({
    required this.atsScore,
    required this.createdAt,
  });

  factory ResumeHistoryItem.fromJson(Map<String, dynamic> json) {
    final analysisData = json['analysis_data'] as Map<String, dynamic>? ?? {};
    return ResumeHistoryItem(
      atsScore: (analysisData['ats_score'] as num?)?.toInt() ?? 0,
      createdAt: json['created_at'] as String? ?? '',
    );
  }
}

class HistoryData {
  final List<AssessmentHistoryItem> assessments;
  final List<ResumeHistoryItem> resumes;

  const HistoryData({
    required this.assessments,
    required this.resumes,
  });

  factory HistoryData.fromJson(Map<String, dynamic> json) {
    final assessmentsRaw = json['assessments'] as List? ?? [];
    final resumesRaw = json['resumes'] as List? ?? [];

    return HistoryData(
      assessments: assessmentsRaw
          .map((item) => AssessmentHistoryItem.fromJson(item as Map<String, dynamic>))
          .toList(),
      resumes: resumesRaw
          .map((item) => ResumeHistoryItem.fromJson(item as Map<String, dynamic>))
          .toList(),
    );
  }
}
