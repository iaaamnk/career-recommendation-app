class AssessmentHistoryItem {
  final String recommendedCareer;
  final double recommendationScore;
  final String createdAt;

  const AssessmentHistoryItem({
    required this.recommendedCareer,
    required this.recommendationScore,
    required this.createdAt,
  });

  factory AssessmentHistoryItem.fromJson(Map<String, dynamic> json) {
    return AssessmentHistoryItem(
      recommendedCareer: json['recommended_career'] as String? ?? 'Unknown',
      recommendationScore: (json['recommendation_score'] as num?)?.toDouble() ?? 0.0,
      createdAt: json['created_at'] as String? ?? '',
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
    return ResumeHistoryItem(
      atsScore: (json['ats_score'] as num?)?.toInt() ?? 0,
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
