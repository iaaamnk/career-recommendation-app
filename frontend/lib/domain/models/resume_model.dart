class ResumeAnalysisResult {
  final double atsScore;
  final String recommendation;
  final List<String> skillsFound;
  final List<String> skillsMissing;
  final List<String> interviewQuestions;
  final String? roadmapUrl;

  const ResumeAnalysisResult({
    required this.atsScore,
    required this.recommendation,
    required this.skillsFound,
    required this.skillsMissing,
    required this.interviewQuestions,
    this.roadmapUrl,
  });

  factory ResumeAnalysisResult.fromJson(Map<String, dynamic> json) {
    final analysis = json['analysis'] as Map<String, dynamic>? ?? {};
    final prep = json['interview_prep'] as Map<String, dynamic>?;

    return ResumeAnalysisResult(
      atsScore: (analysis['ats_score'] as num?)?.toDouble() ?? 0.0,
      recommendation: analysis['recommendation'] as String? ?? '',
      skillsFound: (analysis['skills_found'] as List? ?? [])
          .map((e) => e.toString())
          .toList(),
      skillsMissing: (analysis['skills_missing'] as List? ?? [])
          .map((e) => e.toString())
          .toList(),
      interviewQuestions: (prep?['interview_questions'] as List? ?? [])
          .map((e) => e.toString())
          .toList(),
      roadmapUrl: prep?['roadmap_url'] as String?,
    );
  }
}
