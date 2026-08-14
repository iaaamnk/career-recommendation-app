class ResumeAnalysisResult {
  final double atsScore;
  final String recommendation;
  final List<String> skillsFound;
  final List<String> skillsMissing;
  final List<String> interviewQuestions;
  final String? roadmapUrl;
  final String analyzedBy;
  final String overallAnalysis;
  final List<String> topKeywords;
  final List<Map<String, dynamic>> topMatchingCareers;
  final String? extractedText;

  const ResumeAnalysisResult({
    required this.atsScore,
    required this.recommendation,
    required this.skillsFound,
    required this.skillsMissing,
    required this.interviewQuestions,
    this.roadmapUrl,
    this.analyzedBy = 'AI Analyzer',
    this.overallAnalysis = '',
    this.topKeywords = const [],
    this.topMatchingCareers = const [],
    this.extractedText,
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
      analyzedBy: json['analyzed_by'] as String? ??
          (analysis['analyzed_by'] as String? ?? 'AI Resume Analyzer'),
      overallAnalysis: analysis['overall_analysis'] as String? ?? '',
      topKeywords: (analysis['top_resume_keywords'] as List? ?? [])
          .map((e) => e.toString())
          .toList(),
      topMatchingCareers: (analysis['top_matching_careers'] as List? ?? [])
          .map((e) => Map<String, dynamic>.from(e as Map))
          .toList(),
      extractedText: json['extracted_text'] as String?,
    );
  }
}
