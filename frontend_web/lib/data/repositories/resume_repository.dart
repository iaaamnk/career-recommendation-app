import '../../core/constants/api_constants.dart';
import '../../core/network/api_service.dart';
import '../../domain/models/resume_model.dart';

class ResumeRepository {
  final ApiService _apiService;

  ResumeRepository({ApiService? apiService})
      : _apiService = apiService ?? ApiService();

  /// Analyzes resume text against a target career role
  Future<ResumeAnalysisResult?> analyzeResume({
    required String resumeText,
    required String targetCareer,
  }) async {
    final response = await _apiService.post(
      ApiConstants.resumeAnalyze,
      {
        "resume_text": resumeText,
        "target_career": targetCareer,
      },
    );

    if (response == null) return null;
    return ResumeAnalysisResult.fromJson(response);
  }
}
