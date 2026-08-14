import 'dart:convert';
import '../../core/constants/api_constants.dart';
import '../../core/network/api_service.dart';
import '../../domain/models/resume_model.dart';

class ResumeRepository {
  final ApiService _apiService;

  ResumeRepository({ApiService? apiService})
      : _apiService = apiService ?? ApiService();

  /// Analyzes resume text or uploaded file (PDF / Image) against a target career role
  Future<ResumeAnalysisResult?> analyzeResume({
    String? resumeText,
    required String targetCareer,
    List<int>? fileBytes,
    String? fileName,
    String? mimeType,
    String? geminiApiKey,
  }) async {
    final Map<String, dynamic> body = {
      "target_career": targetCareer,
    };

    if (resumeText != null && resumeText.trim().isNotEmpty) {
      body["resume_text"] = resumeText.trim();
    }

    if (fileBytes != null && fileBytes.isNotEmpty) {
      body["file_data"] = base64Encode(fileBytes);
      body["mime_type"] = mimeType ?? "application/pdf";
    }

    if (geminiApiKey != null && geminiApiKey.trim().isNotEmpty) {
      body["gemini_api_key"] = geminiApiKey.trim();
    }

    final response = await _apiService.post(
      ApiConstants.resumeAnalyze,
      body,
    );

    if (response == null) return null;
    return ResumeAnalysisResult.fromJson(response);
  }
}
