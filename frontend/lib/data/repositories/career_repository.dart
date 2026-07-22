import '../../core/constants/api_constants.dart';
import '../../core/network/api_service.dart';
import '../../domain/models/assessment_model.dart';
import '../../domain/models/history_model.dart';

class CareerRepository {
  final ApiService _apiService;

  CareerRepository({ApiService? apiService})
      : _apiService = apiService ?? ApiService();

  /// Submits assessment quiz inputs and returns predicted recommendation
  Future<AssessmentResult?> submitAssessment(AssessmentInput input) async {
    final response = await _apiService.post(
      ApiConstants.recommend,
      input.toJson(),
    );

    if (response == null) return null;
    return AssessmentResult.fromJson(response);
  }

  /// Fetches historical assessments and resume scans
  Future<HistoryData?> fetchHistory() async {
    final response = await _apiService.get(ApiConstants.history);

    if (response == null) return null;
    return HistoryData.fromJson(response);
  }
}
