import 'package:flutter/foundation.dart';

class ApiConstants {
  static String get baseUrl {
    if (kDebugMode || kIsWeb) {
      return 'http://127.0.0.1:5000';
    }
    return 'https://career-recommendation-app-2-08ny.onrender.com';
  }

  // Endpoint paths
  static const String health = '/health';
  static const String recommend = '/api/recommend';
  static const String resumeAnalyze = '/api/resume/analyze';
  static const String interviewPrep = '/api/interview/prep';
  static const String history = '/api/history';
}
