class ApiConstants {
  static String get baseUrl {
    // Returning production URL by default since local backend isn't running
    return 'https://career-recommendation-app-2-08ny.onrender.com';
  }

  // Endpoint paths
  static const String health = '/health';
  static const String recommend = '/api/recommend';
  static const String resumeAnalyze = '/api/resume/analyze';
  static const String interviewPrep = '/api/interview/prep';
  static const String history = '/api/history';
}
