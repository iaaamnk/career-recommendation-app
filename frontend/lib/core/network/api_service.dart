import 'dart:convert';
import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import '../constants/api_constants.dart';

class ApiService {
  final http.Client _client;

  ApiService({http.Client? client}) : _client = client ?? http.Client();

  /// Retrieves active auth token from Supabase or Firebase
  Future<String?> _getAuthToken() async {
    try {
      final token = Supabase.instance.client.auth.currentSession?.accessToken;
      if (token != null) return token;
    } catch (_) {}

    try {
      final user = fb.FirebaseAuth.instance.currentUser;
      if (user != null) {
        return await user.getIdToken();
      }
    } catch (_) {}

    return null;
  }

  /// Performs GET request with exponential backoff retry
  Future<Map<String, dynamic>?> get(String path, {int maxRetries = 3}) async {
    final token = await _getAuthToken();
    if (token == null) return null;

    final url = Uri.parse('${ApiConstants.baseUrl}$path');

    for (int attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        final response = await _client.get(
          url,
          headers: {
            'Authorization': 'Bearer $token',
            'Content-Type': 'application/json',
          },
        ).timeout(const Duration(seconds: 60));

        if (response.statusCode == 200) {
          return jsonDecode(response.body) as Map<String, dynamic>;
        }
        if (response.statusCode < 500 || attempt == maxRetries) {
          return null;
        }
      } catch (e) {
        debugPrint('GET request error (attempt $attempt): $e');
        if (attempt == maxRetries) return null;
      }
      await Future.delayed(Duration(seconds: (attempt + 1) * 3));
    }
    return null;
  }

  /// Performs POST request with auth headers
  Future<Map<String, dynamic>?> post(String path, Map<String, dynamic> body) async {
    final token = await _getAuthToken();
    if (token == null) return null;

    final url = Uri.parse('${ApiConstants.baseUrl}$path');

    try {
      final response = await _client.post(
        url,
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
        body: jsonEncode(body),
      ).timeout(const Duration(seconds: 60));

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        debugPrint('POST request failed with status: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      debugPrint('POST request exception: $e');
      return null;
    }
  }
}
