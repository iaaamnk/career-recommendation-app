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

  /// Retrieves active auth token from Supabase, Firebase, or Demo token
  Future<String> _getAuthToken() async {
    try {
      final token = Supabase.instance.client.auth.currentSession?.accessToken;
      if (token != null && token.isNotEmpty) return token;
    } catch (_) {}

    try {
      final user = fb.FirebaseAuth.instance.currentUser;
      if (user != null) {
        final token = await user.getIdToken();
        if (token != null && token.isNotEmpty) return token;
      }
    } catch (_) {}

    return 'demo-bearer-token-123';
  }

  /// Performs GET request with fallback
  Future<Map<String, dynamic>?> get(String path) async {
    final token = await _getAuthToken();
    final primaryUrl = '${ApiConstants.baseUrl}$path';

    try {
      final response = await _client.get(
        Uri.parse(primaryUrl),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
      ).timeout(const Duration(seconds: 4));

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        throw Exception("Server returned ${response.statusCode}: ${response.body}");
      }
    } catch (e) {
      debugPrint('GET primary URL error: $e. Attempting localhost fallback...');
      if (e.toString().contains('Server returned')) {
        throw e;
      }
    }

    if (!primaryUrl.contains('127.0.0.1') && !primaryUrl.contains('localhost')) {
      try {
        final response = await _client.get(
          Uri.parse('http://127.0.0.1:5000$path'),
          headers: {
            'Authorization': 'Bearer $token',
            'Content-Type': 'application/json',
          },
        ).timeout(const Duration(seconds: 4));

        if (response.statusCode == 200) {
          return jsonDecode(response.body) as Map<String, dynamic>;
        }
      } catch (_) {}
    }

    throw Exception("Network request failed for $path");
  }

  /// Performs POST request with auth headers & intelligent offline fallback
  Future<Map<String, dynamic>?> post(String path, Map<String, dynamic> body) async {
    final token = await _getAuthToken();
    final primaryUrl = '${ApiConstants.baseUrl}$path';

    try {
      final response = await _client.post(
        Uri.parse(primaryUrl),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
        body: jsonEncode(body),
      ).timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        throw Exception("Server returned ${response.statusCode}: ${response.body}");
      }
    } catch (e) {
      debugPrint('POST primary URL error: $e');
      // If it's a server error from our primary URL, throw it immediately 
      // rather than trying localhost fallback which masks the real error.
      if (e.toString().contains('Server returned')) {
        throw e;
      }
    }

    if (!primaryUrl.contains('127.0.0.1') && !primaryUrl.contains('localhost')) {
      try {
        final response = await _client.post(
          Uri.parse('http://127.0.0.1:5000$path'),
          headers: {
            'Authorization': 'Bearer $token',
            'Content-Type': 'application/json',
          },
          body: jsonEncode(body),
        ).timeout(const Duration(seconds: 5));

        if (response.statusCode == 200) {
          return jsonDecode(response.body) as Map<String, dynamic>;
        }
      } catch (_) {}
    }

    throw Exception("Network request failed for $path");
  }
}
