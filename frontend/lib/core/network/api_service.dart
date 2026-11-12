import 'dart:convert';
import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import '../constants/api_constants.dart';

import '../../data/repositories/auth_repository.dart';

class ApiService {
  final http.Client _client;

  // Render free tier cold starts can take 30-60s, so we use a 60s timeout.
  static const Duration _timeout = Duration(seconds: 60);
  static const int _maxRetries = 2;
  static const Duration _retryDelay = Duration(seconds: 2);

  ApiService({http.Client? client}) : _client = client ?? http.Client();

  /// Retrieves active auth token from Supabase, Firebase, or session token
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

    final sessionUser = AuthRepository().currentUser;
    final userId = sessionUser?.id ?? 'user-default-1';
    return 'bearer-user-token-$userId';
  }

  Map<String, String> _headers(String token) => {
        'Authorization': 'Bearer $token',
        'Content-Type': 'application/json',
      };

  /// Performs GET request with retry logic for cold-start tolerance.
  Future<Map<String, dynamic>?> get(String path) async {
    final token = await _getAuthToken();
    final url = Uri.parse('${ApiConstants.baseUrl}$path');
    Exception? lastError;

    for (int attempt = 0; attempt <= _maxRetries; attempt++) {
      if (attempt > 0) {
        await Future.delayed(_retryDelay);
        debugPrint('GET $path – retry $attempt/$_maxRetries');
      }
      try {
        final response = await _client
            .get(url, headers: _headers(token))
            .timeout(_timeout);

        if (response.statusCode == 200) {
          return jsonDecode(response.body) as Map<String, dynamic>;
        }
        // Non-retriable HTTP error – throw immediately with details.
        throw Exception(
            'Server error ${response.statusCode}: ${response.body}');
      } on TimeoutException catch (e) {
        debugPrint('GET $path timed out (attempt $attempt): $e');
        lastError = Exception(
            'Request timed out. The server may be starting up – please try again in a moment.');
      } catch (e) {
        debugPrint('GET $path failed (attempt $attempt): $e');
        // Only retry on network-level errors, not HTTP errors.
        if (e is Exception && e.toString().contains('Server error')) rethrow;
        lastError = e is Exception ? e : Exception(e.toString());
      }
    }
    throw lastError ?? Exception('GET $path failed after $_maxRetries retries');
  }

  /// Performs POST request with auth headers and retry logic.
  Future<Map<String, dynamic>?> post(
      String path, Map<String, dynamic> body) async {
    final token = await _getAuthToken();
    final url = Uri.parse('${ApiConstants.baseUrl}$path');
    final encodedBody = jsonEncode(body);
    Exception? lastError;

    for (int attempt = 0; attempt <= _maxRetries; attempt++) {
      if (attempt > 0) {
        await Future.delayed(_retryDelay);
        debugPrint('POST $path – retry $attempt/$_maxRetries');
      }
      try {
        final response = await _client
            .post(url, headers: _headers(token), body: encodedBody)
            .timeout(_timeout);

        if (response.statusCode == 200) {
          return jsonDecode(response.body) as Map<String, dynamic>;
        }
        // Non-retriable HTTP error – include status + body for debugging.
        throw Exception(
            'Server error ${response.statusCode}: ${response.body}');
      } on TimeoutException catch (e) {
        debugPrint('POST $path timed out (attempt $attempt): $e');
        lastError = Exception(
            'Request timed out. The server may be starting up – please try again in a moment.');
      } catch (e) {
        debugPrint('POST $path failed (attempt $attempt): $e');
        if (e is Exception && e.toString().contains('Server error')) rethrow;
        lastError = e is Exception ? e : Exception(e.toString());
      }
    }
    throw lastError ?? Exception('POST $path failed after $_maxRetries retries');
  }
}
