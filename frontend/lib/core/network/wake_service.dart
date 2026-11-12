import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../constants/api_constants.dart';

/// Pings the backend's /wake endpoint to wake up the Render free-tier dyno.
/// Should be called once at app startup, fire-and-forget.
class WakeService {
  static bool _woken = false;

  static Future<void> wakeBackend() async {
    if (_woken) return;
    _woken = true;
    try {
      final uri = Uri.parse('${ApiConstants.baseUrl}${ApiConstants.wake}');
      debugPrint('WakeService: pinging $uri');
      final response = await http.get(uri).timeout(const Duration(seconds: 60));
      debugPrint('WakeService: backend responded ${response.statusCode}');
    } catch (e) {
      // Warm-up failed silently — the first real request will trigger the cold start.
      debugPrint('WakeService: warm-up ping failed (backend still waking up): $e');
    }
  }
}
