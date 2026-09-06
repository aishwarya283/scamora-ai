// lib/services/contact_check_service.dart
import 'dart:async';
import 'dart:convert';

import 'package:flutter/services.dart';

class ContactCheckResult {
  final bool known;
  final String reason;

  ContactCheckResult({required this.known, required this.reason});

  factory ContactCheckResult.fromJson(String jsonStr) {
    final Map<String, dynamic> m = jsonDecode(jsonStr);
    return ContactCheckResult(
      known: m['known'] == true,
      reason: m['reason'] ?? '',
    );
  }
}

class ContactCheckService {
  static const MethodChannel _channel = MethodChannel('scamora/contact_check');

  /// Calls platform to normalize (canonical) phone number string.
  static Future<String> normalizePhoneNumber(String number) async {
    try {
      final String res = await _channel.invokeMethod('normalizePhoneNumber', {'number': number});
      return res;
    } on PlatformException {
      // Fallback to local normalization if platform call fails
      return _localNormalize(number);
    }
  }

  /// Returns ContactCheckResult where known==true indicates a contact match.
  static Future<ContactCheckResult> isKnownContact(String number) async {
    try {
      final String raw = await _channel.invokeMethod('isKnownContact', {'number': number});
      return ContactCheckResult.fromJson(raw);
    } on PlatformException catch (e) {
      return ContactCheckResult(known: false, reason: 'platform_error:${e.code}');
    }
  }

  /// Local Dart normalization heuristic (same semantics as Kotlin normalize)
  static String _localNormalize(String input) {
    if (input.isEmpty) return '';
    var s = input.trim();
    // Remove non-digits
    final digits = s.replaceAll(RegExp(r'[^0-9]'), '');
    if (digits.isEmpty) return '';
    if (digits.length == 10) return digits;
    if (digits.length > 10) return digits.substring(digits.length - 10);
    return digits;
  }
}
