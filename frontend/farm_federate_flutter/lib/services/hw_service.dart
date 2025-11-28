// lib/services/hw_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
const String SERVER_BASE = "http://YOUR_BACKEND_IP:8000";

class HWService {
  static Future<Map<String,dynamic>> getLatest(String deviceId) async {
    final r = await http.get(Uri.parse("$SERVER_BASE/telemetry_latest?device_id=$deviceId"));
    if (r.statusCode != 200) throw Exception("Failed to fetch latest telemetry");
    return json.decode(r.body) as Map<String,dynamic>;
  }

  /// request a device action (queued at backend). pin commonly "relay" or "V1"
  static Future<Map<String,dynamic>> setAction(String deviceId, String pin, int value, {String? reason}) async {
    final r = await http.post(Uri.parse("$SERVER_BASE/set_action"),
      headers: {"Content-Type":"application/json"},
      body: json.encode({"device_id": deviceId, "pin": pin, "value": value, "reason": reason})
    );
    if (r.statusCode != 200) throw Exception("Failed to set action: ${r.body}");
    return json.decode(r.body) as Map<String,dynamic>;
  }
}
