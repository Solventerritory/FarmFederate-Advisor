import 'dart:convert';
import 'package:http/http.dart' as http;
const String SERVER_BASE = "http://YOUR_BACKEND_IP:8000";

class HWService {
  static Future<Map<String,dynamic>> getLatest(String deviceId) async {
    final r = await http.get(Uri.parse("$SERVER_BASE/telemetry_latest?device_id=$deviceId"));
    if (r.statusCode != 200) throw Exception("Failed to fetch latest telemetry");
    return json.decode(r.body) as Map<String,dynamic>;
  }
  static Future<Map<String,dynamic>> control(String deviceId, String pin, int value) async {
    final r = await http.post(Uri.parse("$SERVER_BASE/control"),
      headers: {"Content-Type":"application/json"},
      body: json.encode({"device_id": deviceId, "pin": pin, "value": value})
    );
    if (r.statusCode != 200) throw Exception("Control failed: ${r.body}");
    return json.decode(r.body) as Map<String,dynamic>;
  }
}
