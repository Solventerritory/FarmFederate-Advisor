// lib/services/api_service.dart

import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:mime/mime.dart';

import '../models/predict_result.dart';

class ApiService {
  /// IMPORTANT: match your FastAPI server address.
  /// For local desktop/web:
  ///   "http://127.0.0.1:8000"
  /// If running on a different machine, use that machine's LAN IP.
  static String baseUrl = "http://127.0.0.1:8000";

  /// Send text + optional sensor string + optional image bytes
  static Future<PredictResult> predict({
    required String text,
    String sensors = "",
    Uint8List? imageBytes,
    String? imageName,
    String clientId = "flutter_client",
  }) async {
    final uri = Uri.parse("$baseUrl/predict");

    // ---------- CASE 1: multipart (image present) ----------
    if (imageBytes != null && imageBytes.isNotEmpty) {
      final req = http.MultipartRequest("POST", uri);

      req.fields["text"] = text;
      req.fields["client_id"] = clientId;
      if (sensors.isNotEmpty) {
        req.fields["sensors"] = sensors;
      }

      final fname = imageName ?? "upload.jpg";
      final mime = lookupMimeType(fname) ?? "image/jpeg";
      final split = mime.split("/");

      req.files.add(
        http.MultipartFile.fromBytes(
          "image",
          imageBytes,
          filename: fname,
          contentType: MediaType(
            split.first,
            split.length > 1 ? split.last : "jpeg",
          ),
        ),
      );

      final streamRes = await req.send();
      final res = await http.Response.fromStream(streamRes);

      if (res.statusCode >= 200 && res.statusCode < 300) {
        return PredictResult.fromJson(json.decode(res.body));
      } else {
        throw Exception("Server error: ${res.statusCode} ${res.body}");
      }
    }

    // ---------- CASE 2: JSON-only (no image) ----------
    final response = await http.post(
      uri,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "text": text,
        "client_id": clientId,
        "sensors": sensors,
      }),
    );

    if (response.statusCode >= 200 && response.statusCode < 300) {
      return PredictResult.fromJson(json.decode(response.body));
    } else {
      throw Exception("Server error: ${response.statusCode} ${response.body}");
    }
  }
}
