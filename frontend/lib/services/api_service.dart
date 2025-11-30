// lib/services/api_service.dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:mime/mime.dart';
import 'package:http_parser/http_parser.dart';
import '../models/predict_result.dart';

class ApiService {
  // EDIT this to your backend address reachable from your device/browser
  // Examples:
  // - Flutter web (same host): "http://127.0.0.1:8000"
  // - Android emulator: "http://10.0.2.2:8000"
  // - Real device: "http://192.168.0.195:8000" (from your logs)
  static String baseUrl = "http://127.0.0.1:8000";

  /// Send text + optional sensors + optional image as multipart.
  static Future<PredictResult> predict({
    required String text,
    String sensors = "",
    File? imageFile,
    String clientId = "flutter_client_1",
  }) async {
    final uri = Uri.parse("$baseUrl/predict");

    // If image present -> multipart/form-data
    if (imageFile != null && await imageFile.exists()) {
      final request = http.MultipartRequest('POST', uri);
      request.fields['text'] = text;
      if (sensors.isNotEmpty) request.fields['sensors'] = sensors;
      request.fields['client_id'] = clientId;

      final mimeType = lookupMimeType(imageFile.path) ?? 'image/jpeg';
      final parts = mimeType.split('/');
      final multipartFile = await http.MultipartFile.fromPath(
        'image',
        imageFile.path,
        filename: p.basename(imageFile.path),
        contentType: MediaType(parts[0], parts.length > 1 ? parts[1] : ''),
      );
      request.files.add(multipartFile);

      final streamedResp = await request.send();
      final resp = await http.Response.fromStream(streamedResp);

      if (resp.statusCode >= 200 && resp.statusCode < 300) {
        return PredictResult.fromJson(json.decode(resp.body));
      } else {
        throw Exception('Server error: ${resp.statusCode} ${resp.body}');
      }
    }

    // No image: use JSON
    final resp = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'text': text, 'sensors': sensors, 'client_id': clientId}),
    );

    if (resp.statusCode >= 200 && resp.statusCode < 300) {
      return PredictResult.fromJson(json.decode(resp.body));
    } else {
      throw Exception('Server error: ${resp.statusCode} ${resp.body}');
    }
  }
}
