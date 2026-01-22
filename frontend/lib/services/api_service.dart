import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

/// Simple API service that posts to /predict
class ApiService {
  final String baseUrl;
  ApiService(this.baseUrl);

  /// text: free text (optional)
  /// sensors: optional sensors string (we will omit sensors on web if hardware is used)
  /// imageBytes: optional JPEG/PNG bytes
  Future<Map<String, dynamic>> predict({
    required String text,
    String? sensors,
    Uint8List? imageBytes,
    String clientId = "web_client",
  }) async {
    final uri = Uri.parse("$baseUrl/predict");
    // Use JSON body (server supports application/json)
    final body = {
      "text": text,
      "sensors": sensors ?? "",
      "client_id": clientId,
    };

    // If image provided, send multipart; otherwise JSON
    if (imageBytes != null) {
      final request = http.MultipartRequest('POST', uri);
      request.fields.addAll({
        "text": text,
        "sensors": sensors ?? "",
        "client_id": clientId,
      });
      request.files.add(http.MultipartFile.fromBytes('image', imageBytes, filename: 'upload.jpg'));
      final streamed = await request.send();
      final resp = await http.Response.fromStream(streamed);
      return json.decode(resp.body) as Map<String, dynamic>;
    } else {
      final resp = await http.post(uri,
          headers: {"Content-Type": "application/json"},
          body: json.encode(body));
      return json.decode(resp.body) as Map<String, dynamic>;
    }
  }

  /// RAG diagnose: image + description -> /rag
  Future<Map<String, dynamic>> ragDiagnose({
    required String description,
    Uint8List? imageBytes,
    String clientId = "web_client",
  }) async {
    final uri = Uri.parse("$baseUrl/rag");
    // If image provided (bytes), send multipart; otherwise JSON
    if (imageBytes != null) {
      final request = http.MultipartRequest('POST', uri);
      request.fields.addAll({
        "description": description,
        "client_id": clientId,
      });
      request.files.add(http.MultipartFile.fromBytes('image', imageBytes, filename: 'upload.jpg'));
      final streamed = await request.send();
      final resp = await http.Response.fromStream(streamed);
      return json.decode(resp.body) as Map<String, dynamic>;
    } else {
      final resp = await http.post(uri,
          headers: {"Content-Type": "application/json"},
          body: json.encode({"description": description, "client_id": clientId}));
      return json.decode(resp.body) as Map<String, dynamic>;
    }
  }
}
