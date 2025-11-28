import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;

const String SERVER_BASE = "http://YOUR_BACKEND_IP:8000";

class ApiService {
  static Future<Map<String, dynamic>> predict(File? imageFile, String text) async {
    final uri = Uri.parse("$SERVER_BASE/predict");
    final request = http.MultipartRequest('POST', uri);
    request.fields['text'] = text ?? '';
    if (imageFile != null && imageFile.existsSync()) {
      final fileStream = http.ByteStream(imageFile.openRead());
      final length = await imageFile.length();
      final multipartFile = http.MultipartFile('file', fileStream, length, filename: p.basename(imageFile.path));
      request.files.add(multipartFile);
    } else {
      throw Exception('Image required for this endpoint');
    }
    final streamed = await request.send();
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) throw Exception('Server ${resp.statusCode}: ${resp.body}');
    final decoded = json.decode(resp.body) as Map<String, dynamic>;
    return decoded;
  }
}
