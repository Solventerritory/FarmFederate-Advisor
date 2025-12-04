// lib/screens/chat_screen.dart
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';

/// ChatScreen: send text + optional image to backend /predict and show results.
/// Usage: ChatScreen(apiBase: "http://10.0.2.2:8000")
class ChatScreen extends StatefulWidget {
  final String apiBase;
  const ChatScreen({Key? key, required this.apiBase}) : super(key: key);

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _ctrl = TextEditingController();
  File? _imageFile;
  bool _loading = false;
  String _status = "idle";
  List<Map<String, dynamic>> _scores = []; // {label, prob(double)}
  String _advice = "";
  Map<String, dynamic> _debug = {};

  // Use image_picker for camera
  final ImagePicker _picker = ImagePicker();

  // ----- UI helpers -----
  void _setStatusDirect(String s) {
    // helper that sets status (uses setState)
    setState(() => _status = s);
  }

  // Pick from camera
  Future<void> _pickCamera() async {
    try {
      final XFile? photo =
          await _picker.pickImage(source: ImageSource.camera, imageQuality: 80);
      if (photo == null) return;
      setState(() {
        _imageFile = File(photo.path);
      });
    } catch (e) {
      _setStatusDirect("camera error");
      debugPrint("camera error: $e");
    }
  }

  // Pick from file system
  Future<void> _pickFile() async {
    try {
      final res =
          await FilePicker.platform.pickFiles(type: FileType.image, withData: false);
      if (res == null || res.files.isEmpty) return;
      final p = res.files.first.path;
      if (p == null) return;
      setState(() {
        _imageFile = File(p);
      });
    } catch (e) {
      _setStatusDirect("file pick error");
      debugPrint("file pick error: $e");
    }
  }

  void _clearImage() => setState(() => _imageFile = null);

  // ----- Call backend -----
  Future<void> _send() async {
    final text = _ctrl.text.trim();
    if (text.isEmpty && _imageFile == null) {
      _setStatusDirect("provide text or image");
      return;
    }

    // Avoid nested setState by setting values directly inside setState
    setState(() {
      _loading = true;
      _status = "sending...";
      _scores = [];
      _advice = "";
      _debug = {};
    });

    try {
      final api = ApiService(baseUrl: widget.apiBase);
      final resp =
          await api.predict(text: text, imageFile: _imageFile, clientId: "flutter_client");

      if (resp == null) {
        _setStatusDirect("no response");
      } else {
        // parse response: expecting structure similar to server.py
        // result: { active_labels: [...], all_scores: [{label,prob,threshold}, ...], raw_probs: [...] }
        final result = resp['result'] ?? resp['data'] ?? {};
        final allScores = result['all_scores'] as List<dynamic>?;

        // build scores
        final parsed = (allScores ?? []).map<Map<String, dynamic>>((e) {
          final label = e['label'] ?? e['name'] ?? "unknown";
          double prob = 0.0;
          if (e['prob'] is num) {
            prob = (e['prob'] as num).toDouble();
          } else {
            prob = double.tryParse("${e['prob']}") ?? 0.0;
          }
          return {"label": label, "prob": prob};
        }).toList();

        setState(() {
          _scores = parsed;
          _advice = (resp['advice'] ?? result['advice'] ?? resp['advice_text'] ?? "") as String;
          _debug = resp['debug'] ?? {};
          _status = "ok";
        });
      }
    } catch (e, st) {
      _setStatusDirect("error");
      debugPrint("predict error: $e\n$st");
      setState(() {
        _advice = "Error: $e";
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  // ----- Render chips: highlight top-2 -----
  Widget _buildScoreChips() {
    if (_scores.isEmpty) return SizedBox.shrink();

    // Defensive copy and sort descending by prob
    final sorted = List<Map<String, dynamic>>.from(_scores)
      ..sort((a, b) => ((b['prob'] as num).compareTo(a['prob'] as num)));

    final top1Label = sorted.isNotEmpty ? (sorted[0]['label'] as String) : null;
    final top2Label = sorted.length > 1 ? (sorted[1]['label'] as String) : null;

    return Wrap(
      spacing: 12,
      runSpacing: 12,
      children: sorted.map((s) {
        final String label = s['label'] as String;
        final double prob = (s['prob'] as num).toDouble(); // 0..1
        final scoreText = "$label: ${(prob * 100).toStringAsFixed(1)}%";

        final bool isTop1 = (label == top1Label);
        final bool isTop2 = (label == top2Label);

        Color bgColor;
        Color textColor = Colors.black;
        List<BoxShadow>? shadows;

        if (isTop1) {
          bgColor = Colors.red.shade600;
          textColor = Colors.white;
          shadows = [BoxShadow(color: Colors.black26, blurRadius: 8)];
        } else if (isTop2) {
          bgColor = Colors.amber.shade400;
          textColor = Colors.black;
          shadows = [BoxShadow(color: Colors.black12, blurRadius: 4)];
        } else {
          bgColor = Colors.grey.shade100;
          textColor = Colors.black87;
          shadows = null;
        }

        return Container(
          padding: EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            color: bgColor,
            borderRadius: BorderRadius.circular(22),
            border: Border.all(color: Colors.grey.shade300),
            boxShadow: shadows,
          ),
          child: Text(
            scoreText,
            style: TextStyle(
              color: textColor,
              fontWeight: (isTop1 || isTop2) ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        );
      }).toList(),
    );
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final imagePreview = _imageFile != null
        ? Image.file(_imageFile!, width: 120, height: 120, fit: BoxFit.cover)
        : Container(
            width: 120,
            height: 120,
            color: Colors.grey.shade100,
            child: Center(child: Text("No image")),
          );

    return Scaffold(
      appBar: AppBar(
        title: Row(children: [
          // small placeholder logo
          Icon(Icons.agriculture, size: 28),
          SizedBox(width: 8),
          Text("FarmFederate — Chat"),
        ]),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(18),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          TextField(
            controller: _ctrl,
            minLines: 1,
            maxLines: 3,
            decoration: InputDecoration(
              hintText: "Describe the issue (or leave blank for image-only)",
              border: UnderlineInputBorder(),
            ),
          ),
          SizedBox(height: 18),

          // image preview and buttons
          Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            ClipRRect(borderRadius: BorderRadius.circular(6), child: imagePreview),
            SizedBox(width: 14),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                ElevatedButton.icon(
                  onPressed: _pickCamera,
                  icon: Icon(Icons.camera_alt),
                  label: Text("Camera"),
                ),
                SizedBox(height: 8),
                ElevatedButton.icon(
                  onPressed: _pickFile,
                  icon: Icon(Icons.upload_file),
                  label: Text("Upload"),
                ),
                TextButton(onPressed: _clearImage, child: Text("Clear image")),
              ]),
            ),
          ]),
          SizedBox(height: 18),

          // send + status
          Row(children: [
            ElevatedButton.icon(
              onPressed: _loading ? null : _send,
              icon: Icon(Icons.send),
              label: Text(_loading ? "Sending..." : "Send"),
            ),
            SizedBox(width: 12),
            Text("Status: $_status"),
          ]),
          SizedBox(height: 20),

          // scores chips (wrapped in a heading area)
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(height: 8),
              Text('All scores:', style: TextStyle(fontWeight: FontWeight.bold)),
              SizedBox(height: 8),
              _buildScoreChips(),
              SizedBox(height: 18),
            ],
          ),

          // Advice
          if (_advice.isNotEmpty) ...[
            Text("Advice:", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            SizedBox(height: 8),
            Text(_advice),
            SizedBox(height: 12),
          ],

          // debug
          if (_debug.isNotEmpty) ...[
            Text("Debug:", style: TextStyle(fontWeight: FontWeight.bold)),
            SizedBox(height: 6),
            Text(jsonEncode(_debug)),
          ],
        ]),
      ),
    );
  }
}

/// Simple API service to call backend /predict
class ApiService {
  final String baseUrl;
  ApiService({required this.baseUrl});

  /// Predict endpoint: sends text + optional image (multipart).
  /// Expects JSON response (parsed & returned). Returns null on failure.
  Future<Map<String, dynamic>?> predict({
    required String text,
    File? imageFile,
    String clientId = "flutter_client",
  }) async {
    final uri = Uri.parse("$baseUrl/predict");
    try {
      if (imageFile != null && await imageFile.exists()) {
        final request = http.MultipartRequest('POST', uri);
        request.fields['text'] = text;
        request.fields['client_id'] = clientId;
        // sensors intentionally omitted — comes from hardware via MQTT in your architecture.
        final multipartFile = await http.MultipartFile.fromPath('image', imageFile.path);
        request.files.add(multipartFile);
        final streamed = await request.send().timeout(Duration(seconds: 30));
        final resp = await http.Response.fromStream(streamed);
        if (resp.statusCode >= 200 && resp.statusCode < 300) {
          return jsonDecode(resp.body) as Map<String, dynamic>;
        } else {
          debugPrint("predict multipart failed ${resp.statusCode} ${resp.body}");
          return null;
        }
      } else {
        // JSON POST
        final resp = await http
            .post(uri,
                headers: {'Content-Type': 'application/json'},
                body: jsonEncode({'text': text, 'client_id': clientId}))
            .timeout(Duration(seconds: 20));
        if (resp.statusCode >= 200 && resp.statusCode < 300) {
          return jsonDecode(resp.body) as Map<String, dynamic>;
        } else {
          debugPrint("predict json failed ${resp.statusCode} ${resp.body}");
          return null;
        }
      }
    } catch (e) {
      debugPrint("ApiService.predict error: $e");
      // return null on error to avoid bubbling exceptions to UI
      return null;
    }
  }
}
