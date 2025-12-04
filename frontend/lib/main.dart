// frontend/lib/main.dart
import 'dart:convert';
import 'dart:typed_data';
import 'dart:async';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

// Web-only import for file/camera input
// ignore: avoid_web_libraries_in_flutter
import 'dart:html' as html;

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const FarmFederateApp());
}

class FarmFederateApp extends StatelessWidget {
  const FarmFederateApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FarmFederate Advisor',
      theme: ThemeData(primarySwatch: Colors.deepPurple),
      home: const HomePage(),
      debugShowCheckedModeBanner: true,
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with TickerProviderStateMixin {
  // Configure your backend base URL here:
  // If you're running the web app in the same machine as the backend, use:
  //   http://localhost:8000
  // If using Android emulator hit host PC: http://10.0.2.2:8000
  final String apiBase = 'http://localhost:8000';

  late TabController _tabs;

  // Chat state
  final TextEditingController _textController = TextEditingController();
  String _adviceText = '';
  List<Map<String, dynamic>> _scores = [];
  List<String> _activeLabels = [];
  String _status = 'ready';
  Uint8List? _selectedImageBytes;
  String? _selectedImageName;

  // Hardware dashboard state (will poll /sensors/latest if available)
  Map<String, dynamic> _lastSensors = {};
  Timer? _sensorPoller;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 2, vsync: this);

    // start polling hardware sensors endpoint (if exists); safe if endpoint 404s
    _sensorPoller = Timer.periodic(const Duration(seconds: 5), (_) => _fetchLatestSensors());
  }

  @override
  void dispose() {
    _tabs.dispose();
    _textController.dispose();
    _sensorPoller?.cancel();
    super.dispose();
  }

  // -------------------------
  // Web file/camera picker
  // -------------------------
  Future<void> _pickImageWeb({bool camera = false}) async {
    // create input element
    final html.InputElement input = html.document.createElement('input') as html.InputElement;
    input.type = 'file';
    input.accept = 'image/*';
    if (camera) {
      // on mobile browsers, "capture" attribute hints to use camera
      input.setAttribute('capture', 'environment');
    }
    input.click();

    await input.onChange.first;
    final files = input.files;
    if (files == null || files.isEmpty) return;

    final file = files[0];
    final reader = html.FileReader();
    reader.readAsArrayBuffer(file);
    await reader.onLoad.first;
    final result = reader.result;
    if (result is! ByteBuffer) return;
    final bytes = result.asUint8List();
    setState(() {
      _selectedImageBytes = bytes;
      _selectedImageName = file.name;
    });
  }

  // For mobile/native: you can add image_picker or file_picker plugin with conditional imports
  // and then implement a _pickImageNative() method. For now this app is web-focused.

  // -------------------------
  // Call backend predict
  // -------------------------
  Future<void> _sendPredict() async {
    final text = _textController.text.trim();
    if (text.isEmpty && _selectedImageBytes == null) {
      setState(() {
        _status = 'Please enter text or attach an image';
      });
      return;
    }

    setState(() {
      _status = 'sending...';
      _adviceText = '';
      _scores = [];
      _activeLabels = [];
    });

    try {
      final uri = Uri.parse('$apiBase/predict');

      // If we have image bytes, send multipart/form-data; else send JSON
      http.StreamedResponse streamedResp;
      if (_selectedImageBytes != null) {
        // create multipart request
        final req = http.MultipartRequest('POST', uri);
        // text fields
        req.fields['text'] = text;
        req.fields['sensors'] = ''; // hardware provides sensors; we don't send here
        req.fields['client_id'] = 'web_client';

        // image part
        final imgName = _selectedImageName ?? 'upload.jpg';
        req.files.add(http.MultipartFile.fromBytes('image', _selectedImageBytes!,
            filename: imgName, contentType: null)); // contentType optional

        streamedResp = await req.send().timeout(const Duration(seconds: 20));
      } else {
        // JSON POST
        final resp = await http
            .post(uri,
                headers: {'Content-Type': 'application/json'},
                body: jsonEncode({"text": text, "sensors": "", "client_id": "web_client"}))
            .timeout(const Duration(seconds: 12));
        if (resp.statusCode != 200) {
          throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
        }
        final decoded = jsonDecode(resp.body) as Map<String, dynamic>;
        _handlePredictResponse(decoded);
        return;
      }

      // handle multipart streamed response
      final resp = await http.Response.fromStream(streamedResp).timeout(const Duration(seconds: 30));
      if (resp.statusCode != 200) {
        throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
      }
      final decoded = jsonDecode(resp.body) as Map<String, dynamic>;
      _handlePredictResponse(decoded);
    } catch (e, st) {
      setState(() {
        _status = 'error: $e';
      });
      // print to browser console as well
      // ignore: avoid_print
      print('predict error: $e\n$st');
    }
  }

  void _handlePredictResponse(Map<String, dynamic> decoded) {
    final result = decoded['result'] as Map<String, dynamic>?;
    final advice = decoded['advice'] as String?;
    setState(() {
      _status = 'ok';
      _adviceText = advice ?? '';
      _scores = [];
      _activeLabels = [];
      if (result != null) {
        final allScores = (result['all_scores'] as List<dynamic>?)
            ?.map((e) => Map<String, dynamic>.from(e as Map))
            .toList();
        final active = (result['active_labels'] as List<dynamic>?)
            ?.map((e) => (e as Map)['label']?.toString() ?? '')
            .where((s) => s.isNotEmpty)
            .toList();
        if (allScores != null) _scores = allScores;
        if (active != null) _activeLabels = active;
      }
    });
  }

  // -------------------------
  // Poll sensors (optional)
  // -------------------------
  Future<void> _fetchLatestSensors() async {
    try {
      final resp = await http.get(Uri.parse('$apiBase/sensors/latest')).timeout(const Duration(seconds: 4));
      if (resp.statusCode == 200) {
        final decoded = jsonDecode(resp.body) as Map<String, dynamic>;
        setState(() {
          _lastSensors = decoded;
        });
      } else {
        // keep last sensors; no change
      }
    } catch (_) {
      // ignore network errors; many backends don't expose sensors endpoint yet
    }
  }

  // -------------------------
  // UI
  // -------------------------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(children: [
          const FlutterLogo(size: 32),
          const SizedBox(width: 12),
          const Text('FarmFederate'),
        ]),
        bottom: TabBar(controller: _tabs, tabs: const [
          Tab(text: 'Chat'),
          Tab(text: 'Hardware'),
        ]),
      ),
      body: TabBarView(controller: _tabs, children: [
        _buildChatTab(),
        _buildHardwareTab(),
      ]),
    );
  }

  Widget _buildChatTab() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(children: [
        // input
        TextField(
          controller: _textController,
          decoration: const InputDecoration(hintText: 'Describe the issue (or leave blank to send image only)'),
          minLines: 1,
          maxLines: 3,
        ),
        const SizedBox(height: 12),

        // image preview + controls
        Row(children: [
          if (_selectedImageBytes != null)
            Container(
              width: 96,
              height: 96,
              decoration: BoxDecoration(border: Border.all(color: Colors.grey.shade300)),
              child: Image.memory(_selectedImageBytes!, fit: BoxFit.cover),
            )
          else
            Container(
              width: 96,
              height: 96,
              decoration: BoxDecoration(border: Border.all(color: Colors.grey.shade300)),
              child: const Center(child: Text('No image')),
            ),
          const SizedBox(width: 12),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            ElevatedButton.icon(
              onPressed: () => _onPickImage(camera: true),
              icon: const Icon(Icons.camera_alt),
              label: const Text('Camera'),
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              onPressed: () => _onPickImage(camera: false),
              icon: const Icon(Icons.photo_library),
              label: const Text('Upload'),
            ),
            const SizedBox(height: 8),
            TextButton(
              onPressed: () {
                setState(() {
                  _selectedImageBytes = null;
                  _selectedImageName = null;
                });
              },
              child: const Text('Clear image'),
            ),
          ])
        ]),
        const SizedBox(height: 12),

        // send
        Row(children: [
          ElevatedButton.icon(
            onPressed: _sendPredict,
            icon: const Icon(Icons.send),
            label: const Text('Send to backend'),
          ),
          const SizedBox(width: 12),
          Text('Status: $_status'),
        ]),

        const SizedBox(height: 16),
        Expanded(
            child: SingleChildScrollView(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            if (_activeLabels.isNotEmpty) Wrap(
              spacing: 8,
              runSpacing: 8,
              children: _activeLabels.map((l) => Chip(label: Text(l))).toList(),
            ),
            const SizedBox(height: 12),
            if (_scores.isNotEmpty) ...[
              const Text('All scores:', style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 6),
              Wrap(
                spacing: 8,
                children: _scores.map((s) {
                  final label = s['label']?.toString() ?? '';
                  final prob = (s['prob'] is num) ? (s['prob'] as num).toDouble() : 0.0;
                  return Chip(label: Text('$label: ${(prob * 100).toStringAsFixed(1)}%'));
                }).toList(),
              ),
              const SizedBox(height: 12),
            ],
            const Text('Advice:', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 6),
            Text(_adviceText.isNotEmpty ? _adviceText : 'No advice yet'),
            const SizedBox(height: 24),
          ]),
        )),
      ]),
    );
  }

  Widget _buildHardwareTab() {
    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(children: [
        Row(children: [
          const FlutterLogo(size: 32),
          const SizedBox(width: 12),
          const Text('Hardware Dashboard', style: TextStyle(fontSize: 18)),
          const Spacer(),
          ElevatedButton(onPressed: _fetchLatestSensors, child: const Text('Refresh')),
        ]),
        const SizedBox(height: 12),
        if (_lastSensors.isEmpty)
          const Text('No sensor data available (polling /sensors/latest every 5s).')
        else
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Client: ${_lastSensors['client_id'] ?? 'unknown'}'),
                const SizedBox(height: 6),
                Text('Soil moisture: ${_lastSensors['soil_moisture'] ?? '-'}'),
                const SizedBox(height: 4),
                Text('Temperature: ${_lastSensors['temp'] ?? '-'}'),
                const SizedBox(height: 4),
                Text('Humidity: ${_lastSensors['humidity'] ?? '-'}'),
                const SizedBox(height: 4),
                Text('VPD: ${_lastSensors['vpd'] ?? '-'}'),
              ]),
            ),
          )
      ]),
    );
  }

  // Helper that dispatches to web/native pickers
  Future<void> _onPickImage({required bool camera}) async {
    if (kIsWeb) {
      await _pickImageWeb(camera: camera);
    } else {
      // Native (Android/iOS): integrate image_picker or file_picker here.
      // Example: use `image_picker` plugin, then load bytes and set _selectedImageBytes.
      // For now show a dialog telling user to add native plugin.
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Native pick not implemented'),
          content: const Text('This web build handles camera/upload. For native builds add image_picker plugin and implement _pickImageNative().'),
          actions: [TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('OK'))],
        ),
      );
    }
  }
}
