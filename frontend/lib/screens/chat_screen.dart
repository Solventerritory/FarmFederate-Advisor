// lib/screens/chat_screen.dart

import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/predict_result.dart';
import '../services/api_service.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatMessage {
  final bool isUser;
  final String userText;
  final PredictResult? result;

  _ChatMessage({
    required this.isUser,
    required this.userText,
    this.result,
  });
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _textController = TextEditingController();
  final TextEditingController _sensorsController = TextEditingController();

  final List<_ChatMessage> _messages = [];
  bool _isSending = false;

  // image state
  Uint8List? _pickedImageBytes;
  String? _pickedImageName;

  final ImagePicker _picker = ImagePicker();

  @override
  void dispose() {
    _textController.dispose();
    _sensorsController.dispose();
    super.dispose();
  }

  Future<void> _pickFromGallery() async {
    try {
      final XFile? xfile =
          await _picker.pickImage(source: ImageSource.gallery, imageQuality: 80);
      if (xfile == null) return;

      final bytes = await xfile.readAsBytes();
      setState(() {
        _pickedImageBytes = bytes;
        _pickedImageName = xfile.name;
      });
    } catch (e) {
      debugPrint("Gallery pick error: $e");
    }
  }

  Future<void> _pickFromCamera() async {
    try {
      final XFile? xfile =
          await _picker.pickImage(source: ImageSource.camera, imageQuality: 80);
      if (xfile == null) return;

      final bytes = await xfile.readAsBytes();
      setState(() {
        _pickedImageBytes = bytes;
        _pickedImageName = xfile.name;
      });
    } catch (e) {
      debugPrint("Camera pick error: $e");
    }
  }

  Future<void> _sendMessage() async {
    final text = _textController.text.trim();
    final sensors = _sensorsController.text.trim();

    if (text.isEmpty &&
        (_pickedImageBytes == null || _pickedImageBytes!.isEmpty)) {
      return;
    }

    // user bubble
    setState(() {
      _messages.add(
        _ChatMessage(
          isUser: true,
          userText: text.isEmpty ? "(image only)" : text,
        ),
      );
      _isSending = true;
      _textController.clear();
    });

    try {
      final result = await ApiService.predict(
        text: text,
        sensors: sensors,
        imageBytes: _pickedImageBytes,
        imageName: _pickedImageName,
      );

      setState(() {
        _messages.add(
          _ChatMessage(
            isUser: false,
            userText: "",
            result: result,
          ),
        );
      });
    } catch (e) {
      setState(() {
        _messages.add(
          _ChatMessage(
            isUser: false,
            userText: "Error: $e",
          ),
        );
      });
    } finally {
      setState(() {
        _isSending = false;
        _pickedImageBytes = null;
        _pickedImageName = null;
      });
    }
  }

  Widget _farmerBubble(_ChatMessage msg) {
    return Align(
      alignment: Alignment.centerRight,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
        decoration: BoxDecoration(
          color: Colors.green[100],
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(msg.userText),
      ),
    );
  }

  Widget _advisorBubble(_ChatMessage msg) {
    if (msg.result == null) {
      // simple text bubble (errors, etc.)
      return Align(
        alignment: Alignment.centerLeft,
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
          decoration: BoxDecoration(
            color: Colors.purple[50],
            borderRadius: BorderRadius.circular(16),
          ),
          child: Text(msg.userText),
        ),
      );
    }

    final r = msg.result!;
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.purple[50],
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: ISSUE_LABELS.map((lab) {
                final active = r.labels.contains(lab);
                return Chip(
                  label: Text(lab),
                  backgroundColor:
                      active ? Colors.purple[300] : Colors.grey[200],
                );
              }).toList(),
            ),
            const SizedBox(height: 8),
            if (r.probs.isNotEmpty) ...[
              const Text(
                "Probs:",
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              Wrap(
                spacing: 8,
                runSpacing: 4,
                children: ISSUE_LABELS.map((lab) {
                  final p = r.probs[lab] ?? 0.0;
                  return Text("$lab: ${(p * 100).toStringAsFixed(1)}%");
                }).toList(),
              ),
              const SizedBox(height: 8),
            ],
            const Text(
              "Advice:",
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            Text(r.advice),
          ],
        ),
      ),
    );
  }

  Widget _buildImagePreview() {
    if (_pickedImageBytes == null) return const SizedBox.shrink();

    return Padding(
      padding: const EdgeInsets.only(top: 6.0, bottom: 2.0),
      child: Row(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.memory(
              _pickedImageBytes!,
              height: 70,
              width: 70,
              fit: BoxFit.cover,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              _pickedImageName ?? "selected image",
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          IconButton(
            icon: const Icon(Icons.close),
            onPressed: () {
              setState(() {
                _pickedImageBytes = null;
                _pickedImageName = null;
              });
            },
            tooltip: "Remove image",
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("FarmFederate — Advisor Chat"),
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.only(top: 12, bottom: 80),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final msg = _messages[index];
                return msg.isUser ? _farmerBubble(msg) : _advisorBubble(msg);
              },
            ),
          ),
          const Divider(height: 1),
          SafeArea(
            child: Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 8.0, vertical: 6.0),
              child: Column(
                children: [
                  _buildImagePreview(),
                  TextField(
                    controller: _textController,
                    minLines: 1,
                    maxLines: 4,
                    decoration: const InputDecoration(
                      border: OutlineInputBorder(),
                      labelText:
                          "Describe the issue (or leave blank for image-only)",
                    ),
                  ),
                  const SizedBox(height: 6),
                  TextField(
                    controller: _sensorsController,
                    decoration: const InputDecoration(
                      border: OutlineInputBorder(),
                      labelText:
                          "Sensors (optional) e.g. soil_moisture=24,temp=35,humidity=68",
                    ),
                  ),
                  const SizedBox(height: 6),
                  Row(
                    children: [
                      IconButton(
                        icon: const Icon(Icons.photo_library_outlined),
                        tooltip: "Pick image from gallery/files",
                        onPressed: _isSending ? null : _pickFromGallery,
                      ),
                      IconButton(
                        icon: const Icon(Icons.photo_camera_outlined),
                        tooltip: "Take photo with camera",
                        onPressed: _isSending ? null : _pickFromCamera,
                      ),
                      const Spacer(),
                      IconButton(
                        icon: _isSending
                            ? const SizedBox(
                                height: 22,
                                width: 22,
                                child:
                                    CircularProgressIndicator(strokeWidth: 2),
                              )
                            : const Icon(Icons.send),
                        onPressed: _isSending ? null : _sendMessage,
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// SAME ORDER as backend
const ISSUE_LABELS = [
  "water_stress",
  "nutrient_def",
  "pest_risk",
  "disease_risk",
  "heat_stress",
];
