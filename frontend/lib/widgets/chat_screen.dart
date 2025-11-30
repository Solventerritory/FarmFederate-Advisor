// lib/widgets/chat_screen.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import '../services/chat_state.dart';
import '../widgets/chat_bubble.dart';
import '../services/api_service.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({Key? key}) : super(key: key);

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _controller = TextEditingController();
  String _sensors = "";
  File? _pickedImage;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage() async {
    final XFile? xfile = await _picker.pickImage(source: ImageSource.gallery);
    if (xfile != null) {
      setState(() {
        _pickedImage = File(xfile.path);
      });
    }
  }

  Future<void> _takePhoto() async {
    final XFile? xfile = await _picker.pickImage(source: ImageSource.camera);
    if (xfile != null) {
      setState(() {
        _pickedImage = File(xfile.path);
      });
    }
  }

  Future<void> _send() async {
    final text = _controller.text.trim();
    if (text.isEmpty && _pickedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Enter text or pick an image.")));
      return;
    }

    final chatState = context.read<ChatState>();
    chatState.addUserMessage(text.isEmpty ? "<image>" : text, image: _pickedImage);
    _controller.clear();

    await chatState.sendToServer(text.isEmpty ? "(image only)" : text, sensors: _sensors, image: _pickedImage);

    setState(() {
      _pickedImage = null;
    });
  }

  Widget _buildComposer() {
    final loading = context.watch<ChatState>().loading;
    return SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_pickedImage != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              child: Stack(
                children: [
                  ClipRRect(borderRadius: BorderRadius.circular(8), child: Image.file(_pickedImage!, height: 140, fit: BoxFit.cover)),
                  Positioned(
                    top: 6,
                    right: 6,
                    child: InkWell(
                      onTap: () => setState(() { _pickedImage = null; }),
                      child: const CircleAvatar(backgroundColor: Colors.black54, radius: 14, child: Icon(Icons.close, size: 16, color: Colors.white)),
                    ),
                  )
                ],
              ),
            ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
            child: Row(
              children: [
                IconButton(onPressed: _pickImage, icon: const Icon(Icons.photo)),
                IconButton(onPressed: _takePhoto, icon: const Icon(Icons.camera_alt)),
                Expanded(
                  child: TextField(
                    controller: _controller,
                    minLines: 1,
                    maxLines: 4,
                    decoration: const InputDecoration.collapsed(hintText: "Describe the issue (or leave blank for image-only)"),
                  ),
                ),
                IconButton(
                  onPressed: loading ? null : _send,
                  icon: loading ? const CircularProgressIndicator() : const Icon(Icons.send),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            child: TextField(
              onChanged: (v) => _sensors = v,
              decoration: const InputDecoration(
                hintText: "Sensors (optional) e.g. soil_moisture=24,temp=35,humidity=68",
                border: OutlineInputBorder(),
                isDense: true,
                contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
              ),
            ),
          )
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final chatState = context.watch<ChatState>();
    return Scaffold(
      appBar: AppBar(
        title: const Text("FarmFederate — Advisor Chat"),
        actions: [
          IconButton(
            onPressed: () {
              context.read<ChatState>().clear();
            },
            icon: const Icon(Icons.delete_forever),
            tooltip: "Clear chat",
          ),
          IconButton(
            onPressed: () {
              // quick toggle: switch baseUrl for emulator/devices (optional)
              showDialog(
                context: context,
                builder: (c) => AlertDialog(
                  title: const Text("Backend URL"),
                  content: TextFormField(
                    initialValue: ApiService.baseUrl,
                    onChanged: (v) => ApiService.baseUrl = v,
                  ),
                  actions: [
                    TextButton(onPressed: () => Navigator.pop(context), child: const Text("Close")),
                  ],
                ),
              );
            },
            icon: const Icon(Icons.settings_ethernet),
          )
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              reverse: true,
              padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
              itemCount: chatState.messages.length,
              itemBuilder: (ctx, idx) {
                final m = chatState.messages[idx];
                return ChatBubble(message: m);
              },
            ),
          ),
          _buildComposer(),
        ],
      ),
    );
  }
}
