// lib/services/chat_state.dart
import 'dart:io';
import 'package:flutter/material.dart';
import '../models/predict_result.dart';
import 'api_service.dart';

class ChatMessage {
  final String text;
  final bool fromUser;
  final File? image;
  final PredictResult? prediction;

  ChatMessage({
    required this.text,
    required this.fromUser,
    this.image,
    this.prediction,
  });
}

class ChatState extends ChangeNotifier {
  final List<ChatMessage> _messages = [];
  bool _loading = false;

  List<ChatMessage> get messages => List.unmodifiable(_messages);
  bool get loading => _loading;

  void addUserMessage(String text, {File? image}) {
    _messages.insert(0, ChatMessage(text: text, fromUser: true, image: image));
    notifyListeners();
  }

  Future<void> sendToServer(String text, {String sensors = "", File? image}) async {
    _loading = true;
    notifyListeners();
    try {
      final result = await ApiService.predict(text: text, sensors: sensors, imageFile: image);
      // Add server reply (with prediction)
      _messages.insert(0, ChatMessage(text: result.advice.join("\n"), fromUser: false, image: image, prediction: result));
    } catch (e) {
      _messages.insert(0, ChatMessage(text: "Error: $e", fromUser: false, image: null, prediction: null));
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  void clear() {
    _messages.clear();
    notifyListeners();
  }
}
