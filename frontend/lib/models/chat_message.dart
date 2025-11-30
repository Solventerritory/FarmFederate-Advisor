class ChatMessage {
  final String text;
  final bool isUser;
  final String? imagePath; // local path (preview)
  final List<Prediction>? predictions; // optional backend predictions

  ChatMessage({
    required this.text,
    required this.isUser,
    this.imagePath,
    this.predictions,
  });
}

class Prediction {
  final String label;
  final double confidence; // 0..1

  Prediction({required this.label, required this.confidence});

  factory Prediction.fromJson(Map<String, dynamic> j) {
    return Prediction(
      label: j['label']?.toString() ?? '',
      confidence: (j['confidence'] is num) ? (j['confidence'] as num).toDouble() : 0.0,
    );
  }
}
