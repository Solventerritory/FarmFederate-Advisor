// lib/widgets/chat_bubble.dart
import 'package:flutter/material.dart';
import '../services/chat_state.dart';
import '../models/predict_result.dart';

class ChatBubble extends StatelessWidget {
  final ChatMessage message;
  const ChatBubble({Key? key, required this.message}) : super(key: key);

  Widget _buildPrediction(PredictResult pred) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 8,
          children: pred.labels.map((l) => Chip(label: Text(l))).toList(),
        ),
        const SizedBox(height: 6),
        Text("Probs: ${pred.probs.map((p) => p.toStringAsFixed(2)).join(", ")}"),
        const SizedBox(height: 6),
        const Text("Advice:", style: TextStyle(fontWeight: FontWeight.bold)),
        ...pred.advice.map((a) => Text("• $a")).toList(),
        if (pred.rationales.isNotEmpty) ...[
          const SizedBox(height: 6),
          const Text("Rationales:", style: TextStyle(fontWeight: FontWeight.bold)),
          ...pred.rationales.map((r) => Text("- ${r.label}: ${r.reason} (p=${r.prob.toStringAsFixed(2)})")).toList(),
        ],
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final isUser = message.fromUser;
    final align = isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start;
    final bg = isUser ? Colors.green[100] : Colors.white;
    final radius = isUser
        ? const BorderRadius.only(topLeft: Radius.circular(12), topRight: Radius.circular(12), bottomLeft: Radius.circular(12))
        : const BorderRadius.only(topLeft: Radius.circular(12), topRight: Radius.circular(12), bottomRight: Radius.circular(12));

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: [
          if (!isUser)
            const CircleAvatar(child: Icon(Icons.agriculture)),
          const SizedBox(width: 8),
          Flexible(
            child: Container(
              decoration: BoxDecoration(color: bg, borderRadius: radius, boxShadow: [
                BoxShadow(color: Colors.black12, blurRadius: 3, offset: Offset(0,1))
              ]),
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: align,
                children: [
                  if (message.image != null) ...[
                    ClipRRect(borderRadius: BorderRadius.circular(8), child: Image.file(message.image!, height: 160, fit: BoxFit.cover)),
                    const SizedBox(height: 8),
                  ],
                  Text(message.text, style: const TextStyle(fontSize: 14)),
                  const SizedBox(height: 8),
                  if (message.prediction != null) _buildPrediction(message.prediction!),
                ],
              ),
            ),
          ),
          const SizedBox(width: 8),
          if (isUser) const CircleAvatar(child: Icon(Icons.person)),
        ],
      ),
    );
  }
}
