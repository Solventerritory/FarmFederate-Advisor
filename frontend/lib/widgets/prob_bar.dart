import 'package:flutter/material.dart';

class ProbBar extends StatelessWidget {
  final double value; // 0..1
  final String label;
  ProbBar({required this.value, required this.label});
  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label),
        SizedBox(height: 4),
        LinearProgressIndicator(value: value),
        SizedBox(height: 8),
      ],
    );
  }
}
