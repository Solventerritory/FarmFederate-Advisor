import 'package:flutter/material.dart';
import 'package:percent_indicator/linear_percent_indicator.dart';

class ResultCard extends StatelessWidget {
  final String label;
  final double prob;
  const ResultCard({super.key, required this.label, required this.prob});
  String prettyPercent(double p) => '${(p * 100).toStringAsFixed(1)}%';
  Color barColor(double p) {
    if (p >= 0.75) return Colors.redAccent;
    if (p >= 0.4) return Colors.orange;
    return Colors.green;
  }
  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        title: Text(label.replaceAll('_',' ').toUpperCase()),
        subtitle: Padding(
          padding: const EdgeInsets.only(top:8.0),
          child: LinearPercentIndicator(
            lineHeight: 14,
            percent: prob.clamp(0.0,1.0),
            center: Text(prettyPercent(prob), style: const TextStyle(fontSize: 12, color: Colors.white)),
            progressColor: barColor(prob),
            backgroundColor: Colors.grey.shade300,
            barRadius: const Radius.circular(8),
            animation: true,
            animationDuration: 600,
          ),
        ),
      ),
    );
  }
}
