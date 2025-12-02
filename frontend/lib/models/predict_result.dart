// lib/models/predict_result.dart

class PredictResult {
  final List<String> labels;
  final Map<String, double> probs;
  final String advice;

  PredictResult({
    required this.labels,
    required this.probs,
    required this.advice,
  });

  factory PredictResult.fromJson(Map<String, dynamic> json) {
    // labels: ["water_stress", ...]
    final labels = (json['labels'] as List<dynamic>? ?? [])
        .map((e) => e.toString())
        .toList();

    // probs: {"water_stress": 0.51, ...}
    final probsRaw = json['probs'] as Map<String, dynamic>? ?? {};
    final probs = <String, double>{};
    probsRaw.forEach((k, v) {
      probs[k] = (v as num).toDouble();
    });

    final advice = (json['advice'] ?? '').toString();

    return PredictResult(
      labels: labels,
      probs: probs,
      advice: advice,
    );
  }
}
