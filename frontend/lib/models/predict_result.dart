// lib/models/predict_result.dart
class PredictResult {
  final String? clientId;
  final List<String> labels;
  final List<double> probs;
  final List<Rationale> rationales;
  final List<String> advice;

  PredictResult({
    required this.clientId,
    required this.labels,
    required this.probs,
    required this.rationales,
    required this.advice,
  });

  factory PredictResult.fromJson(Map<String, dynamic> j) {
    return PredictResult(
      clientId: j['client_id'] as String?,
      labels: (j['labels'] as List? ?? []).map((e) => e.toString()).toList(),
      probs: (j['probs'] as List? ?? []).map((e) => (e as num).toDouble()).toList(),
      rationales: (j['rationales'] as List? ?? []).map((e) => Rationale.fromJson(e)).toList(),
      advice: (j['advice'] as List? ?? []).map((e) => e.toString()).toList(),
    );
  }
}

class Rationale {
  final String label;
  final double prob;
  final String reason;

  Rationale({
    required this.label,
    required this.prob,
    required this.reason,
  });

  factory Rationale.fromJson(Map<String, dynamic> j) {
    return Rationale(
      label: j['label']?.toString() ?? '',
      prob: (j['prob'] is num) ? (j['prob'] as num).toDouble() : 0.0,
      reason: j['reason']?.toString() ?? '',
    );
  }
}
