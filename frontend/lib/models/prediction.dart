class LabelScore {
  final String label;
  final double prob;
  final double threshold;
  LabelScore({required this.label, required this.prob, required this.threshold});
  factory LabelScore.fromMap(Map m) => LabelScore(
      label: m['label'] as String,
      prob: (m['prob'] as num).toDouble(),
      threshold: (m['threshold'] as num).toDouble());
}

class PredictionResult {
  final List<LabelScore> active;
  final List<LabelScore> allScores;
  final List<double> rawProbs;
  final String advice;
  PredictionResult({
    required this.active,
    required this.allScores,
    required this.rawProbs,
    required this.advice,
  });
  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    final res = json['result'] ?? {};
    final active = (res['active_labels'] as List? ?? [])
        .map((e) => LabelScore.fromMap(e)).toList();
    final all = (res['all_scores'] as List? ?? [])
        .map((e) => LabelScore.fromMap(e)).toList();
    final raw = (res['raw_probs'] as List? ?? [])
        .map<double>((e) => (e as num).toDouble()).toList();
    final advice = json['advice'] ?? "";
    return PredictionResult(active: active, allScores: all, rawProbs: raw, advice: advice);
  }
}
