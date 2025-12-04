class SensorMessage {
  final String topic;
  final Map<String, dynamic> data;
  final DateTime at;
  SensorMessage({required this.topic, required this.data, DateTime? at}) : this.at = at ?? DateTime.now();
}
