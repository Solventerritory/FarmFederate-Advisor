// This file re-exports platform-specific MQTT implementation in real app.
// For web-safe minimal app we provide a no-op stub so code compiles.

typedef SensorCallback = void Function(Map<String, dynamic> sensors);

class MqttService {
  final String host;
  final SensorCallback? onSensor;
  MqttService({required this.host, this.onSensor});

  Future<void> connect() async {
    // stub: on web we don't connect to MQTT.
    return;
  }

  void disconnect() {
    // stub
  }

  Future<void> publish(String topic, String payload) async {
    // stub
  }
}
