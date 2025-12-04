import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import '../services/mqtt_service.dart';

class HardwareDashboard extends StatefulWidget {
  final String mqttHost;
  const HardwareDashboard({Key? key, required this.mqttHost}) : super(key: key);

  @override
  State<HardwareDashboard> createState() => _HardwareDashboardState();
}

class _HardwareDashboardState extends State<HardwareDashboard> {
  Map<String, dynamic> _lastSensors = {};
  late MqttService _mqtt;

  @override
  void initState() {
    super.initState();
    _mqtt = MqttService(host: widget.mqttHost, onSensor: _onSensor);
    // On web we do not connect (stub). On mobile you will replace implementation to actually connect.
    _mqtt.connect();
  }

  void _onSensor(Map<String, dynamic> s) {
    setState(() {
      _lastSensors = s;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (kIsWeb) {
      return Center(child: Padding(
        padding: EdgeInsets.all(20),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Image.asset('assets/logo.png', width: 96),
          SizedBox(height: 12),
          Text("Hardware dashboard (Web stub)", style: TextStyle(fontSize: 18)),
          SizedBox(height: 8),
          Text("Running on web — MQTT hardware integration is disabled here.\nRun the app on mobile/desktop for live sensor data.", textAlign: TextAlign.center),
        ]),
      ));
    }

    return Padding(
      padding: EdgeInsets.all(12),
      child: Column(children: [
        Row(children: [
          Image.asset('assets/logo.png', width: 48),
          SizedBox(width: 12),
          Text("Connected sensor data", style: TextStyle(fontSize: 18)),
        ]),
        SizedBox(height: 16),
        if (_lastSensors.isEmpty) Text("No sensors yet"),
        if (_lastSensors.isNotEmpty) ...[
          ListTile(title: Text("Client"), subtitle: Text("${_lastSensors['client_id'] ?? 'unknown'}")),
          ListTile(title: Text("Soil moisture"), subtitle: Text("${_lastSensors['soil_moisture'] ?? '-'}")),
          ListTile(title: Text("Temp (°C)"), subtitle: Text("${_lastSensors['temp'] ?? '-'}")),
          ListTile(title: Text("Humidity (%)"), subtitle: Text("${_lastSensors['humidity'] ?? '-'}")),
          ListTile(title: Text("VPD"), subtitle: Text("${_lastSensors['vpd'] ?? '-'}")),
        ],
      ]),
    );
  }
}
