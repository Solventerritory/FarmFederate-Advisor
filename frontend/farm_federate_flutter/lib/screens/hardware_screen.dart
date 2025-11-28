// lib/screens/hardware_screen.dart
import 'package:flutter/material.dart';
import '../services/hw_service.dart';

class HardwareScreen extends StatefulWidget {
  const HardwareScreen({super.key});
  @override
  State<HardwareScreen> createState() => _HardwareScreenState();
}

class _HardwareScreenState extends State<HardwareScreen> {
  Map<String, dynamic>? telemetry;
  bool loading = false;

  Future<void> loadLatest() async {
    setState(() => loading = true);
    try {
      telemetry = await HWService.getLatest("esp32-01");
    } catch (e) {
      telemetry = null;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to fetch telemetry: $e')));
    } finally { setState(() => loading = false); }
  }

  Future<void> sendControl(String pin, int value) async {
    try {
      await HWService.control("esp32-01", pin, value);
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Control sent')));
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Control failed: $e')));
    }
  }

  @override
  void initState() {
    super.initState();
    loadLatest();
  }

  Widget row(String k, dynamic v) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 6),
    child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text(k, style: const TextStyle(fontWeight: FontWeight.w600)),
      Text(v?.toString() ?? "-", style: const TextStyle(color: Colors.black54)),
    ]),
  );

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Hardware Dashboard')),
      body: RefreshIndicator(
        onRefresh: () async => loadLatest(),
        child: ListView(padding: const EdgeInsets.all(16), children: [
          Card(
            child: Padding(padding: const EdgeInsets.all(12), child: Column(children: [
              const Text('Device: esp32-01', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              if (loading) const LinearProgressIndicator(),
              const SizedBox(height: 8),
              row('Soil Moisture', telemetry?['soil_moisture'] ?? 'N/A'),
              row('Air Humidity', telemetry?['air_humidity'] ?? 'N/A'),
              row('Temperature (C)', telemetry?['temp_c'] ?? 'N/A'),
              row('Flow (unit)', telemetry?['flow_rate'] ?? 'N/A'),
              const SizedBox(height: 12),
              Row(children: [
                ElevatedButton.icon(onPressed: () => sendControl('V1', 1), icon: const Icon(Icons.play_arrow), label: const Text('Open Valve')),
                const SizedBox(width: 12),
                ElevatedButton.icon(onPressed: () => sendControl('V1', 0), icon: const Icon(Icons.stop), label: const Text('Close Valve')),
              ]),
            ])),
          )
        ]),
      ),
    );
  }
}
