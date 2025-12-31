import 'dart:async';
import 'package:flutter/material.dart';
import '../models/sensor_data.dart';

class AnalyticsScreen extends StatefulWidget {
  final String apiBase;

  const AnalyticsScreen({Key? key, required this.apiBase}) : super(key: key);

  @override
  State<AnalyticsScreen> createState() => _AnalyticsScreenState();
}

class _AnalyticsScreenState extends State<AnalyticsScreen> {
  List<SensorData> _historicalData = [];
  bool _isLoading = true;
  String _selectedTimeRange = '7d';
  
  @override
  void initState() {
    super.initState();
    _loadHistoricalData();
  }

  Future<void> _loadHistoricalData() async {
    setState(() => _isLoading = true);
    
    // Simulate historical data - in production, this would fetch from backend
    await Future.delayed(const Duration(seconds: 1));
    
    // Generate mock historical data for demonstration
    final now = DateTime.now();
    _historicalData = List.generate(30, (index) {
      return SensorData(
        soilMoisture: 45.0 + (index % 20) - 10,
        soilTemperature: 22.0 + (index % 10) - 5,
        soilPH: 6.5 + (index % 4) * 0.2 - 0.4,
        humidity: 60.0 + (index % 15) - 7,
        airTemperature: 25.0 + (index % 12) - 6,
        lightIntensity: 80.0 + (index % 25) - 12,
        timestamp: now.subtract(Duration(days: 30 - index)),
      );
    });
    
    setState(() => _isLoading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('Historical Analytics'),
        actions: [
          PopupMenuButton<String>(
            initialValue: _selectedTimeRange,
            onSelected: (value) {
              setState(() => _selectedTimeRange = value);
              _loadHistoricalData();
            },
            itemBuilder: (context) => [
              const PopupMenuItem(value: '24h', child: Text('Last 24 Hours')),
              const PopupMenuItem(value: '7d', child: Text('Last 7 Days')),
              const PopupMenuItem(value: '30d', child: Text('Last 30 Days')),
              const PopupMenuItem(value: '90d', child: Text('Last 90 Days')),
            ],
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildSummaryCards(),
                  const SizedBox(height: 24),
                  _buildTrendAnalysis(),
                  const SizedBox(height: 24),
                  _buildPredictiveInsights(),
                  const SizedBox(height: 24),
                  _buildDataQuality(),
                ],
              ),
            ),
    );
  }

  Widget _buildSummaryCards() {
    if (_historicalData.isEmpty) return const SizedBox();
    
    final avgMoisture = _historicalData.map((e) => e.soilMoisture).reduce((a, b) => a + b) / _historicalData.length;
    final avgTemp = _historicalData.map((e) => e.airTemperature).reduce((a, b) => a + b) / _historicalData.length;
    final avgPH = _historicalData.map((e) => e.soilPH).reduce((a, b) => a + b) / _historicalData.length;
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Summary Statistics',
          style: TextStyle(
            color: Colors.white,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: _buildStatCard(
                'Avg Moisture',
                '${avgMoisture.toStringAsFixed(1)}%',
                Icons.water_drop,
                Colors.blue,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildStatCard(
                'Avg Temperature',
                '${avgTemp.toStringAsFixed(1)}°C',
                Icons.thermostat,
                Colors.orange,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildStatCard(
                'Avg pH',
                avgPH.toStringAsFixed(1),
                Icons.science,
                Colors.purple,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildStatCard(String label, String value, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Icon(icon, color: color, size: 32),
          const SizedBox(height: 8),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            label,
            style: const TextStyle(
              color: Colors.white60,
              fontSize: 11,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildTrendAnalysis() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.trending_up, color: Colors.green, size: 24),
              SizedBox(width: 12),
              Text(
                'Trend Analysis',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildTrendItem(
            'Soil Moisture',
            'Increasing by 2.3% over last 7 days',
            Icons.arrow_upward,
            Colors.green,
          ),
          const SizedBox(height: 12),
          _buildTrendItem(
            'Temperature',
            'Stable within optimal range',
            Icons.check_circle,
            Colors.blue,
          ),
          const SizedBox(height: 12),
          _buildTrendItem(
            'pH Level',
            'Slight decrease detected',
            Icons.arrow_downward,
            Colors.orange,
          ),
        ],
      ),
    );
  }

  Widget _buildTrendItem(String label, String description, IconData icon, Color color) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: color.withOpacity(0.2),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, color: color, size: 20),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                description,
                style: const TextStyle(
                  color: Colors.white60,
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildPredictiveInsights() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [
            Color(0xFF667eea),
            Color(0xFF764ba2),
          ],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.lightbulb, color: Colors.white, size: 24),
              SizedBox(width: 12),
              Text(
                'Predictive Insights',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildInsightItem(
            '🌱 Optimal Growth Window',
            'Next 3 days show ideal conditions for planting',
          ),
          const SizedBox(height: 12),
          _buildInsightItem(
            '💧 Irrigation Forecast',
            'Recommend reducing water by 15% based on humidity trends',
          ),
          const SizedBox(height: 12),
          _buildInsightItem(
            '🌡️ Temperature Alert',
            'Potential heat stress in 2 days - prepare shade covers',
          ),
        ],
      ),
    );
  }

  Widget _buildInsightItem(String title, String description) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            description,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDataQuality() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.verified, color: Colors.green, size: 24),
              SizedBox(width: 12),
              Text(
                'Data Quality Metrics',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildQualityMetric('Data Completeness', 98.5, Colors.green),
          const SizedBox(height: 12),
          _buildQualityMetric('Sensor Accuracy', 96.2, Colors.green),
          const SizedBox(height: 12),
          _buildQualityMetric('Network Reliability', 94.8, Colors.blue),
        ],
      ),
    );
  }

  Widget _buildQualityMetric(String label, double value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 14,
              ),
            ),
            Text(
              '${value.toStringAsFixed(1)}%',
              style: TextStyle(
                color: color,
                fontSize: 14,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        LinearProgressIndicator(
          value: value / 100,
          backgroundColor: Colors.white12,
          valueColor: AlwaysStoppedAnimation(color),
        ),
      ],
    );
  }
}
