// Models for sensor data
class SensorData {
  final double soilMoisture;
  final double soilTemperature;
  final double soilPH;
  final double airTemperature;
  final double humidity;
  final double lightIntensity;
  final DateTime timestamp;

  SensorData({
    required this.soilMoisture,
    required this.soilTemperature,
    required this.soilPH,
    required this.airTemperature,
    required this.humidity,
    required this.lightIntensity,
    required this.timestamp,
  });

  factory SensorData.fromJson(Map<String, dynamic> json) {
    return SensorData(
      soilMoisture: (json['soil_moisture'] ?? 0).toDouble(),
      soilTemperature: (json['soil_temperature'] ?? 0).toDouble(),
      soilPH: (json['soil_ph'] ?? 0).toDouble(),
      airTemperature: (json['air_temperature'] ?? 0).toDouble(),
      humidity: (json['humidity'] ?? 0).toDouble(),
      lightIntensity: (json['light_intensity'] ?? 0).toDouble(),
      timestamp: DateTime.parse(json['timestamp'] ?? DateTime.now().toIso8601String()),
    );
  }
}

class ControlState {
  final bool waterPump;
  final bool pestControl;
  final bool heater;
  final bool cooler;
  final bool fan;
  final bool growLights;

  ControlState({
    this.waterPump = false,
    this.pestControl = false,
    this.heater = false,
    this.cooler = false,
    this.fan = false,
    this.growLights = false,
  });

  ControlState copyWith({
    bool? waterPump,
    bool? pestControl,
    bool? heater,
    bool? cooler,
    bool? fan,
    bool? growLights,
  }) {
    return ControlState(
      waterPump: waterPump ?? this.waterPump,
      pestControl: pestControl ?? this.pestControl,
      heater: heater ?? this.heater,
      cooler: cooler ?? this.cooler,
      fan: fan ?? this.fan,
      growLights: growLights ?? this.growLights,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'water_pump': waterPump,
      'pest_control': pestControl,
      'heater': heater,
      'cooler': cooler,
      'fan': fan,
      'grow_lights': growLights,
    };
  }
}
