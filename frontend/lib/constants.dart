const String DEFAULT_BACKEND = "http://localhost:8000"; // For web/desktop, use localhost
// For Android emulator use: http://10.0.2.2:8000
// On real device use actual LAN IP: e.g. http://192.168.1.20:8000
const String PREDICT_PATH = "/predict";

// MQTT broker (configure for your mosquitto)
const String MQTT_BROKER = "ws://192.168.1.100:9001"; // WebSocket endpoint
const String MQTT_USERNAME = "";
const String MQTT_PASSWORD = "";
const String SENSOR_TOPIC = "farm/sensors/#";
const String CMD_TOPIC = "farm/cmd/";
