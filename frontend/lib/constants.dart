const String DEFAULT_BACKEND = "http://localhost:8001"; // For web/desktop, use localhost
// For Android emulator use: http://10.0.2.2:8001
// On real device use actual LAN IP: e.g. http://192.168.1.20:8001
const String PREDICT_PATH = "/predict";

// Qdrant-powered endpoints
const String RAG_PATH = "/rag";
const String DEMO_POPULATE_PATH = "/demo_populate";
const String DEMO_SEARCH_PATH = "/demo_search";
const String HEALTH_PATH = "/health";

// Qdrant collections
const String KNOWLEDGE_COLLECTION = "crop_health_knowledge";
const String SESSION_COLLECTION = "farm_session_memory";

// MQTT broker (configure for your mosquitto)
const String MQTT_BROKER = "ws://192.168.1.100:9001"; // WebSocket endpoint
const String MQTT_USERNAME = "";
const String MQTT_PASSWORD = "";
const String SENSOR_TOPIC = "farm/sensors/#";
const String CMD_TOPIC = "farm/cmd/";
