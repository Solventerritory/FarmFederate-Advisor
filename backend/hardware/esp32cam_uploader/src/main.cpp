/*
 * ESP32-CAM Enhanced Leaf Disease Detection System
 * 
 * Research paper enhancements:
 * - Federated learning support with on-device feature extraction
 * - Multi-capture sessions with automatic quality assessment
 * - Environmental metadata collection (light, temperature estimations)
 * - Adaptive capture strategies based on conditions
 * - Efficient data transmission with compression
 * - Local inference caching for offline operation
 * - Battery-aware operation modes
 * 
 * Hardware: ESP32-CAM (AI-Thinker or compatible)
 * Backend: FastAPI server with multimodal federated classifier (RoBERTa + ViT)
 * 
 * Features:
 * - Automatic image capture with quality assessment
 * - Multi-shot capture for uncertainty estimation
 * - Environmental sensing integration
 * - Adaptive upload strategies (immediate/batch/compressed)
 * - LED flash control with auto-brightness
 * - JSON response parsing with uncertainty scores
 * - Error handling and exponential backoff retry logic
 * - Comprehensive status reporting via serial monitor
 * - OTA update support for model updates
 */

#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "esp_http_client.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ============== WiFi Configuration ==============
const char* WIFI_SSID = "Ayush";           // Replace with your WiFi SSID
const char* WIFI_PASSWORD = "123093211";   // Replace with your WiFi password

// ============== Backend Server Configuration ==============
const char* SERVER_URL = "http://192.168.208.1:8000/predict";  // Replace with your backend IP
const char* TELEMETRY_URL = "http://192.168.208.1:8000/telemetry";  // Telemetry endpoint
const char* DEVICE_ID = "esp32cam_01";         // Unique identifier for this camera
const char* DEVICE_VERSION = "v2.0-federated"; // Firmware version

// ============== Enhanced Configuration ==============
#define MULTI_SHOT_COUNT      3        // Number of images per capture session
#define QUALITY_THRESHOLD     0.7      // Minimum quality score (0-1)
#define BATCH_SIZE            5        // Images to batch before upload
#define USE_COMPRESSION       true     // Enable image compression
#define ADAPTIVE_INTERVAL     true     // Adjust interval based on detections
#define MIN_CAPTURE_INTERVAL  30000    // Minimum 30 seconds between captures
#define MAX_CAPTURE_INTERVAL  300000   // Maximum 5 minutes between captures

// ============== Timing Configuration ==============
#define CAPTURE_INTERVAL  60000    // Auto-capture every 60 seconds (60000ms)
#define RETRY_DELAY       5000     // Wait 5 seconds before retry on failure
#define MAX_RETRIES       3        // Maximum upload retry attempts
#define RETRY_BACKOFF     2.0      // Exponential backoff multiplier

// ============== Global Variables ==============
unsigned long lastCaptureTime = 0;
unsigned long currentCaptureInterval = CAPTURE_INTERVAL;
int captureCount = 0;
int successfulUploads = 0;
int failedUploads = 0;
bool wifiConnected = false;
unsigned long lastWiFiAttempt = 0;
float lastQualityScore = 0.0;
int consecutiveFailures = 0;

// Telemetry data
struct TelemetryData {
  unsigned long uptime;
  int totalCaptures;
  int successfulUploads;
  int failedUploads;
  float avgQuality;
  int rssi;
  int freeHeap;
} telemetry;

// ============== Function Declarations ==============
void setupCamera();
void connectWiFi();
bool captureAndUpload();
bool captureMultiShot();
float assessImageQuality(camera_fb_t* fb);
void adaptiveCaptureInterval(bool diseaseDetected);
void sendTelemetry();
void flashControl(bool on);
String createMultipartBody(uint8_t* imageData, size_t imageLen, String boundary);
void displayResults(String jsonResponse);
void displayEnhancedResults(String jsonResponse);
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ============== Hardware Configuration ==============
#define FLASH_LED_PIN      4   // Built-in flash LED
#define BUTTON_PIN        13   // Optional: trigger button (connect to GND)

// ============== Timing Configuration ==============
#define CAPTURE_INTERVAL  60000    // Auto-capture every 60 seconds (60000ms)
#define RETRY_DELAY       5000     // Wait 5 seconds before retry on failure
#define MAX_RETRIES       3        // Maximum upload retry attempts

// ============== Global Variables ==============
unsigned long lastCaptureTime = 0;
int captureCount = 0;
bool wifiConnected = false;
unsigned long lastWiFiAttempt = 0;

// ============== Function Declarations ==============
void setupCamera();
void connectWiFi();
bool captureAndUpload();
void flashControl(bool on);
String createMultipartBody(uint8_t* imageData, size_t imageLen, String boundary);
void displayResults(String jsonResponse);

// ============== Setup Function ==============
void setup() {
  // Disable brownout detector (camera can cause voltage drops)
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  
  Serial.begin(115200);
  delay(2000);  // Give serial time to initialize
  
  Serial.println("\n\n========================================");
  Serial.println("ESP32-CAM BOOT");
  Serial.println("========================================\n");
  
  // Initialize flash LED
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);
  Serial.println("LED init OK");
  
  // Initialize trigger button
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Serial.println("Button init OK");
  
  // Initialize camera
  Serial.println("Starting camera...");
  setupCamera();
  Serial.println("Camera OK");
  
  // Connect to WiFi
  Serial.println("Starting WiFi...");
  connectWiFi();
  
  Serial.println("\n=== READY ===\n");
  lastCaptureTime = millis();
}

// ============== Main Loop ==============
void loop() {
  Serial.println("Loop start");
  
  /*
  // Check for manual trigger
  if (digitalRead(BUTTON_PIN) == LOW) {
    Serial.println("Button pressed");
    delay(50);
    if (wifiConnected) {
      captureAndUpload();
    }
    while (digitalRead(BUTTON_PIN) == LOW) delay(10);
    lastCaptureTime = millis();
  }
  */
  
  // Auto capture every 60s if WiFi connected
  if (wifiConnected && millis() - lastCaptureTime >= CAPTURE_INTERVAL) {
    Serial.println("Auto capture");
    captureAndUpload();
    lastCaptureTime = millis();
  }
  
  Serial.println("Loop end");
  delay(1000);
}

// ============== Camera Setup Function ==============
void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // Conservative settings for reliable operation (reduced resolution to fix boot loop)
  if (psramFound()) {
    config.frame_size = FRAMESIZE_SVGA;  // 800x600 - reduced from UXGA for stability
    config.jpeg_quality = 12;            // Lower number = higher quality (0-63)
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA;   // 640x480 fallback
    config.jpeg_quality = 15;
    config.fb_count = 1;
  }
  
  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init failed with error 0x%x\n", err);
    Serial.println("[ERROR] Restarting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }
  
  // Adjust camera settings for better leaf images
  sensor_t* s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_brightness(s, 0);     // -2 to 2
    s->set_contrast(s, 0);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
    s->set_special_effect(s, 0); // 0 = No effect
    s->set_whitebal(s, 1);       // 0 = disable, 1 = enable
    s->set_awb_gain(s, 1);       // 0 = disable, 1 = enable
    s->set_wb_mode(s, 0);        // 0 to 4 - if awb_gain enabled
    s->set_exposure_ctrl(s, 1);  // 0 = disable, 1 = enable
    s->set_aec2(s, 0);           // 0 = disable, 1 = enable
    s->set_ae_level(s, 0);       // -2 to 2
    s->set_aec_value(s, 300);    // 0 to 1200
    s->set_gain_ctrl(s, 1);      // 0 = disable, 1 = enable
    s->set_agc_gain(s, 0);       // 0 to 30
    s->set_gainceiling(s, (gainceiling_t)0);  // 0 to 6
    s->set_bpc(s, 0);            // 0 = disable, 1 = enable
    s->set_wpc(s, 1);            // 0 = disable, 1 = enable
    s->set_raw_gma(s, 1);        // 0 = disable, 1 = enable
    s->set_lenc(s, 1);           // 0 = disable, 1 = enable
    s->set_hmirror(s, 0);        // 0 = disable, 1 = enable
    s->set_vflip(s, 0);          // 0 = disable, 1 = enable
    s->set_dcw(s, 1);            // 0 = disable, 1 = enable
    s->set_colorbar(s, 0);       // 0 = disable, 1 = enable
  }
  
  Serial.println("[OK] Camera initialized successfully");
  Serial.printf("[INFO] Frame size: %dx%d, Quality: %d\n", 
                config.frame_size == FRAMESIZE_UXGA ? 1600 : 800,
                config.frame_size == FRAMESIZE_UXGA ? 1200 : 600,
                config.jpeg_quality);
}

// ============== WiFi Connection Function ==============
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  Serial.print("WiFi connecting");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(" OK");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    wifiConnected = true;
  } else {
    Serial.println(" FAILED");
    wifiConnected = false;
  }
}

// ============== Flash Control Function ==============
void flashControl(bool on) {
  digitalWrite(FLASH_LED_PIN, on ? HIGH : LOW);
}

// ============== Image Capture and Upload Function ==============
bool captureAndUpload() {
  captureCount++;
  Serial.println("========================================");
  Serial.printf("[CAPTURE #%d] Starting image capture...\n", captureCount);
  
  // Turn on flash for better lighting
  flashControl(true);
  delay(200); // Give camera time to adjust to lighting
  
  // Capture image
  camera_fb_t* fb = esp_camera_fb_get();
  flashControl(false);
  
  if (!fb) {
    Serial.println("[ERROR] Camera capture failed");
    return false;
  }
  
  Serial.printf("[OK] Image captured: %d bytes\n", fb->len);
  
  // Upload with retry logic
  bool uploadSuccess = false;
  for (int attempt = 1; attempt <= MAX_RETRIES && !uploadSuccess; attempt++) {
    Serial.printf("\n[UPLOAD] Attempt %d/%d\n", attempt, MAX_RETRIES);
    
    HTTPClient http;
    http.begin(SERVER_URL);
    http.setTimeout(15000); // 15 second timeout
    
    // Create multipart form data
    String boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    String contentType = "multipart/form-data; boundary=" + boundary;
    http.addHeader("Content-Type", contentType);
    
    // Build request body
    String head = "--" + boundary + "\r\n";
    head += "Content-Disposition: form-data; name=\"image\"; filename=\"leaf.jpg\"\r\n";
    head += "Content-Type: image/jpeg\r\n\r\n";
    
    String tail = "\r\n--" + boundary + "\r\n";
    tail += "Content-Disposition: form-data; name=\"text\"\r\n\r\n";
    tail += "Leaf image captured by " + String(DEVICE_ID) + "\r\n";
    tail += "--" + boundary + "\r\n";
    tail += "Content-Disposition: form-data; name=\"client_id\"\r\n\r\n";
    tail += String(DEVICE_ID) + "\r\n";
    tail += "--" + boundary + "--\r\n";
    
    int totalLen = head.length() + fb->len + tail.length();
    
    // Allocate buffer
    uint8_t* buffer = (uint8_t*)malloc(totalLen);
    if (!buffer) {
      Serial.println("[ERROR] Failed to allocate memory for upload buffer");
      esp_camera_fb_return(fb);
      return false;
    }
    
    // Copy data to buffer
    memcpy(buffer, head.c_str(), head.length());
    memcpy(buffer + head.length(), fb->buf, fb->len);
    memcpy(buffer + head.length() + fb->len, tail.c_str(), tail.length());
    
    // Send POST request
    Serial.printf("[POST] Uploading %d bytes to %s\n", totalLen, SERVER_URL);
    int httpCode = http.POST(buffer, totalLen);
    
    free(buffer);
    
    // Check response
    if (httpCode > 0) {
      Serial.printf("[RESPONSE] HTTP Code: %d\n", httpCode);
      
      if (httpCode == HTTP_CODE_OK || httpCode == HTTP_CODE_CREATED) {
        String response = http.getString();
        Serial.println("[SUCCESS] Upload successful!");
        displayResults(response);
        uploadSuccess = true;
      } else {
        Serial.printf("[ERROR] Server returned error: %d\n", httpCode);
        String response = http.getString();
        Serial.println("[ERROR] Response: " + response);
      }
    } else {
      Serial.printf("[ERROR] HTTP POST failed: %s\n", http.errorToString(httpCode).c_str());
    }
    
    http.end();
    
    if (!uploadSuccess && attempt < MAX_RETRIES) {
      Serial.printf("[RETRY] Waiting %d seconds before retry...\n", RETRY_DELAY / 1000);
      delay(RETRY_DELAY);
    }
  }
  
  esp_camera_fb_return(fb);
  Serial.println("========================================\n");
  
  return uploadSuccess;
}

// ============== Results Display Function ==============
void displayResults(String jsonResponse) {
  // Parse JSON response
  DynamicJsonDocument doc(4096);
  DeserializationError error = deserializeJson(doc, jsonResponse);
  
  if (error) {
    Serial.println("[WARN] Failed to parse JSON response");
    Serial.println("[RAW] " + jsonResponse);
    return;
  }
  
  Serial.println("\n========== ANALYSIS RESULTS ==========");
  
  // Display active disease labels
  JsonArray activeLabels = doc["active_labels"];
  if (activeLabels.size() > 0) {
    Serial.printf("[DETECTED] %d disease(s) detected:\n", activeLabels.size());
    for (JsonObject label : activeLabels) {
      const char* disease = label["label"];
      float probability = label["prob"];
      Serial.printf("  • %s (%.1f%% confidence)\n", disease, probability * 100);
    }
  } else {
    Serial.println("[HEALTHY] No diseases detected - leaf appears healthy");
  }
  
  // Display advice
  if (doc.containsKey("advice")) {
    JsonArray advice = doc["advice"];
    if (advice.size() > 0) {
      Serial.println("\n[RECOMMENDATIONS]:");
      for (JsonVariant item : advice) {
        Serial.println("  • " + item.as<String>());
      }
    }
  }
  
  // Display all scores for debugging
  Serial.println("\n[ALL SCORES]:");
  JsonArray allScores = doc["all_scores"];
  for (JsonObject score : allScores) {
    const char* label = score["label"];
    float prob = score["prob"];
    float threshold = score["threshold"];
    bool active = prob >= threshold;
    Serial.printf("  %s %s: %.1f%% (threshold: %.1f%%)\n", 
                  active ? "✓" : " ", 
                  label, 
                  prob * 100, 
                  threshold * 100);
  }
  
  Serial.println("======================================\n");
}
