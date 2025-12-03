// esp32cam_upload.ino
// Capture image from camera and POST multipart/form-data to backend /api/image_upload
#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// === CONFIG ===
const char* WIFI_SSID   = "YOUR_SSID";
const char* WIFI_PASS   = "YOUR_PASS";
const char* BACKEND_URL = "http://192.168.1.10:8000/api/image_upload";
const char* DEVICE_ID   = "esp32cam-01";
const uint32_t UPLOAD_INTERVAL_MS = 60*1000; // 1 minute
// =============

// If you have AI-Thinker module use this setup (common)
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

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis()-start < 20000) {
    Serial.print('.');
    delay(500);
  }
  Serial.println();
}

void init_camera() {
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
  config.frame_size = FRAMESIZE_SVGA; // change to FRAMESIZE_VGA or smaller if memory constrained
  config.jpeg_quality = 10;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    while (true) { delay(1000); }
  } else {
    Serial.println("Camera init OK");
  }
}

bool http_post_image(const uint8_t* buf, size_t len, const char* filename) {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  String boundary = "----FarmFedBoundary";
  String contentType = "multipart/form-data; boundary=" + boundary;
  String url = String(BACKEND_URL); // e.g. http://host:8000/api/image_upload

  http.begin(url);
  http.addHeader("Content-Type", contentType);
  // build multipart body in streaming manner:
  String head = "--" + boundary + "\r\n";
  head += "Content-Disposition: form-data; name=\"device_id\"\r\n\r\n";
  head += DEVICE_ID;
  head += "\r\n--" + boundary + "\r\n";
  head += "Content-Disposition: form-data; name=\"image\"; filename=\"" + String(filename) + "\"\r\n";
  head += "Content-Type: image/jpeg\r\n\r\n";

  // Send head
  http.sendRequest("POST", (uint8_t*)NULL, 0); // prepare
  WiFiClient* stream = http.getStreamPtr();
  stream->print(head);
  // send binary
  stream->write(buf, len);
  // end
  stream->print("\r\n--" + boundary + "--\r\n");

  int code = http.GET(); // Note: HTTPClient on ESP32 oddities — sometimes use http.sendRequest directly.
  Serial.printf("Upload returned code %d\n", code);
  http.end();
  return (code >= 200 && code < 300);
}

void setup() {
  Serial.begin(115200);
  delay(100);
  connectWiFi();
  init_camera();
}

void loop() {
  connectWiFi();

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(2000);
    return;
  }

  Serial.printf("Captured size=%u\n", fb->len);

  bool ok = http_post_image(fb->buf, fb->len, "img.jpg");
  if (ok) Serial.println("Image uploaded");
  else Serial.println("Image upload failed");

  esp_camera_fb_return(fb);
  delay(UPLOAD_INTERVAL_MS);
}
