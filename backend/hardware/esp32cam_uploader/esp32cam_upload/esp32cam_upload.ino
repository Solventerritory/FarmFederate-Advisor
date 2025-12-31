// esp32cam_upload.ino (conceptual)
// This sketch captures image and posts to backend /predict as multipart form.
// Using HTTPClient and the camera library.

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* ssid = "YOUR_SSID";
const char* pass = "YOUR_PASS";

const char* server = "http://192.168.1.100:8000/predict"; // backend

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  // initialize camera - use your specific pin config
  camera_config_t config;
  // ... set pins for your board ...
  esp_camera_init(&config);
}

void uploadImage() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) { return; }

  HTTPClient http;
  http.begin(server);
  http.addHeader("Content-Type", "image/jpeg"); // some servers accept raw body as image too
  int status = http.POST(fb->buf, fb->len);
  String resp = http.getString();
  esp_camera_fb_return(fb);
  http.end();
}

void loop() {
  // call uploadImage upon button press or interval
  uploadImage();
  delay(60*1000);
}
