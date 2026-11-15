#include <Arduino_LSM9DS1.h>

// Fixed 50 Hz schedule. Wire units are g and degrees/second.
constexpr unsigned long SAMPLE_PERIOD_US = 20000;
unsigned long nextSampleUs = 0;
unsigned long lastSampleMs = 0;
unsigned long lastStatusMs = 0;
bool imuReady = false;

void setup() {
  Serial.begin(115200);
  imuReady = IMU.begin();
  if (imuReady) {
    Serial.println("HEADER,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
  }
  nextSampleUs = micros();
}

void loop() {
  if (!imuReady) {
    // Repeat the error so a serial connection opened later still sees it.
    Serial.println("ERROR,IMU_INIT_FAILED");
    delay(1000);
    return;
  }
  unsigned long now = micros();
  if ((long)(now - nextSampleUs) < 0) return;
  nextSampleUs += SAMPLE_PERIOD_US;
  if ((long)(now - nextSampleUs) >= 0) nextSampleUs = now + SAMPLE_PERIOD_US;

  // Both sensors must be ready. Skip incomplete pairs rather than send stale data.
  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {
    unsigned long currentMs = millis();
    if (currentMs - lastSampleMs >= 3000 && currentMs - lastStatusMs >= 3000) {
      Serial.println("STATUS,WAITING_FOR_IMU");
      lastStatusMs = currentMs;
    }
    return;
  }
  float ax, ay, az, gx, gy, gz;
  if (!IMU.readAcceleration(ax, ay, az)) return;
  if (!IMU.readGyroscope(gx, gy, gz)) return;
  lastSampleMs = millis();

  Serial.print("IMU,"); Serial.print(millis());
  Serial.print(','); Serial.print(ax, 6);
  Serial.print(','); Serial.print(ay, 6);
  Serial.print(','); Serial.print(az, 6);
  Serial.print(','); Serial.print(gx, 6);
  Serial.print(','); Serial.print(gy, 6);
  Serial.print(','); Serial.print(gz, 6);
  Serial.println();
}
