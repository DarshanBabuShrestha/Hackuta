#include <Arduino_LSM9DS1.h>

// Fixed 50 Hz schedule. Wire units are g and degrees/second.
constexpr unsigned long SAMPLE_PERIOD_US = 20000;
unsigned long nextSampleUs = 0;

void setup() {
  Serial.begin(115200);
  if (!IMU.begin()) {
    Serial.println("ERROR,IMU_INIT_FAILED");
    while (true) { delay(1000); }
  }
  Serial.println("HEADER,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
  nextSampleUs = micros();
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - nextSampleUs) < 0) return;
  nextSampleUs += SAMPLE_PERIOD_US;
  if ((long)(now - nextSampleUs) >= 0) nextSampleUs = now + SAMPLE_PERIOD_US;

  // Both sensors must be ready. Skip incomplete pairs rather than send stale data.
  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) return;
  float ax, ay, az, gx, gy, gz;
  if (!IMU.readAcceleration(ax, ay, az)) return;
  if (!IMU.readGyroscope(gx, gy, gz)) return;

  Serial.print("IMU,"); Serial.print(millis());
  Serial.print(','); Serial.print(ax, 6);
  Serial.print(','); Serial.print(ay, 6);
  Serial.print(','); Serial.print(az, 6);
  Serial.print(','); Serial.print(gx, 6);
  Serial.print(','); Serial.print(gy, 6);
  Serial.print(','); Serial.print(gz, 6);
  Serial.println();
}
