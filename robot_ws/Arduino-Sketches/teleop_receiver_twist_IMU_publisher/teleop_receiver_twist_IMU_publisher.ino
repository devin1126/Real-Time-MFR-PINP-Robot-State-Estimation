#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

// ---------- Encoder Setup ----------
const byte encoder0pinA = 2;
const byte encoder0pinB = 3;
const byte encoder1pinA = 19;
const byte encoder1pinB = 18;

volatile long duration0 = 0;
volatile long duration1 = 0;
volatile bool Direction0 = true;
volatile bool Direction1 = true;
byte encoder0PinALast;
byte encoder1PinALast;

// ---------- Motor Pins ----------
#define IN1 A2
#define IN2 A3
#define IN3 A4
#define IN4 A5
#define ENA 7
#define ENB 5
#define carSpeed 255

// ---------- Robot Geometry ----------
float radian = 2 * 3.141592654;
float rad_wheel = 0.0324;
float chassis_width = 0.2619;
int ppr = 1900;
float vel_data[2];

// ---------- Timing ----------
unsigned long lastVelPublishTime = 0;
unsigned long lastIMUPublishTime = 0;
unsigned long lastCmdCheckTime = 0;
const unsigned long velPublishInterval = 20;   // 50 Hz
const unsigned long IMUPublishInterval = 20;   // 50 Hz
const unsigned long cmdCheckInterval = 10;     // 100 Hz

// ---------- IMU Setup ----------
Adafruit_BNO055 bno = Adafruit_BNO055(55);
const float DEG2RAD = 3.141592654 / 180.0;

// ---------- ROS Handshake ----------
bool rosConnected = false;

// ---------- Setup ----------
void setup() {
  Serial.begin(57600);
  delay(1000);

  // Initialize Encoders
  EncoderInit();

  // Initialize Motor Pins
  pinMode(IN1, OUTPUT); 
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Initialize IMU
  if (!bno.begin(OPERATION_MODE_NDOF)) {
    Serial.println("ERROR: No BNO055 detected");
    while (1);
  }
  bno.setExtCrystalUse(true);

  // Optional calibration offsets
  adafruit_bno055_offsets_t offsets;
  offsets.gyro_offset_x = 1;
  offsets.gyro_offset_y = -2;
  offsets.gyro_offset_z = 1;
  offsets.accel_offset_x = -51;
  offsets.accel_offset_y = 8;
  offsets.accel_offset_z = -3;
  offsets.accel_radius = 1000;
  offsets.mag_offset_x = 19;
  offsets.mag_offset_y = 288;
  offsets.mag_offset_z = 96;
  offsets.mag_radius = 697;
  bno.setSensorOffsets(offsets);

  Serial.println("WAITING_FOR_ROS");
}

// ---------- Main Loop ----------
void loop() {
  unsigned long currentTime = millis();

  // ---- 1. Check for Commands / Handshake ----
  if (currentTime - lastCmdCheckTime >= cmdCheckInterval) {
    lastCmdCheckTime = currentTime;
    if (Serial.available() > 0) {
      String msg = Serial.readStringUntil('\n');
      msg.trim();

      if (msg == "rosConnected") {
        rosConnected = true;
        Serial.println("ROS_ACK");
      } else if (msg == "rosDisconnected") {
        rosConnected = false;
        Serial.println("ROS_STOP");
      } else {
        teleop_cmd_receive(msg);
      }
    }
  }

  // ---- 2. Only Publish If Connected ----
  if (!rosConnected)
    return;
  
  // ---- 3. Publish Wheel Velocity ----
  if (currentTime - lastVelPublishTime >= velPublishInterval) {
    lastVelPublishTime = currentTime;

    noInterrupts();
    long d0 = duration0;
    long d1 = duration1;
    duration0 = 0;
    duration1 = 0;
    interrupts();

    float lw_rps = ((float)d0 * (1000/velPublishInterval) / ppr) * radian;
    float rw_rps = ((float)d1 * (1000/velPublishInterval) / ppr) * radian;

    vel_data[0] = (rad_wheel * rw_rps + rad_wheel * lw_rps) / 2.0;
    vel_data[1] = (rad_wheel * rw_rps - rad_wheel * lw_rps) / chassis_width;
    
    Serial.print("VEL,");
    Serial.print(vel_data[0], 6); Serial.print(",");
    Serial.print(vel_data[1], 6); Serial.print(",");
    Serial.print(lw_rps, 6); Serial.print(",");
    Serial.println(rw_rps, 6);
  }

  // ---- 4. Publish IMU Data ----
  if (currentTime - lastIMUPublishTime >= IMUPublishInterval) {
    lastIMUPublishTime = currentTime;

    imu::Quaternion quat = bno.getQuat();
    imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);

    Serial.print("IMU,");
    Serial.print(quat.x(), 6); Serial.print(",");
    Serial.print(quat.y(), 6); Serial.print(",");
    Serial.print(quat.z(), 6); Serial.print(",");
    Serial.print(quat.w(), 6); Serial.print(",");
    Serial.print(gyro.z() * DEG2RAD, 6); Serial.print(",");
    Serial.println(-accel.y(), 6);
  }
}

// ---------- Encoder Initialization ----------
void EncoderInit() {
  Direction0 = true;
  Direction1 = true;
  pinMode(encoder0pinB, INPUT);
  pinMode(encoder1pinB, INPUT);
  attachInterrupt(digitalPinToInterrupt(encoder0pinA), wheelSpeed0, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoder1pinA), wheelSpeed1, CHANGE);
}

// ---------- Encoder ISRs ----------
void wheelSpeed0() {
  int Lstate = digitalRead(encoder0pinA);
  if ((encoder0PinALast == LOW) && (Lstate == HIGH)) {
    int val = digitalRead(encoder0pinB);
    Direction0 = (val == HIGH);
  }
  encoder0PinALast = Lstate;
  if (Direction0) duration0++;
  else duration0--;
}

void wheelSpeed1() {
  int Lstate = digitalRead(encoder1pinA);
  if ((encoder1PinALast == LOW) && (Lstate == HIGH)) {
    int val = digitalRead(encoder1pinB);
    Direction1 = (val == HIGH);
  }
  encoder1PinALast = Lstate;
  if (Direction1) duration1++;
  else duration1--;
}

// ---------- Teleop Handling ----------
void teleop_cmd_receive(String vel_msg) {
  if (!rosConnected) return;
  vel_msg.trim();
  if (vel_msg == "forward") forward();
  else if (vel_msg == "backward") backward();
  else if (vel_msg == "right") right();
  else if (vel_msg == "left") left();
  else STOP();
}

// ---------- Motor Commands ----------
void forward() {
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void backward() {
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void right() {
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW); 
}

void left() {
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void STOP() {
  digitalWrite(ENA, LOW);
  digitalWrite(ENB, LOW);
}
