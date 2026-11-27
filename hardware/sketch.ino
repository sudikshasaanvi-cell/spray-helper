// ESP32-CAM simple spray controller
#define RELAY 4
#define SOIL 34

void setup() {
  Serial.begin(115200);
  pinMode(RELAY, OUTPUT);
  digitalWrite(RELAY, HIGH); // HIGH = off (active-low relay)
}

void spray(int ms) {
  digitalWrite(RELAY, LOW);
  delay(ms);
  digitalWrite(RELAY, HIGH);
}

void loop() {
  int soil = analogRead(SOIL);
  // Backend (or serial) sends "SPRAY" to trigger
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "SPRAY") {
      spray(2000);
      Serial.println("SPRAYED");
    }
  }
  delay(100);
}
