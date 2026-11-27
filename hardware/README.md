# Hardware (mini project)

Simple, low-cost setup for the Smart Spray Planner.

## Parts
- ESP32-CAM (camera + Wi-Fi)
- 1-channel 5V relay module
- Small 5V DC water pump + tube
- Soil moisture sensor (optional, for IoT dashboard)
- DHT11 (temp/humidity, optional)
- 18650 battery + holder (or USB power bank)
- Jumper wires, breadboard

## Wiring
- ESP32-CAM GPIO 4 -> relay IN (active-low)
- Relay COM/NO -> pump + power
- Soil sensor -> ESP32 ADC pin (e.g. GPIO 34)
- DHT11 data -> GPIO 13
- Power: 5V to ESP32 5V, GND common

## Flow
1. ESP32-CAM captures a photo, POSTs it to the backend.
2. Backend runs the AI model; if infected and confidence >= threshold, replies "SPRAY".
3. ESP32 toggles the relay -> pump sprays for ~2 s.
4. Sensor readings feed the IoT dashboard.

## Safety (demo only)
- Use water, not real pesticide.
- Keep the pump in a tray.
- Activate spray only when backend + hardware are connected.

## Sketch
See hardware/sketch.ino
