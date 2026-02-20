#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>

// === Servo Setup ===
Servo panServo;
Servo tiltServo;
const int PAN_SERVO_PIN = 19;
const int TILT_SERVO_PIN = 18;

// === Motor Pins ===
const int MOTOR1_A = 12;
const int MOTOR1_B = 13;
const int MOTOR2_A = 14;
const int MOTOR2_B = 27;

// === WiFi Credentials ===
const char* ssid = "sim";
const char* password = "simple12";

// === Web Server ===
WebServer server(80);

// === Firing Control ===
bool motorsRunning = false;
bool continuousMode = false;
bool isFiring = false;
bool triggerPressed = false;
unsigned long motorStartTime = 0;
unsigned long lastFireTime = 0;
unsigned long triggerPressStartTime = 0;
const unsigned long MOTOR_RUN_TIME = 2000;
const unsigned long FIRE_INTERVAL = 500;
const unsigned long TRIGGER_PRESS_TIME = 200;

// === Optimized Web Page (Stored in Flash) ===
const char MAIN_page[] PROGMEM = R"====(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARYABHATA | Tactical Turret Control</title>
    <!-- Modern Font -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --accent: #3b82f6;
            --danger: #ef4444;
            --success: #10b981;
            --text: #f8fafc;
            --text-dim: #94a3b8;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .dashboard {
            width: 100%;
            max-width: 500px;
            background: var(--card);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255,255,255,0.05);
        }

        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-weight: 800; letter-spacing: -1px; font-size: 1.5rem; display: flex; align-items: center; justify-content: center; gap: 10px; }
        .header p { color: var(--text-dim); font-size: 0.9rem; margin-top: 5px; }

        .control-group { background: rgba(0,0,0,0.2); padding: 20px; border-radius: 16px; margin-bottom: 20px; }
        .label-row { display: flex; justify-content: space-between; margin-bottom: 15px; color: var(--text-dim); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
        .angle-value { color: var(--accent); font-family: monospace; font-size: 1.1rem; }

        /* Range Slider Styling */
        input[type=range] {
            width: 100%;
            height: 6px;
            background: #334155;
            border-radius: 5px;
            appearance: none;
            outline: none;
            margin-bottom: 10px;
        }
        input[type=range]::-webkit-slider-thumb {
            appearance: none;
            width: 24px;
            height: 24px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
            border: 3px solid var(--text);
        }

        /* Button Matrix */
        .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 25px; }
        
        button {
            border: none;
            border-radius: 12px;
            padding: 15px;
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            color: white;
            text-transform: uppercase;
        }

        button:active { transform: scale(0.96); }

        .btn-fire { background: var(--danger); grid-column: span 2; padding: 20px; font-size: 1.1rem; box-shadow: 0 4px 0 #b91c1c; }
        .btn-fire:active { box-shadow: 0 0 0 transparent; }

        .btn-continuous { background: #334155; border: 1px solid #475569; }
        .btn-continuous.active { background: var(--success); box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); border-color: #34d399; }
        
        .btn-stop { background: #475569; }

        .status-footer { 
            margin-top: 25px; 
            padding-top: 20px; 
            border-top: 1px solid rgba(255,255,255,0.05); 
            display: flex; 
            justify-content: center; 
            gap: 15px; 
            font-size: 0.8rem; 
            color: var(--text-dim);
        }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); display: inline-block; box-shadow: 0 0 8px var(--success); }
    </style>
</head>
<body>

<div class="dashboard">
    <div class="header">
        <h1><i data-lucide="crosshair"></i>SPY CAR NERF TURRET</h1>
        <p>Tactical Servo Response Interface</p>
    </div>

    <div class="control-group">
        <div class="label-row">
            <span>Pan Axis (L/R)</span>
            <span class="angle-value"><span id="panVal">90</span>°</span>
        </div>
        <input type="range" min="0" max="180" value="90" oninput="updateVal(this.value, 'pan')" onchange="send(this.value, 'pan')">
        
        <div class="label-row" style="margin-top: 20px;">
            <span>Tilt Axis (U/D)</span>
            <span class="angle-value"><span id="tiltVal">90</span>°</span>
        </div>
        <input type="range" min="0" max="180" value="90" oninput="updateVal(this.value, 'tilt')" onchange="send(this.value, 'tilt')">
    </div>

    <button class="btn-fire" onclick="fire()">
        <i data-lucide="zap"></i> Single Fire
    </button>

    <div class="actions">
        <button class="btn-continuous" onclick="toggleContinuous()" id="continuousBtn">
            <i data-lucide="refresh-cw"></i> Burst Mode
        </button>
        <button class="btn-stop" onclick="stopFiring()">
            <i data-lucide="octagon"></i> Panic Stop
        </button>
    </div>

    <div class="status-footer">
        <span><span class="status-dot"></span> System Ready</span>
        <span>ESP32-S3 Core</span>
    </div>
</div>

<script>
    // Initialize Icons
    lucide.createIcons();

    let continuousMode = false;
    let sendTimeout;

    // Visual updates while sliding
    function updateVal(val, axis) {
        document.getElementById(axis + 'Val').textContent = val;
    }

    // Server communication
    function send(val, axis) {
        clearTimeout(sendTimeout);
        sendTimeout = setTimeout(() => {
            fetch(`/aim?axis=${axis}&value=${val}`)
                .then(r => console.log(`Aim ${axis}: ${val}`))
                .catch(e => console.error("Aim failed"));
        }, 50); // Faster response than 100ms
    }

    function fire() {
        fetch('/fire').then(r => console.log("Fire signal sent"));
    }

    function toggleContinuous() {
        const btn = document.getElementById('continuousBtn');
        if (!continuousMode) {
            continuousMode = true;
            fetch('/continuous?mode=start');
            btn.classList.add('active');
            btn.innerHTML = '<i data-lucide="pause-circle"></i> STOP BURST';
        } else {
            continuousMode = false;
            fetch('/continuous?mode=stop');
            btn.classList.remove('active');
            btn.innerHTML = '<i data-lucide="refresh-cw"></i> Burst Mode';
        }
        lucide.createIcons();
    }

    function stopFiring() {
        continuousMode = false;
        fetch('/stop');
        const btn = document.getElementById('continuousBtn');
        btn.classList.remove('active');
        btn.innerHTML = '<i data-lucide="refresh-cw"></i> Burst Mode';
        lucide.createIcons();
    }
</script>

</body>
</html>
)====";

void setup() {
  Serial.begin(115200);
  panServo.attach(PAN_SERVO_PIN);
  tiltServo.attach(TILT_SERVO_PIN);
  panServo.write(90);
  tiltServo.write(90);

  pinMode(MOTOR1_A, OUTPUT);
  pinMode(MOTOR1_B, OUTPUT);
  pinMode(MOTOR2_A, OUTPUT);
  pinMode(MOTOR2_B, OUTPUT);
  stopMotors();

  WiFi.softAP(ssid, password);
  Serial.println("Access Point started");
  Serial.println(WiFi.softAPIP());

  server.on("/", []() { server.send_P(200, "text/html", MAIN_page); });
  server.on("/fire", handleFire);
  server.on("/continuous", handleContinuous);
  server.on("/stop", handleStop);
  server.on("/aim", handleAim);
  server.begin();
}

void loop() {
  server.handleClient();
  updateTrigger();
  updateMotorTimeout();
  updateContinuousFire();
}

// === Web Handlers ===

void handleFire() {
  runMotors();
  motorsRunning = true;
  motorStartTime = millis();
  fireSingleShot();
  server.send(200, "text/plain", "Fired!");
}

void handleContinuous() {
  if (server.hasArg("mode")) {
    String mode = server.arg("mode");
    continuousMode = (mode == "start");
    isFiring = continuousMode;

    if (continuousMode) {
      runMotors();
      motorsRunning = true;
      motorStartTime = millis();
      lastFireTime = millis();
    } else {
      stopMotors();
      motorsRunning = false;
    }
    server.send(200, "text/plain", continuousMode ? "Continuous ON" : "Continuous OFF");
  } else {
    server.send(400, "text/plain", "Missing mode");
  }
}

void handleStop() {
  continuousMode = false;
  isFiring = false;
  triggerPressed = false;
  stopMotors();
  motorsRunning = false;
  tiltServo.write(90);
  server.send(200, "text/plain", "STOPPED");
}

void handleAim() {
  if (server.hasArg("axis") && server.hasArg("value")) {
    int val = constrain(server.arg("value").toInt(), 0, 180);
    if (server.arg("axis") == "pan") panServo.write(val);
    else tiltServo.write(val);
    server.send(200, "text/plain", "Moved");
  } else {
    server.send(400, "text/plain", "Missing args");
  }
}

// === Utility Functions ===

void fireSingleShot() {
  tiltServo.write(0);
  triggerPressed = true;
  triggerPressStartTime = millis();
  Serial.println("Dart fired");
}

void updateTrigger() {
  if (triggerPressed && millis() - triggerPressStartTime > TRIGGER_PRESS_TIME) {
    tiltServo.write(90);
    triggerPressed = false;
  }
}

void updateMotorTimeout() {
  if (!continuousMode && motorsRunning && millis() - motorStartTime > MOTOR_RUN_TIME) {
    stopMotors();
    motorsRunning = false;
  }
}

void updateContinuousFire() {
  if (continuousMode && isFiring && millis() - lastFireTime > FIRE_INTERVAL) {
    fireSingleShot();
    lastFireTime = millis();
  }
}

void runMotors() {
  digitalWrite(MOTOR1_A, HIGH);
  digitalWrite(MOTOR1_B, LOW);
  digitalWrite(MOTOR2_A, HIGH);
  digitalWrite(MOTOR2_B, LOW);
  Serial.println("Motors spinning");
}

void stopMotors() {
  digitalWrite(MOTOR1_A, LOW);
  digitalWrite(MOTOR1_B, LOW);
  digitalWrite(MOTOR2_A, LOW);
  digitalWrite(MOTOR2_B, LOW);
  Serial.println("Motors stopped");
}
