# controllers/mavic2pro/mavic2pro.py
# Webots R2025a – Python
#
# Mavic2 Pro "sensor-only" controller:
# - NO flight physics, NO motors.
# - Reads downward camera recognition and reports victim identities to Supervisor.
# - Supervisor moves the drone kinematically (translation/rotation fields).
#
# Protocol (channel 1):
# - Receives: {"type":"MISSION","mode":"A"} to start aerial scouting (mode "G" keeps it idle)
# - Receives: {"type":"AERIAL_FREEZE"} to stop sending detections (after operator presses E)
# - Sends:    {"type":"VICTIM_NAME","robot":"mavic2pro","victim":"VICTIM_1"}

from controller import Robot
import json
import math

SCOUT_CHANNEL = 1
VICTIM_PREFIX = "VICTIM_"
CHECK_EVERY_STEPS = 2
PER_VICTIM_COOLDOWN_S = 1.0
GLOBAL_COOLDOWN_S = 0.20

def recog_name(obj):
    for fn in ("getModel", "getDef", "getName"):
        try:
            v = getattr(obj, fn)()
            if v:
                return str(v)
        except Exception:
            pass
    return ""

robot = Robot()
timestep = int(robot.getBasicTimeStep())
name = robot.getName()

# Optional: spin propellers for a more "alive" feel.
# This is purely visual on most Mavic models (rotors are separate joints).
# If your drone prototype uses different device names, extend this list.
PROP_MOTOR_NAMES = [
    "front left propeller",
    "front right propeller",
    "rear left propeller",
    "rear right propeller",
]
PROP_RADS_PER_SEC = 60.0

prop_motors = []
for mn in PROP_MOTOR_NAMES:
    try:
        m = robot.getDevice(mn)
        # Continuous rotation
        try:
            m.setPosition(float('inf'))
        except Exception:
            pass
        prop_motors.append(m)
    except Exception:
        pass

camera = robot.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)

receiver = robot.getDevice("receiver")
receiver.enable(timestep)
try:
    receiver.setChannel(SCOUT_CHANNEL)
except Exception:
    pass

emitter = robot.getDevice("emitter")
try:
    emitter.setChannel(SCOUT_CHANNEL)
except Exception:
    pass

mission_mode = None   # None until supervisor broadcasts MISSION
active = False
frozen = False

last_sent_global = -1e9
last_sent_victim = {}  # victimName -> time

step = 0
print(f"[{name}] mavic2pro sensor controller started (waiting for MISSION)")

while robot.step(timestep) != -1:
    t = robot.getTime()
    step += 1

    # Receive commands
    while receiver.getQueueLength() > 0:
        raw = receiver.getString()
        receiver.nextPacket()
        try:
            data = json.loads(raw)
        except Exception:
            continue

        if data.get("type") == "MISSION":
            mm = str(data.get("mode", "")).upper()
            if mm in ("A", "G"):
                mission_mode = mm
                active = (mission_mode == "A")
                frozen = False
                print(f"[{name}] mission mode set to {mission_mode}")
        elif data.get("type") == "AERIAL_FREEZE":
            frozen = True
            active = False
            print(f"[{name}] frozen (aerial scouting ended)")

    # Visual rotor spin: on in aerial mode, off otherwise.
    v = PROP_RADS_PER_SEC if (active and not frozen) else 0.0
    for m in prop_motors:
        try:
            m.setVelocity(v)
        except Exception:
            pass

    if not active or frozen:
        continue

    if step % CHECK_EVERY_STEPS != 0:
        continue

    if (t - last_sent_global) < GLOBAL_COOLDOWN_S:
        continue

    objs = camera.getRecognitionObjects()
    if not objs:
        continue

    for obj in objs:
        model = recog_name(obj)
        if not model.startswith(VICTIM_PREFIX):
            continue

        last_t = last_sent_victim.get(model, -1e9)
        if (t - last_t) < PER_VICTIM_COOLDOWN_S:
            continue

        msg = {"type": "VICTIM_NAME", "robot": name, "victim": model, "t": float(t)}
        try:
            emitter.send(json.dumps(msg))
        except Exception:
            try:
                emitter.send(json.dumps(msg).encode("utf-8"))
            except Exception:
                pass

        last_sent_victim[model] = t
        last_sent_global = t
        print(f"[{name}] ✅ VICTIM detected: {model} (sent)")
        break
