# controllers/khepera3_rescue_controller/khepera3_rescue_controller.py
# Webots R2025a – Python
#
# IMPORTANT AXIS NOTE (your world):
# - Ground plane is X–Y
# - Height is Z
#
# Khepera3 rescue (operator-in-the-loop):
# - Listens on channel 2 for RESCUE_START from supervisor.
# - After RESCUE_START, it STOPS and waits for supervisor commands:
#     * CMD_WAIT
#     * CMD_GO_VICTIM (victim index + target xy)
#     * CMD_GO_HOME   (home xy)
# - Uses A* on the supervisor occupancy grid for shortest path.
# - Sends back to supervisor on channel 2:
#     * RESCUER_HOME (once)
#     * POSE (periodic)
#     * ARRIVED_VICTIM / ARRIVED_HOME
#
# NOTE: The supervisor is the ONLY place where the operator types commands.

from controller import Robot
import math
import json
import heapq

# -------------------------
# Motion params (tune if needed)
# -------------------------
WHEEL_RADIUS = 0.021
AXLE_LENGTH = 0.088
MOTOR_MAX_RAD_S = 10.0

V_FWD = 0.22
W_GAIN = 2.6
SLOW_ANGLE = 0.9
WAYPOINT_REACH_M = 0.10
GOAL_REACH_M = 0.14

WAYPOINT_STRIDE = 1  # patched: follow every grid cell to avoid corner-cutting
REPLAN_EVERY_STEPS = 25

POSE_SEND_EVERY_STEPS = 10

# -------------------------
# Helpers
# -------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def heading_from_compass_xy(north):
    # For X–Y ground plane (Z up): use x,y components
    return math.atan2(north[0], north[1])

def world_to_grid(x, y, origin_x, origin_y, res, w, h):
    gx = int((x - origin_x) / res)
    gy = int((y - origin_y) / res)
    gx = clamp(gx, 0, w - 1)
    gy = clamp(gy, 0, h - 1)
    return gx, gy

def grid_to_world(gx, gy, origin_x, origin_y, res):
    x = origin_x + (gx + 0.5) * res
    y = origin_y + (gy + 0.5) * res
    return x, y

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def astar(occ, w, h, start, goal):
    """A* on 8-neighborhood grid with no corner cutting."""
    sx, sy = start
    gx, gy = goal
    if occ is None:
        return []
    if occ[sy][sx] or occ[gy][gx]:
        return []

    gscore = [[math.inf] * w for _ in range(h)]
    parent = [[None] * w for _ in range(h)]
    pq = []
    gscore[sy][sx] = 0.0
    heapq.heappush(pq, (heuristic(start, goal), 0.0, sx, sy))

    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    ]

    while pq:
        f, g, x, y = heapq.heappop(pq)
        if (x, y) == (gx, gy):
            path = []
            cur = (x, y)
            while cur is not None:
                path.append(cur)
                cx, cy = cur
                cur = parent[cy][cx]
            path.reverse()
            return path

        if g != gscore[y][x]:
            continue

        for dx, dy, step_len in neighbors:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if occ[ny][nx]:
                continue

            # prevent diagonal corner cutting
            if dx != 0 and dy != 0:
                if occ[y][nx] or occ[ny][x]:
                    continue

            ng = g + step_len
            if ng < gscore[ny][nx]:
                gscore[ny][nx] = ng
                parent[ny][nx] = (x, y)
                heapq.heappush(pq, (ng + heuristic((nx, ny), goal), ng, nx, ny))

    return []

    gscore = [[math.inf] * w for _ in range(h)]
    parent = [[None] * w for _ in range(h)]
    pq = []
    gscore[sy][sx] = 0.0
    heapq.heappush(pq, (heuristic(start, goal), 0.0, sx, sy))

    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    ]

    while pq:
        f, g, x, y = heapq.heappop(pq)
        if (x, y) == (gx, gy):
            path = []
            cur = (x, y)
            while cur is not None:
                path.append(cur)
                cx, cy = cur
                cur = parent[cy][cx]
            path.reverse()
            return path

        if g != gscore[y][x]:
            continue

        for dx, dy, step_len in neighbors:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if occ[ny][nx]:
                continue
            ng = g + step_len
            if ng < gscore[ny][nx]:
                gscore[ny][nx] = ng
                parent[ny][nx] = (x, y)
                heapq.heappush(pq, (ng + heuristic((nx, ny), goal), ng, nx, ny))

    return []

def path_to_waypoints(path, origin_x, origin_y, res):
    if not path:
        return []
    wps = []
    for i in range(0, len(path), WAYPOINT_STRIDE):
        gx, gy = path[i]
        wps.append(grid_to_world(gx, gy, origin_x, origin_y, res))
    gx, gy = path[-1]
    last = grid_to_world(gx, gy, origin_x, origin_y, res)
    if not wps or (abs(wps[-1][0] - last[0]) > 1e-6 or abs(wps[-1][1] - last[1]) > 1e-6):
        wps.append(last)
    return wps

def nearest_free_cell(occ, w, h, gx, gy, max_r=6):
    """If (gx,gy) is blocked, find nearest free cell within max_r (Manhattan ring)."""
    if occ is None:
        return gx, gy
    if not occ[gy][gx]:
        return gx, gy
    for r in range(1, max_r + 1):
        for dy in range(-r, r + 1):
            y = gy + dy
            if y < 0 or y >= h:
                continue
            dx = r - abs(dy)
            for sx in (-dx, dx):
                x = gx + sx
                if x < 0 or x >= w:
                    continue
                if not occ[y][x]:
                    return x, y
        for dx in range(-r + 1, r):
            x = gx + dx
            if x < 0 or x >= w:
                continue
            dy = r - abs(dx)
            for sy in (-dy, dy):
                y = gy + sy
                if y < 0 or y >= h:
                    continue
                if not occ[y][x]:
                    return x, y
    return gx, gy

# -------------------------
# Main
# -------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())
name = robot.getName()

lm = robot.getDevice("left wheel motor")
rm = robot.getDevice("right wheel motor")
lm.setPosition(float("inf"))
rm.setPosition(float("inf"))
lm.setVelocity(0.0)
rm.setVelocity(0.0)

gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

receiver = robot.getDevice("receiver")
receiver.enable(timestep)
try:
    receiver.setChannel(2)
except Exception:
    pass

emitter = robot.getDevice("emitter")
try:
    emitter.setChannel(2)
except Exception:
    pass

# Rescue data (filled on RESCUE_START)
origin_x = -1.0
origin_y = -1.0
res = 0.05
w = 40
h = 40
occ = None
victims = []

# Operator / command state
MODE_WAIT = "WAIT"
MODE_MOVE = "MOVE"
mode = MODE_WAIT

# Home is defined as the first pose after RESCUE_START (reported to supervisor)
home_xy = None
home_sent = False

# Current goal
goal_kind = None   # "VICTIM" or "HOME"
goal_id = None     # victim index (int) or None
goal_xy = None     # (x,y)

# Nav
waypoints = []
wp_i = 0
replan_counter = 0
step_counter = 0

def set_speed(v, w_cmd):
    vl = (v - (AXLE_LENGTH / 2.0) * w_cmd) / WHEEL_RADIUS
    vr = (v + (AXLE_LENGTH / 2.0) * w_cmd) / WHEEL_RADIUS
    vl = clamp(vl, -MOTOR_MAX_RAD_S, MOTOR_MAX_RAD_S)
    vr = clamp(vr, -MOTOR_MAX_RAD_S, MOTOR_MAX_RAD_S)
    lm.setVelocity(vl)
    rm.setVelocity(vr)

def get_pose_xy():
    g = gps.getValues()      # [x, y, z] in YOUR world
    c = compass.getValues()  # north vector
    x = float(g[0])
    y = float(g[1])
    th = heading_from_compass_xy(c)
    return x, y, th

def unpack_occ(s, w, h):
    if not s or len(s) < w * h:
        return None
    grid = [[False] * w for _ in range(h)]
    k = 0
    for yy in range(h):
        row = grid[yy]
        for xx in range(w):
            row[xx] = (s[k] == '1')
            k += 1
    return grid

def send(msg_dict):
    try:
        emitter.send(json.dumps(msg_dict))
    except Exception:
        # Webots also accepts bytes in some versions; keep fallback
        try:
            emitter.send(json.dumps(msg_dict).encode("utf-8"))
        except Exception:
            pass

def send_pose(t):
    x, y, th = get_pose_xy()
    send({"type": "POSE", "robot": name, "p": [x, y, th], "t": float(t)})

def plan_to_xy(tx, ty):
    global waypoints, wp_i, replan_counter
    if occ is None:
        return False
    x, y, _ = get_pose_xy()
    sx, sy = world_to_grid(x, y, origin_x, origin_y, res, w, h)
    gx, gy = world_to_grid(tx, ty, origin_x, origin_y, res, w, h)

    # If inflated obstacles mark start/goal as occupied, snap to nearest free cell
    sx, sy = nearest_free_cell(occ, w, h, sx, sy, max_r=6)
    gx, gy = nearest_free_cell(occ, w, h, gx, gy, max_r=6)

    path = astar(occ, w, h, (sx, sy), (gx, gy))
    if not path:
        return False

    waypoints = path_to_waypoints(path, origin_x, origin_y, res)
    wp_i = 0
    replan_counter = 0
    return True

def stop_and_wait():
    global mode, waypoints, wp_i, replan_counter
    set_speed(0.0, 0.0)
    waypoints = []
    wp_i = 0
    replan_counter = 0
    mode = MODE_WAIT

print(f"[{name}] rescue controller started (channel 2)")

while robot.step(timestep) != -1:
    step_counter += 1
    t = robot.getTime()

    # Periodic pose updates (supervisor preview needs it)
    if step_counter % POSE_SEND_EVERY_STEPS == 0:
        send_pose(t)

    # Receive commands / RESCUE_START
    while receiver.getQueueLength() > 0:
        msg = receiver.getString()
        receiver.nextPacket()
        try:
            data = json.loads(msg)
        except Exception:
            continue

        mtype = data.get("type", "")

        if mtype == "RESCUE_START":
            origin_x, origin_y = float(data["origin"][0]), float(data["origin"][1])
            res = float(data["res"])
            w = int(data["w"])
            h = int(data["h"])
            occ = unpack_occ(data.get("occ", ""), w, h)

            victims = []
            for v in data.get("victims", []):
                victims.append({"def": v.get("def", ""), "translation": v.get("translation", [0, 0, 0])})

            # Define HOME as current pose at the time rescue starts
            x0, y0, _ = get_pose_xy()
            home_xy = (x0, y0)
            home_sent = False

            print(f"[{name}] 🚑 RESCUE_START received: victims={len(victims)} grid={w}x{h} res={res}")
            stop_and_wait()
            goal_kind = None
            goal_id = None
            goal_xy = None

        elif mtype == "CMD_WAIT":
            stop_and_wait()

        elif mtype == "CMD_GO_HOME":
            if occ is None:
                continue
            p = data.get("p", None)
            if not p or len(p) < 2:
                continue
            tx, ty = float(p[0]), float(p[1])
            if plan_to_xy(tx, ty):
                goal_kind = "HOME"
                goal_id = None
                goal_xy = (tx, ty)
                mode = MODE_MOVE
                print(f"[{name}] 🏠 command: GO_HOME ({tx:.2f},{ty:.2f})")
            else:
                print(f"[{name}] ⚠️ no path to HOME")

        elif mtype == "CMD_GO_VICTIM":
            if occ is None:
                continue
            vid = data.get("id", None)
            p = data.get("p", None)
            if vid is None or not p or len(p) < 2:
                continue
            tx, ty = float(p[0]), float(p[1])
            if plan_to_xy(tx, ty):
                goal_kind = "VICTIM"
                goal_id = int(vid)
                goal_xy = (tx, ty)
                mode = MODE_MOVE
                print(f"[{name}] 🚑 command: GO_VICTIM {goal_id} ({tx:.2f},{ty:.2f})")
            else:
                print(f"[{name}] ⚠️ no path to VICTIM_{vid}")

    # If no map yet, idle
    if occ is None:
        set_speed(0.0, 0.0)
        continue

    # Send HOME once after RESCUE_START (once we know it)
    if home_xy is not None and (not home_sent):
        send({"type": "RESCUER_HOME", "robot": name, "p": [home_xy[0], home_xy[1]], "t": float(t)})
        home_sent = True

    # Waiting state
    if mode == MODE_WAIT or goal_xy is None:
        set_speed(0.0, 0.0)
        continue

    # Moving state
    replan_counter += 1
    if replan_counter >= REPLAN_EVERY_STEPS:
        replan_counter = 0
        plan_to_xy(goal_xy[0], goal_xy[1])

    x, y, th = get_pose_xy()
    tx, ty = goal_xy

    # Goal reached
    if math.hypot(tx - x, ty - y) <= GOAL_REACH_M:
        set_speed(0.0, 0.0)
        if goal_kind == "VICTIM":
            send({"type": "ARRIVED_VICTIM", "robot": name, "id": int(goal_id), "p": [tx, ty], "t": float(t)})
            print(f"[{name}] ✅ reached VICTIM_{goal_id} at x=[{tx:.3f}], y=[{ty:.3f}]")
        elif goal_kind == "HOME":
            send({"type": "ARRIVED_HOME", "robot": name, "p": [tx, ty], "t": float(t)})
            print(f"[{name}] ✅ reached HOME at x=[{tx:.3f}], y=[{ty:.3f}]")
        stop_and_wait()
        goal_kind = None
        goal_id = None
        goal_xy = None
        continue

    # Waypoint following
    if not waypoints:
        # If planning failed mid-way, rotate slowly to "unstick"
        set_speed(0.0, 0.9)
        continue

    while wp_i < len(waypoints) and math.hypot(waypoints[wp_i][0] - x, waypoints[wp_i][1] - y) < WAYPOINT_REACH_M:
        wp_i += 1
    if wp_i >= len(waypoints):
        # replan towards goal
        plan_to_xy(goal_xy[0], goal_xy[1])
        continue

    wx, wy = waypoints[wp_i]
    desired = math.atan2(wy - y, wx - x)
    err = wrap_pi(desired - th)

    v = V_FWD * (1.0 if abs(err) < SLOW_ANGLE else 0.55)
    w_cmd = W_GAIN * err
    set_speed(v, w_cmd)