# controllers/epuck_controller/epuck_controller.py
# Exploration + LiDAR mapping + frontier bidding + A*/DIRECT navigation
# + baseline-calibrated + persistent obstacle avoidance (ps0..ps7)
# + Camera RECOGNITION-based victim detection (no LiDAR dependency)
#
# IMPORTANT (your world):
# - Ground plane is X–Y
# - Height is Z
#
# CHANGE (requested):
# - Scouts go HOME ONLY when supervisor sends TARGET with tag "HOME".
# - Removed local "0 frontiers for HOLD seconds -> go home" trigger.
# - Everything else stays the same.

from controller import Robot
import math
import json
import heapq

# -------------------------
# Map parameters (2x2m arena)
# -------------------------
MAP_SIZE_M = 2.0
RES = 0.05
W = int(MAP_SIZE_M / RES)
H = W

UNKNOWN = -1
FREE = 0
OCC = 1

# Frontier / bidding
FRONTIER_MIN_DIST_M = 0.08
MAX_FRONTIERS_TO_SEND = 12
BID_PERIOD_STEPS = 10
RETURN_REPLAN_EVERY_STEPS = 15  # replan interval during supervisor-map RETURN_HOME

# Lidar integration
NO_HIT_FREE = 0.20
RAY_STEP = RES

# Info gain
INFO_RADIUS_M = 0.70
PRINT_EVERY_BID = 8

# Planning
COST_FREE = 1.0
COST_UNKNOWN = 3.0
OBSTACLE_INFLATION_CELLS = 1
WAYPOINT_STRIDE = 4
WAYPOINT_REACH_M = 0.08
GOAL_SNAP_MAX_RADIUS_CELLS = 14
REPLAN_PERIOD_STEPS = 120

# Target completion
TARGET_REACH_M = 0.09
TARGET_FACE_ERR_MAX = 1.4

# -------------------------
# Persistent avoidance (ps sensors)
# -------------------------
PS_FRONT_MARGIN = 70.0
PS_SIDE_MARGIN = 50.0
AVOID_HOLD_STEPS = 18
ESCAPE_HOLD_STEPS = 26

AVOID_TURN_W = 3.2
AVOID_FWD_V = 0.01
AVOID_BACK_V = -0.04

# -------------------------
# e-puck constants
# -------------------------
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
MOTOR_MAX_RAD_S = 6.28

# -------------------------
# Recognition-based victim detection
# -------------------------
VICTIM_NAME_PREFIX = "VICTIM_"
VICTIM_CHECK_EVERY_STEPS = 3
VICTIM_COOLDOWN_S = 2.0
VICTIM_MERGE_DIST_M = 0.40
VICTIM_GLOBAL_COOLDOWN_S = 0.35
VICTIM_MAX_RANGE_M = 2.5

# Camera offset in robot frame (meters)
# camera coords: +x right, +y up, +z forward
CAM_OFFSET_RIGHT = 0.0
CAM_OFFSET_FWD = 0.02

# -------------------------
# Home / parking (supervisor-driven)
# -------------------------
HOME_REACH_M = 0.10

VICTIM_DEBUG = False

# -------------------------
# Utility
# -------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def is_bad(v):
    return v is None or math.isnan(v) or math.isinf(v)

# World is X–Y ground, Z height:
# Grid origin is (-1,-1) to (+1,+1)
ORIGIN_X = -1.0
ORIGIN_Y = -1.0

def world_to_grid(x, y):
    gx = int((x - ORIGIN_X) / RES)
    gy = int((y - ORIGIN_Y) / RES)
    gx = clamp(gx, 0, W - 1)
    gy = clamp(gy, 0, H - 1)
    return gx, gy

def grid_to_world(gx, gy):
    x = ORIGIN_X + (gx + 0.5) * RES
    y = ORIGIN_Y + (gy + 0.5) * RES
    return x, y

# -------------------------
# Odometry (X–Y ground)
# -------------------------
def odom_update(x, y, th, dl, dr):
    dc = (dr + dl) * 0.5
    dth = (dr - dl) / AXLE_LENGTH
    th2 = wrap_pi(th + dth)
    mid = wrap_pi(th + 0.5 * dth)
    x2 = x + dc * math.cos(mid)
    y2 = y + dc * math.sin(mid)
    return x2, y2, th2

# -------------------------
# LiDAR integration (X–Y ground)
# -------------------------
def integrate_lidar(grid, rx, ry, rth, ranges, fov, max_range):
    n = len(ranges)
    if n <= 1:
        return
    step = RAY_STEP

    for k, r in enumerate(ranges):
        if r is None:
            continue
        if isinstance(r, float) and math.isnan(r):
            continue
        if r <= 0.0:
            continue

        hit = True
        if math.isinf(r) or r >= max_range:
            hit = False
            r = NO_HIT_FREE

        ang = rth + (-0.5 * fov + (k * fov / (n - 1)))

        dist = 0.0
        while dist < r:
            x = rx + dist * math.cos(ang)
            y = ry + dist * math.sin(ang)
            gx, gy = world_to_grid(x, y)
            if grid[gy][gx] == UNKNOWN:
                grid[gy][gx] = FREE
            dist += step

        if hit:
            hx = rx + r * math.cos(ang)
            hy = ry + r * math.sin(ang)
            gx, gy = world_to_grid(hx, hy)
            grid[gy][gx] = OCC

# -------------------------
# Frontiers / downsample
# -------------------------
def find_frontiers(grid):
    fr = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if grid[y][x] != FREE:
                continue
            if (grid[y][x-1] == UNKNOWN or grid[y][x+1] == UNKNOWN or
                grid[y-1][x] == UNKNOWN or grid[y+1][x] == UNKNOWN):
                fr.append((x, y))
    return fr

def downsample_frontiers(frontiers):
    min_cells = max(1, int(FRONTIER_MIN_DIST_M / RES))
    kept = []
    for fx, fy in frontiers:
        ok = True
        for kx, ky in kept:
            if abs(fx - kx) <= min_cells and abs(fy - ky) <= min_cells:
                ok = False
                break
        if ok:
            kept.append((fx, fy))
    return kept

# -------------------------
# Info gain
# -------------------------
def info_gain_and_rect(grid, fx, fy, radius_m):
    r_cells = int(radius_m / RES)
    r2 = r_cells * r_cells
    gain = 0
    minx = maxx = fx
    miny = maxy = fy

    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            if dx*dx + dy*dy > r2:
                continue
            x = fx + dx
            y = fy + dy
            if x < 0 or x >= W or y < 0 or y >= H:
                continue
            if grid[y][x] == UNKNOWN:
                gain += 1
                if x < minx: minx = x
                if x > maxx: maxx = x
                if y < miny: miny = y
                if y > maxy: maxy = y

    return gain, [minx, maxx, miny, maxy]

# -------------------------
# A* helpers
# -------------------------
def inflated_occupied(grid, infl):
    occ = [[False] * W for _ in range(H)]
    if infl <= 0:
        for y in range(H):
            for x in range(W):
                occ[y][x] = (grid[y][x] == OCC)
        return occ
    for y in range(H):
        for x in range(W):
            if grid[y][x] != OCC:
                continue
            for dy in range(-infl, infl + 1):
                for dx in range(-infl, infl + 1):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        occ[ny][nx] = True
    return occ

def cell_cost(grid, x, y):
    v = grid[y][x]
    if v == FREE:
        return COST_FREE
    if v == UNKNOWN:
        return COST_UNKNOWN
    return math.inf

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def snap_goal(goal, occ):
    gx, gy = goal
    if 0 <= gx < W and 0 <= gy < H and not occ[gy][gx]:
        return goal
    for r in range(1, GOAL_SNAP_MAX_RADIUS_CELLS + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < W and 0 <= ny < H and not occ[ny][nx]:
                    return (nx, ny)
    return None

def astar(grid, start, goal, occ):
    sx, sy = start
    gx, gy = goal
    if occ[sy][sx] or occ[gy][gx]:
        return []

    gscore = [[math.inf] * W for _ in range(H)]
    parent = [[None] * W for _ in range(H)]
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
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            if occ[ny][nx]:
                continue
            step_cost = cell_cost(grid, nx, ny)
            if math.isinf(step_cost):
                continue
            ng = g + step_len * step_cost
            if ng < gscore[ny][nx]:
                gscore[ny][nx] = ng
                parent[ny][nx] = (x, y)
                heapq.heappush(pq, (ng + heuristic((nx, ny), goal), ng, nx, ny))

    return []

def path_to_waypoints(path):
    if not path:
        return []
    wps = []
    for i in range(0, len(path), WAYPOINT_STRIDE):
        cx, cy = path[i]
        wps.append(grid_to_world(cx, cy))
    lx, ly = path[-1]
    last = grid_to_world(lx, ly)
    if not wps or (abs(wps[-1][0] - last[0]) > 1e-6 or abs(wps[-1][1] - last[1]) > 1e-6):
        wps.append(last)
    return wps

# -------------------------
# Recognition helpers
# -------------------------
def recog_name(obj):
    for fn in ("getModel", "getDef", "getName"):
        try:
            val = getattr(obj, fn)()
            if val:
                return str(val)
        except Exception:
            pass
    return ""


def cam_rel_to_world(robot_x, robot_y, robot_th, cam_pos):
    # camera coords: right=pos[0], up=pos[1], forward=pos[2]
    right = float(cam_pos[0]) + CAM_OFFSET_RIGHT
    fwd = float(cam_pos[2]) + CAM_OFFSET_FWD

    dx = fwd * math.cos(robot_th) - right * math.sin(robot_th)
    dy = fwd * math.sin(robot_th) + right * math.cos(robot_th)

    d = math.hypot(dx, dy)
    return (robot_x + dx, robot_y + dy, d)

# -------------------------
# Main
# -------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())
name = robot.getName()

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

left_ps = robot.getDevice("left wheel sensor")
right_ps = robot.getDevice("right wheel sensor")
left_ps.enable(timestep)
right_ps.enable(timestep)

# Proximity sensors (ps0..ps7)
ps = []
for i in range(8):
    s = robot.getDevice(f"ps{i}")
    s.enable(timestep)
    ps.append(s)

lidar = robot.getDevice("lidar")
lidar.enable(timestep)

camera = robot.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)

receiver = robot.getDevice("receiver")
receiver.enable(timestep)
emitter = robot.getDevice("emitter")

# -------------------------
# Mission gating (Ground vs Aerial)
# - Supervisor sends {"type":"MISSION","mode":"G"} or {"type":"MISSION","mode":"A"} on SCOUT channel.
# - In mode "A": e-puck stays inactive/silent (no bids, no victim msgs, no movement).
# - In mode "G": normal behavior.
# -------------------------
mission_mode = None   # None until supervisor selects, then "G" or "A"

# Optional ground-truth pose sensors (used ONLY for supervisor-map RETURN_HOME)
gps = None
compass = None
try:
    gps = robot.getDevice("gps")
    gps.enable(timestep)
except Exception:
    gps = None
try:
    compass = robot.getDevice("compass")
    compass.enable(timestep)
except Exception:
    compass = None

def _heading_from_compass_xy(north):
    # For X–Y ground plane (Z up): use x,y components
    try:
        return math.atan2(float(north[0]), float(north[1]))
    except Exception:
        return 0.0

def _get_pose_gt_if_available():
    if gps is None or compass is None:
        return None
    try:
        g = gps.getValues()
        c = compass.getValues()
        return (float(g[0]), float(g[1]), _heading_from_compass_xy(c))
    except Exception:
        return None

# Home targets (X,Y). Z is height and ignored here.
HOME = (-0.75, -0.85) if name == "epuck1" else (-0.85, -0.85)

# --- supervisor-driven RETURN_HOME state ---
home_from_sup = None        # (x,y) ground-truth home from supervisor
global_map_ready = False
global_origin = (0.0, 0.0)
global_res = RES
global_w = 0
global_h = 0
global_occ = None           # 2D bool grid (True=blocked)
frozen = False              # stop-in-place latch before RETURN_HOME
returning_home_global = False
return_replan_counter = 0  # used only in supervisor-map RETURN_HOME


# Initial pose (hardcoded to match your start)
if name == "epuck1":
    x, y, th = -0.75, -0.85, 0.0
else:
    x, y, th = -0.85, -0.85, 0.0

# Warmup wheel sensors
prev_l = left_ps.getValue()
prev_r = right_ps.getValue()
for _ in range(20):
    robot.step(timestep)
    prev_l = left_ps.getValue()
    prev_r = right_ps.getValue()

# -------- Baseline calibration for ps sensors --------
base_acc = [0.0] * 8
base_n = 0
for _ in range(32):
    robot.step(timestep)
    vals = [s.getValue() for s in ps]
    for i in range(8):
        base_acc[i] += vals[i]
    base_n += 1
ps_base = [v / base_n for v in base_acc]
print(f"[{name}] ps baseline={['%.1f' % v for v in ps_base]}")
print(f"[{name}] camera {camera.getWidth()}x{camera.getHeight()} fov={camera.getFov():.2f} recognition=ON")
print(f"[{name}] HOME target (x,y)=({HOME[0]:.2f},{HOME[1]:.2f})")

grid = [[UNKNOWN] * W for _ in range(H)]

has_target = False
target_world = (0.0, 0.0)  # (x,y)
target_cell = None

nav_mode = "EXPLORE"
waypoints = []
wp_idx = 0
replan_timer = 0

bid_count = 0
step_count = 0

# Persistent avoidance state
avoid_timer = 0
avoid_v = 0.0
avoid_w = 0.0

# Victim reporting state
victim_last_time = {}
victim_last_pos = {}
last_global_victim_time = -1e9

# ✅ Supervisor-driven home/park state
returning_home = False
parked_sent = False

def set_speed(v, w_cmd):
    vl = (v - (AXLE_LENGTH / 2.0) * w_cmd) / WHEEL_RADIUS
    vr = (v + (AXLE_LENGTH / 2.0) * w_cmd) / WHEEL_RADIUS
    vl = clamp(vl, -MOTOR_MAX_RAD_S, MOTOR_MAX_RAD_S)
    vr = clamp(vr, -MOTOR_MAX_RAD_S, MOTOR_MAX_RAD_S)
    left_motor.setVelocity(vl)
    right_motor.setVelocity(vr)

def dist_to_target():
    tx, ty = target_world
    return math.hypot(tx - x, ty - y)

def desired_err_to(px, py):
    desired = math.atan2(py - y, px - x)
    return wrap_pi(desired - th)

def at_waypoint(wx, wy):
    return math.hypot(wx - x, wy - y) < WAYPOINT_REACH_M

def read_ps():
    return [s.getValue() for s in ps]

def avoidance_decide(psv):
    front_th = max(ps_base[0], ps_base[7]) + PS_FRONT_MARGIN
    side_th = max(ps_base[1], ps_base[6]) + PS_SIDE_MARGIN

    front = max(psv[0], psv[7])
    right = max(psv[0], psv[1], psv[2])
    left = max(psv[7], psv[6], psv[5])

    if front > front_th:
        w = -AVOID_TURN_W if left > right else AVOID_TURN_W
        return True, ESCAPE_HOLD_STEPS, AVOID_BACK_V, w

    if left > side_th or right > side_th:
        if left > right:
            return True, AVOID_HOLD_STEPS, AVOID_FWD_V, -AVOID_TURN_W
        else:
            return True, AVOID_HOLD_STEPS, AVOID_FWD_V, AVOID_TURN_W

    return False, 0, 0.0, 0.0


def world_to_global_grid(wx, wy, origin, res, w, h):
    ox, oy = origin
    gx = int((wx - ox) / res)
    gy = int((wy - oy) / res)
    gx = max(0, min(w - 1, gx))
    gy = max(0, min(h - 1, gy))
    return gx, gy

def global_grid_to_world(gx, gy, origin, res):
    ox, oy = origin
    return (ox + (gx + 0.5) * res, oy + (gy + 0.5) * res)

def plan_to_home_global(current_xy, home_xy):
    global waypoints, wp_idx, nav_mode
    if not global_map_ready or global_occ is None:
        nav_mode = "DIRECT"
        waypoints = []
        wp_idx = 0
        return

    cx, cy = current_xy
    hx, hy = home_xy

    sg = world_to_global_grid(cx, cy, global_origin, global_res, global_w, global_h)
    gg = world_to_global_grid(hx, hy, global_origin, global_res, global_w, global_h)

    path = astar_occ(global_occ, global_w, global_h, sg, gg)
    if path:
        waypoints = [global_grid_to_world(px, py, global_origin, global_res) for (px, py) in path[1:]]
        wp_idx = 0
        nav_mode = "PATH"
    else:
        waypoints = []
        wp_idx = 0
        nav_mode = "DIRECT"

def plan_to_target():
    global waypoints, wp_idx, nav_mode, replan_timer
    waypoints = []
    wp_idx = 0
    replan_timer = 0

    if target_cell is None:
        nav_mode = "EXPLORE"
        return

    rgx, rgy = world_to_grid(x, y)

    for infl in (OBSTACLE_INFLATION_CELLS, 0):
        occ = inflated_occupied(grid, infl)

        # clear immediate neighborhood
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx = rgx + dx
                ny = rgy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    occ[ny][nx] = False

        g = snap_goal(target_cell, occ)
        if g is None:
            continue

        path = astar(grid, (rgx, rgy), g, occ)
        if path:
            waypoints = path_to_waypoints(path)
            nav_mode = "PATH"
            return

    nav_mode = "DIRECT"


# -------------------------
# Global-map (supervisor) A* helpers for RETURN_HOME
# (mirrors khepera3_rescue_controller logic; does not affect EXPLORE mapping)
# -------------------------

def unpack_occ_str(occ_str, w, h):
    # occ_str is a row-major string of '0'/'1' with length w*h
    occ = [[False] * w for _ in range(h)]
    if not occ_str:
        return occ
    n = min(len(occ_str), w * h)
    k = 0
    for y in range(h):
        row = occ[y]
        for x in range(w):
            if k >= n:
                return occ
            row[x] = (occ_str[k] == "1")
            k += 1
    return occ

def astar_occ(occ, w, h, start, goal):
    sx, sy = start
    gx, gy = goal
    if sx < 0 or sx >= w or sy < 0 or sy >= h:
        return []
    if gx < 0 or gx >= w or gy < 0 or gy >= h:
        return []
    if occ[sy][sx] or occ[gy][gx]:
        return []

    gscore = [[math.inf] * w for _ in range(h)]
    parent = [[None] * w for _ in range(h)]
    pq = []
    gscore[sy][sx] = 0.0
    heapq.heappush(pq, (heuristic((sx, sy), (gx, gy)), 0.0, sx, sy))

    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    ]

    while pq:
        _, g, x, y = heapq.heappop(pq)
        if (x, y) == (gx, gy):
            path = []
            cx, cy = gx, gy
            while (cx, cy) != (sx, sy):
                path.append((cx, cy))
                cx, cy = parent[cy][cx]
            path.append((sx, sy))
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
                f = ng + heuristic((nx, ny), (gx, gy))
                heapq.heappush(pq, (f, ng, nx, ny))
    return []

def try_report_victims(sim_time):
    global last_global_victim_time

    objs = camera.getRecognitionObjects()
    if not objs:
        return

    if sim_time - last_global_victim_time < VICTIM_GLOBAL_COOLDOWN_S:
        return

    for obj in objs:
        model = recog_name(obj)
        if not model.startswith(VICTIM_NAME_PREFIX):
            continue

        rid = None
        try:
            rid = int(obj.getId())
        except Exception:
            rid = None
        key = f"{model}:{rid}" if rid is not None else model

        last_t = victim_last_time.get(key, -1e9)
        if sim_time - last_t < VICTIM_COOLDOWN_S:
            continue

        try:
            pos = obj.getPosition()
        except Exception:
            continue
        if pos is None or len(pos) < 3:
            continue

        vx, vy, d = cam_rel_to_world(x, y, th, pos)
        if is_bad(d) or d <= 0.0 or d > VICTIM_MAX_RANGE_M:
            continue

        last_p = victim_last_pos.get(key, None)
        if last_p is not None:
            if math.hypot(vx - last_p[0], vy - last_p[1]) < VICTIM_MERGE_DIST_M:
                continue

        victim_last_time[key] = sim_time
        victim_last_pos[key] = (vx, vy)
        last_global_victim_time = sim_time

        meta = {"victimName": model, "victimId": rid}
        emitter.send(json.dumps({
            "type": "VICTIM",
            "robot": name,
            "p": [float(vx), float(vy)],  # X–Y ground!
            "meta": meta
        }))

        print(f"[{name}] ✅ VICTIM recog: {model} (sent to supervisor)")
        return

print(f"[{name}] controller started")

while robot.step(timestep) != -1:
    sim_time = robot.getTime()

    # ---- mission selection gate ----
    # Always listen for MISSION messages even before start.
    while receiver.getQueueLength() > 0:
        _raw = receiver.getString()
        receiver.nextPacket()
        try:
            _data = json.loads(_raw)
        except Exception:
            continue
        if _data.get("type") == "MISSION":
            mm = str(_data.get("mode", "")).upper()
            if mm in ("G", "A"):
                mission_mode = mm
                if mission_mode == "A":
                    # hard stop + silence
                    frozen = True
                    returning_home = False
                    returning_home_global = False
                    has_target = False
                    waypoints = []
                    wp_idx = 0
                    set_speed(0.0, 0.0)
                print(f"[{name}] mission mode set to {mission_mode}")
        else:
            # put non-mission messages back into a small local buffer by handling below:
            # we can't un-read receiver packets, so store it for later processing this step
            # using a simple list.
            try:
                _pending_msgs.append(_data)
            except NameError:
                _pending_msgs = [_data]

    # If mission is not chosen yet: stay idle and do nothing.
    if mission_mode is None:
        set_speed(0.0, 0.0)
        continue

    # Aerial mission: keep scout inactive and silent forever.
    if mission_mode == "A":
        set_speed(0.0, 0.0)
        continue


    sim_time = robot.getTime()

    # ---- PARKED LATCH ----
    # Once we declare parked, we must stay parked: motors forced to zero and no more scouting/bidding.
    if parked_sent:
        nav_mode = "PARKED"
        set_speed(0.0, 0.0)
        continue

    # ---- odometry ----
    l = left_ps.getValue()
    r = right_ps.getValue()
    if (not is_bad(l)) and (not is_bad(r)) and (not is_bad(prev_l)) and (not is_bad(prev_r)):
        dl = (l - prev_l) * WHEEL_RADIUS
        dr = (r - prev_r) * WHEEL_RADIUS
        prev_l, prev_r = l, r
        if not is_bad(dl) and not is_bad(dr):
            x, y, th = odom_update(x, y, th, dl, dr)

# ---- supervisor-map RETURN_HOME uses GT pose if available ----
    if returning_home_global:
        pgt = _get_pose_gt_if_available()
        if pgt is not None:
            x, y, th = pgt
        # Periodic replanning on supervisor map (Khepera3-style)
        if (home_from_sup is not None) and global_map_ready:
            return_replan_counter += 1
            if (not waypoints) or (return_replan_counter >= RETURN_REPLAN_EVERY_STEPS):
                return_replan_counter = 0
                plan_to_home_global((x, y), home_from_sup)


    # ---- map stability: keep robot neighborhood FREE ----
    rgx, rgy = world_to_grid(x, y)
    grid[rgy][rgx] = FREE
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nx = rgx + dx
            ny = rgy + dy
            if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] != OCC:
                grid[ny][nx] = FREE

    # ---- mapping ----
    ranges = lidar.getRangeImage()
    integrate_lidar(grid, x, y, th, ranges, lidar.getFov(), lidar.getMaxRange())

    # ---- victim detection (SCOUTING ONLY; disabled in RETURN_HOME/frozen) ----
    step_count += 1
    if (not returning_home_global) and (not frozen) and (step_count % VICTIM_CHECK_EVERY_STEPS == 0):
        try_report_victims(sim_time)    # ---- receive TARGET ----
    got_new_target = False

    # Messages already pulled during mission gating (non-MISSION)
    _msgs = []
    try:
        _msgs.extend(_pending_msgs)
        _pending_msgs = []
    except Exception:
        pass

    # Drain any new receiver packets
    while receiver.getQueueLength() > 0:
        _raw = receiver.getString()
        receiver.nextPacket()
        try:
            _msgs.append(json.loads(_raw))
        except Exception:
            pass

    for data in _msgs:
        try:
            if data.get("type") == "TARGET" and data.get("robot") == name:
                tx, ty = float(data["p"][0]), float(data["p"][1])
                target_world = (tx, ty)
                target_cell = world_to_grid(tx, ty)
                has_target = True
                got_new_target = True

                # Supervisor decides HOME using tag
                tag = data.get("tag", "")
                returning_home = (tag == "HOME")

            elif data.get("type") == "STOP" and data.get("robot") == name:
                # Supervisor requests freeze-in-place (used right after scouting completes)
                frozen = True
                returning_home = False
                returning_home_global = False
                return_replan_counter = 0
                has_target = False
                waypoints = []
                wp_idx = 0
                set_speed(0.0, 0.0)
                emitter.send(json.dumps({"type": "SCOUT_FROZEN", "robot": name}))

            elif data.get("type") == "SCOUT_RETURN_HOME" and data.get("robot") == name:
                # Enter supervisor-map RETURN_HOME mode (global A*)
                frozen = False
                returning_home = True
                returning_home_global = True

                return_replan_counter = 0
                # map params
                global_origin = (float(data["origin"][0]), float(data["origin"][1]))
                global_res = float(data["res"])
                global_w = int(data["w"])
                global_h = int(data["h"])
                global_occ = unpack_occ_str(data.get("occ", ""), global_w, global_h)
                global_map_ready = True

                # start + goal (ground truth from supervisor)
                sx, sy = float(data["start"][0]), float(data["start"][1])
                hx, hy = float(data["home"][0]), float(data["home"][1])
                home_from_sup = (hx, hy)

                # reset our pose translation to ground-truth start (and reset wheel sensor deltas)
                x = sx
                y = sy
                try:
                    prev_l = left_ps.getValue()
                    prev_r = right_ps.getValue()
                except Exception:
                    pass

                # set target to home
                target_world = (hx, hy)
                has_target = True
                got_new_target = True
        except Exception:
            pass

    # If supervisor froze us (between scouting and RETURN_HOME), hold position.
    if frozen and (not returning_home_global):
        set_speed(0.0, 0.0)
        continue

    if got_new_target:
        if returning_home_global and home_from_sup is not None:
            plan_to_home_global((x, y), home_from_sup)
        else:
            plan_to_target()

    # ---- BID (SCOUTING ONLY; disabled in RETURN_HOME/frozen) ----
    if (not returning_home_global) and (not frozen) and (step_count % BID_PERIOD_STEPS == 0):
        bid_count += 1
        raw_frontiers = find_frontiers(grid)
        frontiers = downsample_frontiers(raw_frontiers)

        candidates = []
        for fx, fy in frontiers:
            dx = fx - rgx
            dy = fy - rgy
            c = math.hypot(dx, dy) * RES
            i, rect = info_gain_and_rect(grid, fx, fy, INFO_RADIUS_M)
            u = float(i) - float(c)
            wx, wy = grid_to_world(fx, fy)
            candidates.append((u, wx, wy, c, i, rect))

        candidates.sort(key=lambda t: t[0], reverse=True)
        candidates = candidates[:MAX_FRONTIERS_TO_SEND]

        if bid_count % PRINT_EVERY_BID == 0:
            dt = dist_to_target() if has_target else -1.0
            print(f"[{name}] mode={nav_mode} tgt={int(has_target)} wps={len(waypoints)} dT={dt:.2f}")

        emitter.send(json.dumps({
            "type": "BID",
            "robot": name,
            "pose": [x, y, th],
            "res": RES,
            "frontiers": [{"p": [wx, wy], "c": c, "i": i, "r": rect}
                          for (_, wx, wy, c, i, rect) in candidates],
        }))

    # ✅ Removed: local scouting done -> drive home
    # (Supervisor will send TARGET with tag "HOME" when it's time.)

    # ---- completion (target reached) ----
    if has_target:
        d = dist_to_target()
        face_err = abs(desired_err_to(target_world[0], target_world[1]))
        if d < TARGET_REACH_M and face_err < TARGET_FACE_ERR_MAX:
            has_target = False
            nav_mode = "EXPLORE"
            waypoints = []
            wp_idx = 0
            set_speed(0.0, 0.0)

            # if we were returning home, declare parked
            if returning_home and (math.hypot(x - (home_from_sup[0] if home_from_sup is not None else HOME[0]), y - (home_from_sup[1] if home_from_sup is not None else HOME[1])) <= HOME_REACH_M) and (not parked_sent):
                parked_sent = True
                print(f"[{name}] 🏁 parked at home ({x:.2f},{y:.2f})")
                emitter.send(json.dumps({"type": "SCOUT_PARKED", "robot": name}))
            continue

    # ---- periodic replan ----
    if has_target:
        replan_timer += 1
        if replan_timer >= REPLAN_PERIOD_STEPS:
            replan_timer = 0
            plan_to_target()

    # -------------------------
    # Persistent avoidance logic
    # -------------------------
    psv = read_ps()

    if avoid_timer <= 0:
        trig, hold, v_cmd, w_cmd = avoidance_decide(psv)
        if trig:
            avoid_timer = hold
            avoid_v = v_cmd
            avoid_w = w_cmd

    if avoid_timer > 0:
        avoid_timer -= 1
        set_speed(avoid_v, avoid_w)
        continue

    # -------------------------
    # Navigation
    # -------------------------
    if has_target and nav_mode == "PATH" and waypoints:
        while wp_idx < len(waypoints) and at_waypoint(waypoints[wp_idx][0], waypoints[wp_idx][1]):
            wp_idx += 1

        if wp_idx >= len(waypoints):
            if returning_home_global and home_from_sup is not None:
                plan_to_home_global((x, y), home_from_sup)
                # if still no waypoints, rotate a bit to unstick
                if not waypoints:
                    set_speed(0.0, 0.9)
                    continue
            else:
                nav_mode = "DIRECT"
        else:
            wx, wy = waypoints[wp_idx]
            err = desired_err_to(wx, wy)
            v = 0.11 * (1.0 if abs(err) < 0.9 else 0.55)
            w_cmd = 1.5 * err
            set_speed(v, w_cmd)
            continue

    if has_target and nav_mode == "DIRECT":
        tx, ty = target_world
        err = desired_err_to(tx, ty)
        if returning_home_global and (home_from_sup is not None) and global_map_ready:
            # If we are using DIRECT in RETURN_HOME, try to replan instead of pushing into obstacles
            plan_to_home_global((x, y), home_from_sup)
            if waypoints:
                nav_mode = "PATH"
                continue
        v = 0.11 * (1.0 if abs(err) < 0.9 else 0.55)
        w_cmd = 1.4 * err
        set_speed(v, w_cmd)
        continue

    # Explore when no target
    nav_mode = "EXPLORE"
    set_speed(0.04, 0.10)