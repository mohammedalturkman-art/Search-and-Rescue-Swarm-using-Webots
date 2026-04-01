# controllers/supervisor_controller/supervisor_controller.py
# Webots R2025a – Python
#
# Supervisor:
# 1) Frontier assignment for scouts (channel 1)
# 2) Victim storage (from scouts) + ground-truth print (x,y,z)
# 3) Detects "scouting complete" (both scouts: 0 frontiers for hold time AND >= expected victims)
# 4) Sends HOME targets to scouts (drive home, no teleport)
# 5) Waits for SCOUT_PARKED from BOTH scouts
# 6) Starts rescue: sends RESCUE_START to Khepera3 (channel 2) containing:
#    - Ground-truth victim translations (x,y,z)
#    - A supervisor-built occupancy grid from obstacle bounding boxes (for A* shortest path)
#
# NEW (operator-in-the-loop dispatch):
# - After RESCUE_START, the supervisor does NOT auto-send "nearest victim".
# - The supervisor enters DISPATCH MODE and the operator chooses victims:
#     * Type 1/2/3... to dispatch to VICTIM_n
#     * Type H to go home
# - Before dispatch, the supervisor prints a preview:
#     victim coords, helped status, shortest-path length and ETA
# - If a victim is already helped, supervisor asks for confirmation (Y/N) first.
# - If operator presses H while rescuer already at home: message shown, no command sent.

from controller import Supervisor, Keyboard
import json
import math
import heapq

# -------------------------
# Channels
# -------------------------
SCOUT_CHANNEL = 1
RESCUE_CHANNEL = 2

# -------------------------
# Scouts and home positions (X,Y)
# -------------------------
SCOUTS = ("epuck1", "epuck2")
SCOUT_HOME = {
    "epuck1": (-0.75, -0.85),
    "epuck2": (-0.85, -0.85),
}

# -------------------------
# Victims (ground-truth via DEF names)
# -------------------------
EXPECTED_VICTIMS = 3
VICTIM_DEFS = ["VICTIM_1", "VICTIM_2", "VICTIM_3"]  # must match DEF names in scene tree

# -------------------------
# Scouting completion gate
# -------------------------
SCOUT_DONE_HOLD_S = 4.0
REQUIRE_EXPECTED_VICTIMS = True

# -------------------------
# Frontier assignment config
# -------------------------
COVERAGE_POINTS = [
    (-0.75, -0.75),
    (-0.75,  0.75),
    ( 0.75, -0.75),
    ( 0.75,  0.75),
    ( 0.00,  0.75),
    ( 0.00, -0.75),
    ( 0.75,  0.00),
    (-0.75,  0.00),
    ( 0.00,  0.00),
]
COVER_REACH = 0.22
STALL_TIME = 12.0

TABOO_RADIUS = 0.35
TABOO_TIME = 10.0
CLAIM_RADIUS = 0.35
CLAIM_TIME = 18.0
WATCHDOG_TIME = 6.0
WATCHDOG_MIN_PROGRESS = 0.15
REACH_DIST = 0.12
ASSIGN_PERIOD = 0.5
RELAX_TABOO_IF_EMPTY = True
RELAX_CLAIMS_IF_EMPTY = True
FORCE_FARTHEST_ON_WATCHDOG = True

# -------------------------
# Victim storage parameters
# -------------------------
VICTIM_MERGE_DIST = 0.85
NEW_VICTIM_COOLDOWN = 4.0
VICTIM_MAX_STORE = 50

# -------------------------
# Occupancy grid (for Khepera A*)
# -------------------------
# Arena spans [-1,+1] x [-1,+1] in (x,y)
MAP_SIZE_M = 2.0
RES = 0.05
W = int(MAP_SIZE_M / RES)
H = W
ORIGIN_X = -1.0
ORIGIN_Y = -1.0

# Inflate obstacles a bit for safer paths
INFLATE_M = 0.10
ROBOT_RADIUS_M = 0.05  # v11: slightly smaller footprint radius to keep corridors open
EXTRA_CLEARANCE_M = 0.00  # v11: remove extra margin; goal clearing handles safety

# -------------------------
# Operator dispatch config
# -------------------------
HOME_REACH_M = 0.14
NOMINAL_RESCUER_SPEED_MPS = 0.20  # for ETA preview (approx)

# -------------------------
# Helpers (X,Y ground)
# -------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy

def dist(a, b):
    return math.sqrt(dist2(a, b))

def world_to_grid(x, y):
    gx = int((x - ORIGIN_X) / RES)
    gy = int((y - ORIGIN_Y) / RES)
    return clamp(gx, 0, W - 1), clamp(gy, 0, H - 1)

def grid_to_world(gx, gy):
    x = ORIGIN_X + (gx + 0.5) * RES
    y = ORIGIN_Y + (gy + 0.5) * RES
    return x, y

# -------------------------
# A* for preview (uses occ grid built by supervisor)
# -------------------------

def _clear_free_around(grid, x, y, r_cells=1):
    """Force cells around (x,y) world coordinate to be free (0)."""
    gx, gy = world_to_grid(x, y)
    h = len(grid)
    w = len(grid[0]) if h else 0
    for dy in range(-r_cells, r_cells + 1):
        yy = gy + dy
        if yy < 0 or yy >= h:
            continue
        for dx in range(-r_cells, r_cells + 1):
            xx = gx + dx
            if 0 <= xx < w:
                grid[yy][xx] = 0


def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def astar(occ, w, h, start, goal):
    """A* on 8-neighborhood grid with *no corner cutting* (safer near walls)."""
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
            ng = g + step_len
            if ng < gscore[ny][nx]:
                gscore[ny][nx] = ng
                parent[ny][nx] = (x, y)
                heapq.heappush(pq, (ng + heuristic((nx, ny), goal), ng, nx, ny))

    return []

def path_length_m(path, res):
    if not path or len(path) < 2:
        return 0.0
    L = 0.0
    for i in range(1, len(path)):
        x0, y0 = path[i-1]
        x1, y1 = path[i]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx == 1 and dy == 1:
            L += math.sqrt(2) * res
        else:
            L += 1.0 * res
    return L

# -------------------------
# Supervisor setup
# -------------------------
sup = Supervisor()
timestep = int(sup.getBasicTimeStep())

receiver = sup.getDevice("receiver")
receiver.enable(timestep)
receiver.setChannel(SCOUT_CHANNEL)

emitter_scout = sup.getDevice("emitter")
emitter_scout.setChannel(SCOUT_CHANNEL)

# -------------------------
# Scout node lookup (for ground-truth translations) + dynamic HOME capture
# -------------------------
def _get_node_for_robot_name(robot_name):
    """Return the Webots node for a scout robot.

    Preference order:
      1) getFromDef() with common DEF naming variants
      2) scene-tree search: find a node that has a 'name' field equal to robot_name
         (works even if the robot has no DEF in the world file)
    """
    # 1) Try common DEF naming variants
    candidates = [
        robot_name,
        robot_name.upper(),
        robot_name.capitalize(),
        robot_name.replace("epuck", "EPUCK"),
        robot_name.replace("epuck", "EPUCK").upper(),
        "EPUCK1" if robot_name == "epuck1" else None,
        "EPUCK2" if robot_name == "epuck2" else None,
    ]
    for c in candidates:
        if not c:
            continue
        n = sup.getFromDef(c)
        if n is not None:
            return n

    # 2) Fallback: search the scene tree by the node's 'name' field
    root = sup.getRoot()
    if root is None:
        return None

    def _iter_nodes_dfs(start_node):
        stack = [start_node]
        while stack:
            node = stack.pop()
            yield node
            try:
                children_field = node.getField("children")
            except Exception:
                children_field = None
            if children_field is None:
                continue
            try:
                n_children = children_field.getCount()
            except Exception:
                n_children = 0
            for i in range(n_children):
                try:
                    ch = children_field.getMFNode(i)
                    if ch is not None:
                        stack.append(ch)
                except Exception:
                    pass

    for node in _iter_nodes_dfs(root):
        try:
            nf = node.getField("name")
            if nf is None:
                continue
            nm = nf.getSFString()
            if nm == robot_name:
                return node
        except Exception:
            continue

    return None

scout_nodes = {s: _get_node_for_robot_name(s) for s in SCOUTS}

def _scout_xy(robot_name):
    n = scout_nodes.get(robot_name)
    if n is None:
        return None
    f = n.getField("translation")
    if f is None:
        return None
    tr = f.getSFVec3f()
    if tr is None or len(tr) < 2:
        return None
    return (float(tr[0]), float(tr[1]))

# Capture each scout's starting translation as HOME (ground-truth)
for s in SCOUTS:
    xy = _scout_xy(s)
    if xy is not None:
        SCOUT_HOME[s] = (xy[0], xy[1])
print("[SUP] 🏠 saved scout HOME (from start translations):")
for s in SCOUTS:
    hx, hy = SCOUT_HOME[s]
    print(f"  - {s}: ({hx:.3f}, {hy:.3f})")


# optional second emitter for rescue (else reuse one)
try:
    emitter_rescue = sup.getDevice("emitter_k3")
    emitter_rescue.setChannel(RESCUE_CHANNEL)
except Exception:
    emitter_rescue = emitter_scout

# Keyboard for operator dispatch (supervisor-side)
keyboard = sup.getKeyboard()
keyboard.enable(timestep)

# -------------------------
# Mission-level mode selection
# -------------------------
MISSION_NONE = None
MISSION_GROUND = "G"
MISSION_AERIAL = "A"

mission_mode = MISSION_NONE
mission_prompt_printed = False

# Drone (Mavic2 Pro) supervisor-kinematic control in aerial mode
DRONE_DEF_CANDIDATES = ["MAVIC2PRO", "Mavic2Pro", "mavic2pro", "DRONE", "Drone", "MAVIC", "Mavic"]
drone_node = None
drone_translation_field = None
drone_rotation_field = None
drone_pose = None  # (x,y,z,yaw)
drone_freeze_active = False
drone_freeze_pose = None  # (x,y,z,yaw) pose to hold after aerial ends


# Control gains
DRONE_V_XY = 0.35      # m/s
DRONE_V_Z = 0.25       # m/s
DRONE_W_YAW = 1.8      # rad/s
def _get_drone_node():
    for d in DRONE_DEF_CANDIDATES:
        n = sup.getFromDef(d)
        if n is not None:
            return n
    # fallback: search by name field (if drone has name="mavic2pro" etc.)
    root = sup.getRoot()
    if root is None:
        return None
    def _iter_nodes_dfs(start_node):
        stack = [start_node]
        while stack:
            node = stack.pop()
            yield node
            try:
                children_field = node.getField("children")
            except Exception:
                children_field = None
            if children_field is None:
                continue
            try:
                n_children = children_field.getCount()
            except Exception:
                n_children = 0
            for i in range(n_children):
                try:
                    ch = children_field.getMFNode(i)
                    if ch is not None:
                        stack.append(ch)
                except Exception:
                    pass
    for node in _iter_nodes_dfs(root):
        try:
            nf = node.getField("name")
            if nf is None:
                continue
            nm = nf.getSFString()
            if nm in ("mavic2pro", "Mavic2Pro", "drone", "Drone"):
                return node
        except Exception:
            continue
    return None

def _init_drone_fields():
    global drone_node, drone_translation_field, drone_rotation_field, drone_pose
    if drone_node is None:
        drone_node = _get_drone_node()
    if drone_node is None:
        return False
    try:
        drone_translation_field = drone_node.getField("translation")
        drone_rotation_field = drone_node.getField("rotation")
    except Exception:
        return False
    try:
        tr = drone_translation_field.getSFVec3f()
        rot = drone_rotation_field.getSFRotation()
        yaw = float(rot[3]) if rot and len(rot) >= 4 else 0.0
        drone_pose = (float(tr[0]), float(tr[1]), float(tr[2]), yaw)
    except Exception:
        drone_pose = (0.0, 0.0, 1.0, 0.0)
    return True

def _set_drone_pose(x, y, z, yaw):
    global drone_pose
    if drone_translation_field is None or drone_rotation_field is None:
        return
    # keep within arena-ish bounds (optional safety clamp)
    x = max(-0.98, min(0.98, float(x)))
    y = max(-0.98, min(0.98, float(y)))
    z = max(0.20, min(2.50, float(z)))
    drone_translation_field.setSFVec3f([x, y, z])
    drone_rotation_field.setSFRotation([0.0, 0.0, 1.0, float(yaw)])
    try:
        # Prevent physics forces/velocities from accumulating (keeps drone kinematic & stable)
        if drone_node is not None:
            drone_node.resetPhysics()
    except Exception:
        pass
    drone_pose = (x, y, z, float(yaw))

def _broadcast_mission(mode):
    # Send to BOTH scouts and drone controller (all listen on channel 1).
    emitter_scout.setChannel(SCOUT_CHANNEL)
    emitter_scout.send(json.dumps({"type": "MISSION", "mode": mode}))
    print(f"[SUP] ✅ mission selected: {mode} (G=ground, A=aerial)")


print("[SUP] supervisor started")

robots = {}     # per-scout state
claims = []     # global claims
coverage_visited = [False] * len(COVERAGE_POINTS)
last_coverage_progress_time = 0.0
last_assign_time = -1e9
boot_sent = False

# Victim DB (from scouts): also keep a name-index to prevent duplicates
victims = []          # list of dicts
victims_by_name = {}  # victimName -> victim dict
next_victim_id = 1
last_new_victim_time = -1e9

# Track 0-frontier duration
zero_since = {s: None for s in SCOUTS}

# Scouting & rescue state
home_command_sent = False
scout_parked = {s: False for s in SCOUTS}
rescue_started = False
receiver_on_rescue_channel = False
# Stage-3 fix: freeze scouts, then send supervisor-map RETURN_HOME
freeze_sent = False
scout_frozen = {s: False for s in SCOUTS}
return_home_sent = False
scout_freeze_xy = {s: None for s in SCOUTS}
cached_occ_payload = None


# Dispatch mode state
dispatch_mode = False
helped_victims = set()     # {1,2,3,...}
rescuer_pose = None        # (x,y,th)
rescuer_home = None        # (x,y)
last_occ_grid = None       # bool grid
gt_victims = []            # [{"id":1,"def":"VICTIM_1","xy":(x,y),"xyz":(x,y,z)}...]

# confirmation state machine
pending_action = None      # dict like {"kind":"VICTIM","id":2,"xy":(x,y)}
awaiting_helped_confirm = False
awaiting_dispatch_confirm = False

def _print_dispatch_menu():
    ids = [v["id"] for v in gt_victims]
    ids_str = ",".join(str(i) for i in ids) if ids else "(none)"
    helped_str = ",".join(str(i) for i in sorted(helped_victims)) if helped_victims else "(none)"
    print(f"[SUP] DISPATCH MODE | victims: {ids_str} | helped: {helped_str}")
    print("[SUP] Type victim number (1..N) to dispatch, or H for home.")

def _send_rescuer(msg):
    emitter_rescue.setChannel(RESCUE_CHANNEL)
    emitter_rescue.send(json.dumps(msg))

def _preview_and_prompt(kind, vid, xy):
    """Print preview: coords, helped status, path length, ETA, then ask Dispatch? (Y/N)."""
    global awaiting_dispatch_confirm, pending_action

    helped = (vid in helped_victims) if (kind == "VICTIM" and vid is not None) else False
    helped_txt = "YES" if helped else "NO"

    if rescuer_pose is None or last_occ_grid is None:
        print("[SUP] Preview unavailable (missing rescuer pose or occupancy grid).")
        awaiting_dispatch_confirm = True
        pending_action = {"kind": kind, "id": vid, "xy": xy}
        print("[SUP] Dispatch anyway? (Y/N)")
        return

    sx, sy = world_to_grid(rescuer_pose[0], rescuer_pose[1])
    gx, gy = world_to_grid(xy[0], xy[1])

    path = astar(last_occ_grid, W, H, (sx, sy), (gx, gy))
    if not path:
        print(f"[SUP] Preview: {kind}_{vid if vid is not None else ''} at ({xy[0]:.2f},{xy[1]:.2f}) | helped={helped_txt} | NO PATH")
        awaiting_dispatch_confirm = True
        pending_action = {"kind": kind, "id": vid, "xy": xy}
        print("[SUP] Dispatch anyway? (Y/N)")
        return

    L = path_length_m(path, RES)
    eta = (L / NOMINAL_RESCUER_SPEED_MPS) if NOMINAL_RESCUER_SPEED_MPS > 1e-6 else 0.0
    label = f"VICTIM_{vid}" if kind == "VICTIM" else "HOME"
    print(f"[SUP] Preview: {label} at ({xy[0]:.2f},{xy[1]:.2f}) | helped={helped_txt} | path≈{L:.2f} m | ETA≈{eta:.0f} s")
    awaiting_dispatch_confirm = True
    pending_action = {"kind": kind, "id": vid, "xy": xy}
    print(f"[SUP] Dispatch to {label}? (Y/N)")

# -------------------------
# Claim helpers
# -------------------------
def clean_claims(t):
    global claims
    claims = [c for c in claims if c["until"] > t]

def is_claimed(p, t, requester):
    r2 = CLAIM_RADIUS * CLAIM_RADIUS
    for c in claims:
        if c["until"] <= t:
            continue
        if c["owner"] == requester:
            continue
        if dist2(p, c["p"]) <= r2:
            return True
    return False

def add_claim(p, t, owner):
    claims.append({"p": p, "until": t + CLAIM_TIME, "owner": owner})

def taboo_ok(rs, p, t):
    if rs["last_target"] is None:
        return True
    if t >= rs["taboo_until"]:
        return True
    return dist(p, rs["last_target"]) > TABOO_RADIUS

def reached(rs):
    if rs["current_target"] is None or rs.get("pose") is None:
        return False
    px, py, _ = rs["pose"]
    tx, ty = rs["current_target"]
    return dist((px, py), (tx, ty)) <= REACH_DIST

def watchdog_trigger(rs, t):
    if rs["current_target"] is None or rs.get("pose") is None:
        return False
    if rs["assigned_time"] <= 0.0:
        return False
    if t - rs["assigned_time"] < WATCHDOG_TIME:
        return False
    px, py, _ = rs["pose"]
    tx, ty = rs["current_target"]
    d_now = dist((px, py), (tx, ty))
    d0 = rs.get("assigned_d0", None)
    if d0 is None:
        return False
    return (d0 - d_now) < WATCHDOG_MIN_PROGRESS

# -------------------------
# Coverage helpers
# -------------------------
def mark_coverage_progress(t):
    global last_coverage_progress_time
    last_coverage_progress_time = t

def update_coverage_from_poses(t):
    changed = False
    for i, (cx, cy) in enumerate(COVERAGE_POINTS):
        if coverage_visited[i]:
            continue
        for rs in robots.values():
            if rs.get("pose") is None:
                continue
            px, py, _ = rs["pose"]
            if dist((px, py), (cx, cy)) <= COVER_REACH:
                coverage_visited[i] = True
                changed = True
                break
    if changed:
        mark_coverage_progress(t)

def choose_unvisited_coverage_point_for(name):
    rs = robots[name]
    px, py, _ = rs["pose"]
    candidates = []
    for i, (cx, cy) in enumerate(COVERAGE_POINTS):
        if coverage_visited[i]:
            continue
        candidates.append((dist((px, py), (cx, cy)), (cx, cy)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

# -------------------------
# Target selection (scouts)
# -------------------------
def pick_target_for_robot(name, t, force_coverage=False):
    rs = robots[name]
    pose = rs.get("pose", None)
    if pose is None:
        return None

    if force_coverage:
        return choose_unvisited_coverage_point_for(name)

    frs = rs.get("frontiers", [])
    if not frs:
        return None

    def filtered(frontiers, use_taboo=True, use_claims=True):
        out = []
        for f in frontiers:
            p = f["p"]  # (x,y)
            if use_taboo and not taboo_ok(rs, p, t):
                continue
            if use_claims and is_claimed(p, t, name):
                continue
            out.append(f)
        return out

    cand = filtered(frs, use_taboo=True, use_claims=True)
    if not cand and RELAX_TABOO_IF_EMPTY:
        cand = filtered(frs, use_taboo=False, use_claims=True)
    if not cand and RELAX_CLAIMS_IF_EMPTY:
        cand = filtered(frs, use_taboo=False, use_claims=False)

    if not cand:
        return None

    if FORCE_FARTHEST_ON_WATCHDOG and rs.get("force_farthest", False):
        rs["force_farthest"] = False
        px, py, _ = rs["pose"]
        far = max(cand, key=lambda f: dist((px, py), f["p"]))
        return far["p"]

    best = max(cand, key=lambda f: (f["i"] - f["c"]))
    return best["p"]

def send_target(name, p, t, tag=None):
    rs = robots[name]
    rs["current_target"] = p
    rs["last_target"] = p
    rs["taboo_until"] = t + TABOO_TIME
    rs["assigned_time"] = t

    px, py, _ = rs["pose"]
    rs["assigned_d0"] = dist((px, py), p)

    add_claim(p, t, owner=name)

    msg = {"type": "TARGET", "robot": name, "p": [p[0], p[1]]}
    if tag:
        msg["tag"] = tag

    emitter_scout.setChannel(SCOUT_CHANNEL)
    emitter_scout.send(json.dumps(msg))
    print(f"[SUP] TARGET -> {name}: ({p[0]:.2f}, {p[1]:.2f})" + (f" [{tag}]" if tag else ""))

# -------------------------
# Victim ground-truth
# -------------------------
def get_victim_translation(def_name):
    """Return victim WORLD translation (x,y,z) for DEF.

    Important: The 'translation' field shown in the scene tree can be *local* to a parent Transform.
    For navigation we need world coordinates, so we prefer node.getPosition() on a Pose-derived node.
    If DEF is attached to a non-Pose node (e.g., Shape), we walk up to the nearest Pose ancestor.
    """
    node = sup.getFromDef(def_name)
    if node is None:
        return None

    # Walk up to nearest Pose-derived ancestor (Solid/Robot/Transform/Pose)
    pose_node = node
    for _ in range(32):
        try:
            tname = pose_node.getTypeName()
        except Exception:
            tname = ""
        if tname in ("Solid", "Robot", "Transform", "Pose"):
            break
        try:
            parent = pose_node.getParentNode()
        except Exception:
            parent = None
        if parent is None:
            break
        pose_node = parent

    # Prefer WORLD position
    try:
        pos = pose_node.getPosition()
        if pos is not None and len(pos) >= 3:
            return (float(pos[0]), float(pos[1]), float(pos[2]))
    except Exception:
        pass

    # Fallback: local translation field (may be parent-relative)
    try:
        f = pose_node.getField("translation")
        if f is None:
            return None
        tr = f.getSFVec3f()
        if tr is None or len(tr) < 3:
            return None
        return (float(tr[0]), float(tr[1]), float(tr[2]))
    except Exception:
        return None


# -------------------------
# Victim storage (from scouts) - de-dupe by victimName first
# -------------------------
def _nearest_victim(p):
    if not victims:
        return None, float("inf")
    best_i = None
    best_d = float("inf")
    for i, v in enumerate(victims):
        d = dist(p, v["p"])
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d

def store_victim(candidate_p, t, by_robot, meta=None):
    global next_victim_id, last_new_victim_time

    if len(victims) >= VICTIM_MAX_STORE:
        return

    victim_name = None
    if isinstance(meta, dict):
        vn = meta.get("victimName", None)
        if isinstance(vn, str) and vn.startswith("VICTIM_"):
            victim_name = vn

    # primary de-dupe key
    if victim_name and victim_name in victims_by_name:
        v = victims_by_name[victim_name]
        k = v["count"]
        nx = (v["p"][0] * k + candidate_p[0]) / (k + 1)
        ny = (v["p"][1] * k + candidate_p[1]) / (k + 1)
        v["p"] = (nx, ny)
        v["last_seen"] = t
        v["count"] = k + 1
        v["by"] = by_robot
        return

    idx, d = _nearest_victim(candidate_p)
    if idx is not None and d <= VICTIM_MERGE_DIST:
        v = victims[idx]
        k = v["count"]
        nx = (v["p"][0] * k + candidate_p[0]) / (k + 1)
        ny = (v["p"][1] * k + candidate_p[1]) / (k + 1)
        v["p"] = (nx, ny)
        v["last_seen"] = t
        v["count"] = k + 1
        v["by"] = by_robot
        if victim_name:
            v["meta"]["victimName"] = victim_name
            victims_by_name[victim_name] = v
        return

    if (t - last_new_victim_time) < NEW_VICTIM_COOLDOWN:
        return

    vid = next_victim_id
    next_victim_id += 1
    entry = {
        "id": vid,
        "p": (candidate_p[0], candidate_p[1]),
        "first_seen": t,
        "last_seen": t,
        "by": by_robot,
        "count": 1,
        "meta": meta or {},
    }
    victims.append(entry)
    last_new_victim_time = t

    if victim_name:
        victims_by_name[victim_name] = entry

    print(f"[SUP] ✅ NEW VICTIM stored #{vid} at ({candidate_p[0]:.2f},{candidate_p[1]:.2f}) by {by_robot}")

    # Print GT translation immediately after NEW victim
    if victim_name:
        tr = get_victim_translation(victim_name)
        if tr:
            print(f"[SUP] {victim_name} translation: x=[{tr[0]:.3f}], y=[{tr[1]:.3f}], z=[{tr[2]:.3f}]")

# -------------------------
# Scouting completion detection
# -------------------------
def scouts_zero_for_hold(t, hold_s=SCOUT_DONE_HOLD_S):
    for n in SCOUTS:
        zt = zero_since.get(n, None)
        if zt is None:
            return False
        if (t - zt) < hold_s:
            return False
    return True

def scouting_done_now(t):
    ok = scouts_zero_for_hold(t, hold_s=SCOUT_DONE_HOLD_S)
    if REQUIRE_EXPECTED_VICTIMS:
        ok = ok and (len(victims_by_name) >= EXPECTED_VICTIMS)
    return ok

# -------------------------
# Occupancy grid from obstacle Box boundingObjects (X,Y)
# (unchanged from your working version)
# -------------------------
def _point_in_convex_quad(px, py, quad_xy):
    inside = False
    n = 4
    for i in range(n):
        x1, y1 = quad_xy[i]
        x2, y2 = quad_xy[(i + 1) % n]
        if ((y1 > py) != (y2 > py)):
            xinters = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1
            if px < xinters:
                inside = not inside
    return inside


def _collect_solids_with_box_bounding():
    """Collect obstacle rectangles in (x,y) from:
    - Solid with boundingObject Box (internal obstacles, like your DEF b1..b5)
    - SolidBox nodes (RectangleArena walls are SolidBox PROTOs)
    Returned list items contain:
      pos (x,y), R (2x2), half (hx,hy), def, name
    """
    root = sup.getRoot()
    if root is None:
        return []
    children_field = root.getField("children")
    solids = []

    def _find_first_box_node(n, depth=0, max_depth=12):
        if n is None or depth > max_depth:
            return None
        try:
            if n.getTypeName() == "Box":
                return n
        except Exception:
            return None

        # Shape -> geometry
        try:
            geom_f = n.getField("geometry")
            if geom_f is not None:
                g = geom_f.getSFNode()
                b = _find_first_box_node(g, depth + 1, max_depth)
                if b is not None:
                    return b
        except Exception:
            pass

        # Transform/Group/... -> children
        try:
            chf = n.getField("children")
            if chf is not None:
                for i in range(chf.getCount()):
                    b = _find_first_box_node(chf.getMFNode(i), depth + 1, max_depth)
                    if b is not None:
                        return b
        except Exception:
            pass

        # Nested boundingObject (safe)
        try:
            bof = n.getField("boundingObject")
            if bof is not None:
                b = _find_first_box_node(bof.getSFNode(), depth + 1, max_depth)
                if b is not None:
                    return b
        except Exception:
            pass

        return None

    # DFS over scene tree
    stack = []
    if children_field is not None:
        for i in range(children_field.getCount()):
            stack.append(children_field.getMFNode(i))

    while stack:
        node = stack.pop()
        if node is None:
            continue

        # recurse children
        try:
            chf = node.getField("children")
            if chf is not None:
                for i in range(chf.getCount()):
                    stack.append(chf.getMFNode(i))
        except Exception:
            pass

        try:
            tname = node.getTypeName()
        except Exception:
            continue

        try:
            dname = node.getDef()
        except Exception:
            dname = ""

        # exclude victims
        if isinstance(dname, str) and dname.startswith("VICTIM_"):
            continue

        # exclude robots
        node_name = None
        try:
            nf = node.getField("name")
            if nf is not None:
                node_name = nf.getSFString()
                if node_name in SCOUTS or ("Khepera" in node_name):
                    continue
        except Exception:
            pass

        # Case A: SolidBox walls
        if tname == "SolidBox":
            try:
                sf = node.getField("size")
                if sf is None:
                    continue
                size = sf.getSFVec3f()
                if size is None or len(size) < 3:
                    continue
                pos = node.getPosition()
                ori = node.getOrientation()
                if pos is None or ori is None or len(ori) < 9:
                    continue
                R00, R01 = float(ori[0]), float(ori[1])
                R10, R11 = float(ori[3]), float(ori[4])
                solids.append({
                    "def": dname,
                    "name": node_name,
                    "pos": (float(pos[0]), float(pos[1])),
                    "R": (R00, R01, R10, R11),
                    "half": (0.5 * float(size[0]), 0.5 * float(size[1])),
                })
            except Exception:
                continue
            continue

        # Case B: Solid obstacles with boundingObject Box
        if tname != "Solid":
            continue

        try:
            bof = node.getField("boundingObject")
        except Exception:
            bof = None
        if bof is None:
            continue

        try:
            bo_root = bof.getSFNode()
        except Exception:
            bo_root = None
        if bo_root is None:
            continue

        box_node = _find_first_box_node(bo_root)
        if box_node is None:
            continue

        sf = box_node.getField("size")
        if sf is None:
            continue
        size = sf.getSFVec3f()
        if size is None or len(size) < 3:
            continue

        try:
            pos = node.getPosition()
            ori = node.getOrientation()
        except Exception:
            continue
        if pos is None or ori is None or len(ori) < 9:
            continue

        R00, R01 = float(ori[0]), float(ori[1])
        R10, R11 = float(ori[3]), float(ori[4])

        solids.append({
            "def": dname,
            "name": node_name,
            "pos": (float(pos[0]), float(pos[1])),
            "R": (R00, R01, R10, R11),
            "half": (0.5 * float(size[0]), 0.5 * float(size[1])),
        })

    return solids


def build_occupancy_grid():
    """Build a conservative occupancy grid:
    mark a cell occupied if the obstacle rectangle overlaps that cell (not just center-in-rect).
    This prevents small 0.1m boxes (b1..b5) from disappearing at RES=0.05.
    """
    grid = [[0] * W for _ in range(H)]
    obstacles = _collect_solids_with_box_bounding()

    # DEBUG: always print what b* obstacles are included (or not)
    try:
        b_defs = sorted(set(o["def"] for o in obstacles if isinstance(o.get("def",""), str) and o["def"].startswith("b")))
        if b_defs:
            print("[SUP] ✅ occ: collected b* obstacles:", ", ".join(b_defs))
        else:
            print("[SUP] ⚠️ occ: NO b* obstacles collected (expected b1..b5).")
    except Exception:
        pass

    inflate = INFLATE_M  # v10: avoid double inflation; rely on dilation for footprint
    half_cell = 0.5 * RES

    for obs in obstacles:
        cx, cy = obs["pos"]
        R00, R01, R10, R11 = obs["R"]
        hx, hy = obs["half"]
        hx = float(hx) + inflate
        hy = float(hy) + inflate

        # Expand by half a cell -> conservative overlap rasterization
        hx_exp = hx + half_cell
        hy_exp = hy + half_cell

        # Bounding iteration region (world corners -> grid AABB)
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                lx = sx * hx_exp
                ly = sy * hy_exp
                wx = cx + (R00 * lx + R01 * ly)
                wy = cy + (R10 * lx + R11 * ly)
                corners.append((wx, wy))

        min_x = min(p[0] for p in corners)
        max_x = max(p[0] for p in corners)
        min_y = min(p[1] for p in corners)
        max_y = max(p[1] for p in corners)

        ix0 = int(math.floor((min_x - ORIGIN_X) / RES))
        ix1 = int(math.ceil((max_x - ORIGIN_X) / RES))
        iy0 = int(math.floor((min_y - ORIGIN_Y) / RES))
        iy1 = int(math.ceil((max_y - ORIGIN_Y) / RES))

        ix0 = max(0, min(W - 1, ix0))
        ix1 = max(0, min(W - 1, ix1))
        iy0 = max(0, min(H - 1, iy0))
        iy1 = max(0, min(H - 1, iy1))

        # local = R^T * (world - center)
        for iy in range(iy0, iy1 + 1):
            wy = ORIGIN_Y + (iy + 0.5) * RES
            dy = wy - cy
            for ix in range(ix0, ix1 + 1):
                wx = ORIGIN_X + (ix + 0.5) * RES
                dx = wx - cx

                local_x = R00 * dx + R10 * dy
                local_y = R01 * dx + R11 * dy

                if abs(local_x) <= hx_exp and abs(local_y) <= hy_exp:
                    grid[iy][ix] = 1


    # --- patched v9: dilate occupied cells by robot footprint ---
    try:
        dilate_cells = max(1, int(math.floor((ROBOT_RADIUS_M + EXTRA_CLEARANCE_M) / RES)))  # v11: less aggressive dilation
        if dilate_cells > 0:
            h = len(grid)
            w = len(grid[0]) if h else 0
            # copy
            base = [row[:] for row in grid]
            for y in range(h):
                for x in range(w):
                    if base[y][x] != 1:
                        continue
                    for dy in range(-dilate_cells, dilate_cells + 1):
                        yy = y + dy
                        if yy < 0 or yy >= h:
                            continue
                        # tighten to circle for fewer fills
                        dx_max = int(math.floor((dilate_cells * dilate_cells - dy * dy) ** 0.5))
                        for dx in range(-dx_max, dx_max + 1):
                            xx = x + dx
                            if 0 <= xx < w:
                                grid[yy][xx] = 1
    except Exception:
        pass

    try:
        occ_count = sum(sum(1 for c in row if c) for row in grid)
        print(f"[SUP] occ occupied cells (after dilation) = {occ_count}")
    except Exception:
        pass

    return grid

def pack_grid(grid):
    """Pack occupancy grid (list of list of ints) into a compact string."""
    return ''.join(''.join('1' if c else '0' for c in row) for row in grid)



def unpack_occ(s, w, h):
    """Unpack packed occupancy string back to 2D int grid."""
    if s is None:
        return [[0]*w for _ in range(h)]
    s = str(s)
    if len(s) < w*h:
        # tolerate truncation by padding with zeros
        s = s.ljust(w*h, '0')
    grid = []
    idx = 0
    for _ in range(h):
        row = [1 if s[idx + j] == '1' else 0 for j in range(w)]
        grid.append(row)
        idx += w

    # --- patched v9: dilate occupied cells by robot footprint ---
    try:
        dilate_cells = int(math.ceil((ROBOT_RADIUS_M + EXTRA_CLEARANCE_M) / RES))
        if dilate_cells > 0:
            h = len(grid)
            w = len(grid[0]) if h else 0
            # copy
            base = [row[:] for row in grid]
            for y in range(h):
                for x in range(w):
                    if base[y][x] != 1:
                        continue
                    for dy in range(-dilate_cells, dilate_cells + 1):
                        yy = y + dy
                        if yy < 0 or yy >= h:
                            continue
                        # tighten to circle for fewer fills
                        dx_max = int(math.floor((dilate_cells * dilate_cells - dy * dy) ** 0.5))
                        for dx in range(-dx_max, dx_max + 1):
                            xx = x + dx
                            if 0 <= xx < w:
                                grid[yy][xx] = 1
    except Exception:
        pass

    try:
        occ_count = sum(sum(1 for c in row if c) for row in grid)
        print(f"[SUP] occ occupied cells (after dilation) = {occ_count}")
    except Exception:
        pass

    return grid



# -------------------------
# Occupancy grid patch v7:
# - Ensures small 0.1m boxes (e.g., DEF b1..b5) are rasterized conservatively
#   so they don't disappear at RES=0.05.
# - Adds a one-time debug print of which b* obstacles were collected.
# -------------------------
def _collect_solids_with_box_bounding():
    """Collect obstacle rectangles in (x,y) from:
    - Solid with boundingObject Box (internal obstacles, like your b1..b5)
    - SolidBox nodes (RectangleArena walls are SolidBox PROTOs)
    Returned list items contain:
      pos (x,y), R (2x2), half (hx,hy), def, name
    """
    root = sup.getRoot()
    if root is None:
        return []
    children_field = root.getField("children")
    solids = []

    def _find_first_box_node(n, depth=0, max_depth=12):
        if n is None or depth > max_depth:
            return None
        try:
            if n.getTypeName() == "Box":
                return n
        except Exception:
            return None

        # Shape -> geometry
        try:
            geom_f = n.getField("geometry")
            if geom_f is not None:
                g = geom_f.getSFNode()
                b = _find_first_box_node(g, depth + 1, max_depth)
                if b is not None:
                    return b
        except Exception:
            pass

        # Transform/Group/... -> children
        try:
            chf = n.getField("children")
            if chf is not None:
                for i in range(chf.getCount()):
                    b = _find_first_box_node(chf.getMFNode(i), depth + 1, max_depth)
                    if b is not None:
                        return b
        except Exception:
            pass

        # Nested boundingObject (safe)
        try:
            bof = n.getField("boundingObject")
            if bof is not None:
                b = _find_first_box_node(bof.getSFNode(), depth + 1, max_depth)
                if b is not None:
                    return b
        except Exception:
            pass

        return None

    # DFS over scene tree
    stack = []
    if children_field is not None:
        for i in range(children_field.getCount()):
            stack.append(children_field.getMFNode(i))

    while stack:
        node = stack.pop()
        if node is None:
            continue

        # recurse children
        try:
            chf = node.getField("children")
            if chf is not None:
                for i in range(chf.getCount()):
                    stack.append(chf.getMFNode(i))
        except Exception:
            pass

        # type name
        try:
            tname = node.getTypeName()
        except Exception:
            continue

        # DEF name (if any)
        try:
            dname = node.getDef()
        except Exception:
            dname = ""

        # exclude victims by DEF
        if isinstance(dname, str) and dname.startswith("VICTIM_"):
            continue

        # exclude robots by name field if present
        node_name = None
        try:
            name_field = node.getField("name")
            if name_field is not None:
                node_name = name_field.getSFString()
                if node_name in SCOUTS or ("Khepera" in node_name):
                    continue
        except Exception:
            pass

        # Case A: SolidBox (arena walls)
        if tname == "SolidBox":
            try:
                sf = node.getField("size")
                if sf is None:
                    continue
                size = sf.getSFVec3f()
                if size is None or len(size) < 3:
                    continue

                pos = node.getPosition()
                ori = node.getOrientation()
                if pos is None or ori is None or len(ori) < 9:
                    continue

                R00, R01 = float(ori[0]), float(ori[1])
                R10, R11 = float(ori[3]), float(ori[4])

                solids.append({
                    "def": dname,
                    "name": node_name,
                    "pos": (float(pos[0]), float(pos[1])),
                    "R": (R00, R01, R10, R11),
                    "half": (0.5 * float(size[0]), 0.5 * float(size[1])),
                })
            except Exception:
                continue
            continue

        # Case B: Solid + boundingObject Box (obstacles)
        if tname != "Solid":
            continue

        try:
            bof = node.getField("boundingObject")
        except Exception:
            bof = None
        if bof is None:
            continue

        try:
            bo_root = bof.getSFNode()
        except Exception:
            bo_root = None
        if bo_root is None:
            continue

        box_node = _find_first_box_node(bo_root)
        if box_node is None:
            continue

        sf = box_node.getField("size")
        if sf is None:
            continue
        size = sf.getSFVec3f()
        if size is None or len(size) < 3:
            continue

        try:
            pos = node.getPosition()
            ori = node.getOrientation()
        except Exception:
            continue
        if pos is None or ori is None or len(ori) < 9:
            continue

        R00, R01 = float(ori[0]), float(ori[1])
        R10, R11 = float(ori[3]), float(ori[4])

        solids.append({
            "def": dname,
            "name": node_name,
            "pos": (float(pos[0]), float(pos[1])),
            "R": (R00, R01, R10, R11),
            "half": (0.5 * float(size[0]), 0.5 * float(size[1])),
        })

    return solids


def build_occupancy_grid():
    """Build a conservative occupancy grid:
    mark a cell occupied if the obstacle rectangle overlaps that cell (not just center-in-rect).
    """
    grid = [[0] * W for _ in range(H)]
    obstacles = _collect_solids_with_box_bounding()

    # One-time debug: confirm b* obstacles are collected
    try:
        b_defs = sorted(set(o["def"] for o in obstacles if isinstance(o.get("def",""), str) and o["def"].startswith("b")))
        if b_defs:
            print("[SUP] ✅ occ: collected b* obstacles:", ", ".join(b_defs))
        else:
            print("[SUP] ⚠️ occ: NO b* obstacles collected (expected b1..b5).")
    except Exception:
        pass

    inflate = INFLATE_M
    half_cell = 0.5 * RES

    for obs in obstacles:
        cx, cy = obs["pos"]
        R00, R01, R10, R11 = obs["R"]
        hx, hy = obs["half"]
        hx = float(hx) + inflate
        hy = float(hy) + inflate

        # Expand by half a cell -> conservative overlap rasterization
        hx_exp = hx + half_cell
        hy_exp = hy + half_cell

        # Bounding iteration region (world corners -> grid AABB)
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                lx = sx * hx_exp
                ly = sy * hy_exp
                wx = cx + (R00 * lx + R01 * ly)
                wy = cy + (R10 * lx + R11 * ly)
                corners.append((wx, wy))

        min_x = min(p[0] for p in corners)
        max_x = max(p[0] for p in corners)
        min_y = min(p[1] for p in corners)
        max_y = max(p[1] for p in corners)

        ix0 = int(math.floor((min_x - ORIGIN_X) / RES))
        ix1 = int(math.ceil((max_x - ORIGIN_X) / RES))
        iy0 = int(math.floor((min_y - ORIGIN_Y) / RES))
        iy1 = int(math.ceil((max_y - ORIGIN_Y) / RES))

        ix0 = max(0, min(W - 1, ix0))
        ix1 = max(0, min(W - 1, ix1))
        iy0 = max(0, min(H - 1, iy0))
        iy1 = max(0, min(H - 1, iy1))

        # local = R^T * (world - center)
        for iy in range(iy0, iy1 + 1):
            wy = ORIGIN_Y + (iy + 0.5) * RES
            dy = wy - cy
            for ix in range(ix0, ix1 + 1):
                wx = ORIGIN_X + (ix + 0.5) * RES
                dx = wx - cx

                local_x = R00 * dx + R10 * dy
                local_y = R01 * dx + R11 * dy

                if abs(local_x) <= hx_exp and abs(local_y) <= hy_exp:
                    grid[iy][ix] = 1


    # --- patched v9: dilate occupied cells by robot footprint ---
    try:
        dilate_cells = int(math.ceil((ROBOT_RADIUS_M + EXTRA_CLEARANCE_M) / RES))
        if dilate_cells > 0:
            h = len(grid)
            w = len(grid[0]) if h else 0
            # copy
            base = [row[:] for row in grid]
            for y in range(h):
                for x in range(w):
                    if base[y][x] != 1:
                        continue
                    for dy in range(-dilate_cells, dilate_cells + 1):
                        yy = y + dy
                        if yy < 0 or yy >= h:
                            continue
                        # tighten to circle for fewer fills
                        dx_max = int(math.floor((dilate_cells * dilate_cells - dy * dy) ** 0.5))
                        for dx in range(-dx_max, dx_max + 1):
                            xx = x + dx
                            if 0 <= xx < w:
                                grid[yy][xx] = 1
    except Exception:
        pass

    try:
        occ_count = sum(sum(1 for c in row if c) for row in grid)
        print(f"[SUP] occ occupied cells (after dilation) = {occ_count}")
    except Exception:
        pass

    return grid




def maybe_start_rescue(t):
    global rescue_started, dispatch_mode, last_occ_grid, gt_victims, receiver_on_rescue_channel

    if rescue_started:
        return

    for s in SCOUTS:
        if not scout_parked.get(s, False):
            return

    gt = []
    for dname in VICTIM_DEFS:
        tr = get_victim_translation(dname)
        if tr is None:
            continue
        gt.append({"def": dname, "translation": [tr[0], tr[1], tr[2]]})

    # If ground-truth victims are missing/misnamed in the world file, don't block rescue forever.
    if REQUIRE_EXPECTED_VICTIMS and len(gt) < EXPECTED_VICTIMS:
        print(f"[SUP] ⚠️ GT victims found {len(gt)}/{EXPECTED_VICTIMS} via DEF; starting rescue anyway.")
        # Fallback: if we have scout-reported victims-by-name, include them as well (z=0).
        for vn, v in sorted(victims_by_name.items()):
            if not isinstance(vn, str) or not vn.startswith("VICTIM_"):
                continue
            if any(e["def"] == vn for e in gt):
                continue
            p = v.get("p", None)
            if p and len(p) >= 2:
                gt.append({"def": vn, "translation": [float(p[0]), float(p[1]), 0.0]})


    occ_int = build_occupancy_grid()
    # v10: force start/goal free so A* doesn't fail due to dilation touching them
    try:
        # Clear around rescuer home (start)
        if rescuer_home is not None:
            _clear_free_around(occ_int, rescuer_home[0], rescuer_home[1], r_cells=3)
        # Clear around every victim (goal)
        for v in gt:
            x, y, _z = v["translation"]
            _clear_free_around(occ_int, x, y, r_cells=3)
    except Exception:
        pass

    occ_str = pack_grid(occ_int)
    # Debug: verify walls/obstacles are included
    try:
        occ_count = sum(sum(1 for c in row if c) for row in occ_int)
        print(f"[SUP] occ occupied cells = {occ_count}")
    except Exception:
        pass

    payload = {
        "type": "RESCUE_START",
        "res": RES,
        "origin": [ORIGIN_X, ORIGIN_Y],
        "w": W,
        "h": H,
        "occ": occ_str,
        "victims": gt,
        "t": float(t),
    }

    emitter_rescue.setChannel(RESCUE_CHANNEL)
    emitter_rescue.send(json.dumps(payload))

    print("[SUP] 🚑 RESCUE_START -> Khepera3 (channel 2)")
    for v in gt:
        x, y, z = v["translation"]
        print(f"[SUP] victim {v['def']} translation: x=[{x:.3f}], y=[{y:.3f}], z=[{z:.3f}]")

    # Store for preview/dispatch
    last_occ_grid = unpack_occ(occ_str, W, H)
    gt_victims = []
    for v in gt:
        d = v["def"]
        try:
            vid = int(d.split("_")[-1])
        except Exception:
            continue
        x, y, z = v["translation"]
        gt_victims.append({"id": vid, "def": d, "xy": (float(x), float(y)), "xyz": (float(x), float(y), float(z))})
    gt_victims.sort(key=lambda e: e["id"])

    rescue_started = True
    dispatch_mode = True
    receiver_on_rescue_channel = False  # will switch in main loop

    # Ensure rescuer waits until operator dispatches
    _send_rescuer({"type": "CMD_WAIT"})
    print("[SUP] ✅ DISPATCH MODE enabled (operator selects victims).")
    _print_dispatch_menu()

# -------------------------
# Keyboard input helper
# -------------------------
def _consume_key():
    """Return single-char command or None."""
    k = keyboard.getKey()
    if k == -1:
        return None
    # Webots key codes for digits/letters match ASCII for basic keys in most cases
    try:
        ch = chr(k)
    except Exception:
        return None
    # normalize
    if ch.isalpha():
        ch = ch.upper()
    return ch

# -------------------------
# Main loop
# -------------------------
while sup.step(timestep) != -1:
    t = sup.getTime()
    clean_claims(t)

    # GLOBAL drone freeze: once aerial scouting ends, keep the drone pinned to the pose where E was pressed.
    if drone_freeze_active and drone_freeze_pose is not None:
        if drone_node is None or drone_translation_field is None or drone_rotation_field is None:
            _init_drone_fields()
        _set_drone_pose(drone_freeze_pose[0], drone_freeze_pose[1], drone_freeze_pose[2], drone_freeze_pose[3])

    # -------------------------
    # Mission selection (blocks mission flow until chosen)
    # -------------------------
    if mission_mode is MISSION_NONE:
        if not mission_prompt_printed:
            print("[SUP] Choose scouting mode: (G) Ground scouting | (A) Aerial scouting")
            print("[SUP] Press G or A to start. (In aerial: use W/S/A/D + ↑/↓ to move drone, E to end scouting)")
            mission_prompt_printed = True

        k = keyboard.getKey()
        if k != -1:
            try:
                ch = chr(k).upper()
            except Exception:
                ch = ""
            if ch in ("G", "A"):
                mission_mode = ch
                _broadcast_mission(mission_mode)
                if mission_mode == MISSION_AERIAL:
                    # In aerial mode, scouts are ignored for completion/parking.
                    scout_parked["epuck1"] = True
                    scout_parked["epuck2"] = True
                    home_command_sent = True  # allow rescue start once aerial ends
                    # initialize drone fields for kinematic control
                    if _init_drone_fields():
                        print("[SUP] 🛩️ AERIAL scouting enabled. Drone control active.")
                    else:
                        print("[SUP] ⚠️ AERIAL selected but drone node not found. Check DRONE DEF/name.")
                else:
                    print("[SUP] 🚗 GROUND scouting enabled. e-pucks will explore as before.")
        # stay idle until mode chosen
        continue


    # Receive messages:
    # - while scouting: channel 1 (scouts)
    # - after rescue starts: switch to channel 2 (rescuer)
    if rescue_started and (not receiver_on_rescue_channel):
        receiver.setChannel(RESCUE_CHANNEL)
        receiver_on_rescue_channel = True
        print("[SUP] receiver switched to RESCUE_CHANNEL (2)")

    while receiver.getQueueLength() > 0:
        raw = receiver.getString()
        receiver.nextPacket()
        try:
            data = json.loads(raw)
        except Exception:
            continue

        mtype = data.get("type", "")

        # -------------------
        # Scout channel messages
        # -------------------
        if (not rescue_started) and mtype == "BID":
            rname = data.get("robot", "unknown")
            pose = data.get("pose", None)
            frontiers = data.get("frontiers", [])

            if rname not in robots:
                robots[rname] = {
                    "pose": None,
                    "frontiers": [],
                    "frontier_count": 999,
                    "last_bid_time": 0.0,
                    "current_target": None,
                    "last_target": None,
                    "taboo_until": 0.0,
                    "assigned_time": 0.0,
                    "assigned_d0": None,
                    "force_farthest": False,
                }

            if pose and len(pose) >= 3:
                robots[rname]["pose"] = (float(pose[0]), float(pose[1]), float(pose[2]))

            parsed = []
            for f in frontiers:
                try:
                    p = (float(f["p"][0]), float(f["p"][1]))
                    c = float(f.get("c", 0.0))
                    i = float(f.get("i", 0.0))
                    parsed.append({"p": p, "c": c, "i": i})
                except Exception:
                    pass

            robots[rname]["frontiers"] = parsed
            robots[rname]["frontier_count"] = len(parsed)
            robots[rname]["last_bid_time"] = t

            if rname in zero_since:
                if robots[rname]["frontier_count"] == 0:
                    if zero_since[rname] is None:
                        zero_since[rname] = t
                else:
                    zero_since[rname] = None

        elif (not rescue_started) and mtype == "VICTIM_NAME":
            # From aerial drone: identity only (ground-truth translation is fetched here)
            vn = data.get("victim", None)
            if isinstance(vn, str) and vn.startswith("VICTIM_"):
                tr = get_victim_translation(vn)
                if tr is not None:
                    store_victim((float(tr[0]), float(tr[1])), t, data.get("robot", "mavic2pro"), meta={"victimName": vn})
                else:
                    # Still store by name with dummy position (won't affect rescue if GT later exists)
                    store_victim((0.0, 0.0), t, data.get("robot", "mavic2pro"), meta={"victimName": vn})

        elif (not rescue_started) and mtype == "VICTIM":
            rname = data.get("robot", "unknown")
            p = data.get("p", None)
            meta = data.get("meta", None)
            if p and len(p) >= 2:
                try:
                    vx = float(p[0])
                    vy = float(p[1])
                    store_victim((vx, vy), t, rname, meta=meta)
                except Exception:
                    pass

        
        elif (not rescue_started) and mtype == "SCOUT_FROZEN":
            rname = data.get("robot", "")
            if rname in scout_frozen and not scout_frozen[rname]:
                scout_frozen[rname] = True
                print(f"[SUP] 🧊 SCOUT_FROZEN from {rname}")

        elif (not rescue_started) and mtype == "SCOUT_PARKED":
            rname = data.get("robot", "")
            if rname in scout_parked and not scout_parked[rname]:
                scout_parked[rname] = True
                print(f"[SUP] 🏁 SCOUT_PARKED received from {rname}")

        # -------------------
        # Rescuer channel messages
        # -------------------
        elif rescue_started and mtype == "POSE":
            p = data.get("p", None)
            if p and len(p) >= 3:
                rescuer_pose = (float(p[0]), float(p[1]), float(p[2]))

        elif rescue_started and mtype == "RESCUER_HOME":
            p = data.get("p", None)
            if p and len(p) >= 2:
                rescuer_home = (float(p[0]), float(p[1]))
                print(f"[SUP] rescuer home set to ({rescuer_home[0]:.2f},{rescuer_home[1]:.2f})")

        elif rescue_started and mtype == "ARRIVED_VICTIM":
            vid = data.get("id", None)
            if vid is not None:
                vid = int(vid)
                helped_victims.add(vid)
                print(f"[SUP] ✅ ARRIVED: VICTIM_{vid}")
                _print_dispatch_menu()

        elif rescue_started and mtype == "ARRIVED_HOME":
            print("[SUP] ✅ ARRIVED: HOME")
            _print_dispatch_menu()


    # -------------------------
    # AERIAL SCOUTING control (supervisor-kinematic drone)
    # -------------------------
    if mission_mode == MISSION_AERIAL and (not rescue_started):
        # IMPORTANT:
        # Always re-apply the drone pose every step.
        # Otherwise the drone falls under physics when no keys are pressed, then "snaps" back
        # on the next key event (janky vanish/reappear behavior).
        if drone_node is None or drone_translation_field is None or drone_rotation_field is None or drone_pose is None:
            _init_drone_fields()

        if drone_pose is not None:
            x, y, z, yaw = drone_pose
            dt = float(timestep) / 1000.0

            # Consume all key events this step (Webots keyboard is event-based).
            k = keyboard.getKey()
            while k != -1:
                # letters (ASCII)
                if k in (ord('W'), ord('w')):
                    x += math.cos(yaw) * DRONE_V_XY * dt
                    y += math.sin(yaw) * DRONE_V_XY * dt
                elif k in (ord('S'), ord('s')):
                    x -= math.cos(yaw) * DRONE_V_XY * dt
                    y -= math.sin(yaw) * DRONE_V_XY * dt
                elif k in (ord('A'), ord('a')):
                    yaw += DRONE_W_YAW * dt
                elif k in (ord('D'), ord('d')):
                    yaw -= DRONE_W_YAW * dt
                elif k == Keyboard.UP:
                    z += DRONE_V_Z * dt
                elif k == Keyboard.DOWN:
                    z -= DRONE_V_Z * dt
                elif k in (ord('E'), ord('e')):
                    # Gate: do NOT allow ending aerial scouting until ALL victims are detected.
                    # (The drone/operator should not know how many victims exist, but the supervisor does.)
                    needed = len(VICTIM_DEFS) if VICTIM_DEFS else EXPECTED_VICTIMS
                    if len(victims_by_name) < needed:
                        print("[SUP] You didn't scout the entire area, there might be more victims to detect")
                    else:
                        print("[SUP] ✅ AERIAL scouting ended (E pressed). Stopping simulation.")
                        # Optional: freeze the drone at current pose (useful if you switch to PAUSE instead of QUIT).
                        drone_freeze_active = True
                        drone_freeze_pose = (x, y, z, yaw)
                        # Optional: tell the drone controller to stop emitting detections.
                        emitter_scout.setChannel(SCOUT_CHANNEL)
                        emitter_scout.send(json.dumps({"type": "AERIAL_FREEZE"}))
                        # Stop the entire simulation (PAUSE).
                        sup.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)  # paused instead of quit
                        # sup.simulationQuit(0)
                k = keyboard.getKey()

            # Update stored pose (even if no key pressed).
            drone_pose = (x, y, z, yaw)

            # Always enforce pose to keep drone kinematic/hovering (no falling).
            _set_drone_pose(x, y, z, yaw)

        # While aerial scouting is ongoing, skip ground-scout logic
        # (victim detections are coming from mavic2pro.py)
        continue
    # -------------------------
    # GROUND MODE: keep drone fully frozen (avoid it falling/crashing and destabilizing physics)
    # -------------------------
    if mission_mode == MISSION_GROUND and (not rescue_started):
        if drone_node is None or drone_translation_field is None or drone_rotation_field is None:
            _init_drone_fields()
        if drone_translation_field is not None and drone_rotation_field is not None and drone_pose is not None:
            # Re-apply the initial pose every step and reset physics to keep it inert
            _set_drone_pose(drone_pose[0], drone_pose[1], drone_pose[2], drone_pose[3])

# BOOT message
    if not boot_sent and len(robots) >= 2:
        print("[SUP] BOOT targets sent")
        boot_sent = True
        last_coverage_progress_time = t

    # Update coverage visited (optional)
    if robots and (not rescue_started):
        update_coverage_from_poses(t)

    stalled = (t - last_coverage_progress_time) > STALL_TIME

    # Reach + watchdog processing (scouts only)
    if not rescue_started:
        for rname, rs in robots.items():
            if rs.get("pose") is None:
                continue
            if reached(rs):
                rs["current_target"] = None
                rs["assigned_d0"] = None
            if watchdog_trigger(rs, t):
                if rs["current_target"] is not None:
                    rs["last_target"] = rs["current_target"]
                    rs["taboo_until"] = t + TABOO_TIME
                rs["current_target"] = None
                rs["assigned_d0"] = None
                rs["assigned_time"] = 0.0
                if FORCE_FARTHEST_ON_WATCHDOG:
                    rs["force_farthest"] = True

    # Scouting done => freeze scouts, read exact translations, then send supervisor-map RETURN_HOME
    if (not rescue_started) and scouting_done_now(t):
        # 1) Freeze-in-place once (so supervisor reads exact translation)
        if not freeze_sent:
            freeze_sent = True
            home_command_sent = True  # keep legacy flag to stop other transitions
            print("[SUP] ✅ scouting complete -> freezing scouts before RETURN_HOME")
            for s in SCOUTS:
                emitter_scout.setChannel(SCOUT_CHANNEL)
                emitter_scout.send(json.dumps({"type": "STOP", "robot": s}))

        # 2) After both scouts confirm frozen, snapshot their exact translations
        if freeze_sent and (not return_home_sent) and all(scout_frozen.values()):
            for s in SCOUTS:
                xy = _scout_xy(s)
                scout_freeze_xy[s] = xy
            print("[SUP] 📍 frozen translations:")
            for s in SCOUTS:
                xy = scout_freeze_xy[s]
                if xy is None:
                    print(f"  - {s}: (unknown)")
                else:
                    print(f"  - {s}: ({xy[0]:.3f}, {xy[1]:.3f})")

            # 3) Build supervisor occupancy map (scene-tree bounding boxes) and send to scouts
            occ_grid = build_occupancy_grid()
            occ_str = pack_grid(occ_grid)

            for s in SCOUTS:
                start_xy = scout_freeze_xy[s]
                home_xy = SCOUT_HOME[s]
                if start_xy is None:
                    # Fallback #1: use last supervisor-known pose from BID stream
                    rs = robots.get(s, {})
                    p = rs.get("pose", None)
                    if p is not None and len(p) >= 2:
                        start_xy = (float(p[0]), float(p[1]))
                    else:
                        # Fallback #2: as a last resort, use HOME (will make the scout instantly "park")
                        start_xy = home_xy

                payload = {
                    "type": "SCOUT_RETURN_HOME",
                    "robot": s,
                    "origin": [ORIGIN_X, ORIGIN_Y],
                    "res": RES,
                    "w": W,
                    "h": H,
                    "occ": occ_str,
                    "start": [start_xy[0], start_xy[1]],
                    "home": [home_xy[0], home_xy[1]],
                }
                emitter_scout.setChannel(SCOUT_CHANNEL)
                emitter_scout.send(json.dumps(payload))

            return_home_sent = True
            print("[SUP] 🗺️ RETURN_HOME map sent to scouts (channel 1)")
# Frontier assignment only while scouting
    if (not rescue_started) and (t - last_assign_time) >= ASSIGN_PERIOD and (not home_command_sent):
        last_assign_time = t
        for rname in sorted(robots.keys()):
            rs = robots[rname]
            if rs.get("pose") is None:
                continue
            if rs.get("current_target") is not None:
                continue
            p = pick_target_for_robot(rname, t, force_coverage=stalled)
            if p is None and stalled:
                p = pick_target_for_robot(rname, t, force_coverage=False)
            if p is not None:
                send_target(rname, p, t)

    # Start rescue once scouts parked
    if home_command_sent and (not rescue_started):
        if mission_mode != MISSION_AERIAL:
            maybe_start_rescue(t)

    # -------------------------
    # DISPATCH MODE: operator input
    # -------------------------
    if dispatch_mode and rescue_started:
        ch = _consume_key()
        if ch is None:
            continue

        # Confirmation handling
        if awaiting_helped_confirm:
            if ch == 'Y':
                awaiting_helped_confirm = False
                # proceed to preview
                _preview_and_prompt(pending_action["kind"], pending_action["id"], pending_action["xy"])
            elif ch == 'N':
                awaiting_helped_confirm = False
                pending_action = None
                print("[SUP] cancelled.")
                _print_dispatch_menu()
            continue

        if awaiting_dispatch_confirm:
            if ch == 'Y':
                awaiting_dispatch_confirm = False
                if pending_action is None:
                    _print_dispatch_menu()
                    continue
                kind = pending_action["kind"]
                vid = pending_action["id"]
                xy = pending_action["xy"]

                if kind == "VICTIM":
                    _send_rescuer({"type": "CMD_GO_VICTIM", "id": int(vid), "p": [xy[0], xy[1]]})
                    print(f"[SUP] 🚑 dispatched -> VICTIM_{vid}")
                else:
                    _send_rescuer({"type": "CMD_GO_HOME", "p": [xy[0], xy[1]]})
                    print("[SUP] 🏠 dispatched -> HOME")

                pending_action = None
            elif ch == 'N':
                awaiting_dispatch_confirm = False
                pending_action = None
                print("[SUP] cancelled.")
                _print_dispatch_menu()
            continue

        # Normal commands (no pending confirmations)
        if ch == 'H':
            if rescuer_home is None:
                print("[SUP] rescuer home not known yet (waiting for RESCUER_HOME).")
                continue
            if rescuer_pose is not None and dist((rescuer_pose[0], rescuer_pose[1]), rescuer_home) <= HOME_REACH_M:
                print("[SUP] Rescuer is already at home.")
                _print_dispatch_menu()
                continue
            _preview_and_prompt("HOME", None, rescuer_home)
            continue

        # digits: victim selection
        if ch.isdigit():
            vid = int(ch)
            found = None
            for v in gt_victims:
                if v["id"] == vid:
                    found = v
                    break
            if found is None:
                print(f"[SUP] Unknown victim number: {vid}")
                _print_dispatch_menu()
                continue

            xy = found["xy"]
            pending_action = {"kind": "VICTIM", "id": vid, "xy": xy}

            # already-helped prompt first (as agreed)
            if vid in helped_victims:
                awaiting_helped_confirm = True
                print(f"[SUP] You have already helped VICTIM_{vid}. Go there again? (Y/N)")
                continue

            # then preview + dispatch confirm
            _preview_and_prompt("VICTIM", vid, xy)
            continue

        # Anything else: ignore but remind menu
        _print_dispatch_menu()