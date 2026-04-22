# gaze_core.py
import os, math, time, threading, collections
import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Backward compat stubs
EM_MODEL_NAME  = "enet_b0_8_best_vgaf"
EM_ENGINE      = "onnx"
EM_CROP_MARGIN = 0.22
EM_FEED_EVERY  = 1
EM_KALMAN_Q    = 0.015
EM_KALMAN_R    = 0.08
EM_EMA_ALPHA   = 0.40
EM_DEBOUNCE    = 4

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
CALIB_FILE = os.path.join(BASE_DIR, "calib_data.npz")

# ── Calibration UI points (normalized screen coords)
CALIB_POINTS = [
    (0.05, 0.05), (0.50, 0.05), (0.95, 0.05),
    (0.05, 0.50), (0.50, 0.50), (0.95, 0.50),
    (0.05, 0.95), (0.50, 0.95), (0.95, 0.95),
]
CALIB_LABELS = ["TL", "TC", "TR", "ML", "C", "MR", "BL", "BC", "BR"]
CALIB_WAIT_S = 1.5
CALIB_FRAMES = 70

# ── Camera / Display
DISPLAY_W = 1920; DISPLAY_H = 1080
CAM_W = 1280;     CAM_H = 720
INFER_W = 640;    INFER_H = 480
CAM_FPS = 30;     CAM_INDEX = 0

# ── Landmarks
L_IRIS_IDX = [468, 469, 470, 471, 472]
R_IRIS_IDX = [473, 474, 475, 476, 477]
L_EYE      = {"outer": 33,  "inner": 133, "top": 159, "bot": 145}
R_EYE      = {"outer": 362, "inner": 263, "top": 386, "bot": 374}
L_EAR_PTS  = [33, 160, 158, 133, 153, 144]
R_EAR_PTS  = [362, 385, 387, 263, 373, 380]
EM_EYE_A   = 33
EM_EYE_B   = 263

# ── Blink / Confidence
EAR_BLINK_TH = 0.22
CONF_MIN     = 0.30

# ── Zones
EDGE_MARGIN      = 0.08
ZONE_INNER_H_DEF = 0.13
ZONE_INNER_V_DEF = 0.11
OFFSCREEN_HOLD   = 1.0

# ── Head-Eye Fusion
HEAD_YAW_SCALE   = 0.020
HEAD_PITCH_SCALE = 0.024
HEAD_YAW_SOFT    = 6.0
HEAD_YAW_HARD    = 30.0
HEAD_PITCH_SOFT  = 4.0
HEAD_PITCH_HARD  = 22.0
FUSION_EXP_ALPHA = 0.30

# ── Head offscreen fallback thresholds
HEAD_YAW_OFFSCREEN   = 22.0
HEAD_PITCH_OFFSCREEN = 18.0

# ── Voting
VOTE_WINDOW   = 6
VOTE_MAJORITY = 0.50

# ── Light adapt
CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)

# ── Signature matching
#    Wider threshold → more tolerant of pose / distance variation
SIG_MATCH_THRESHOLD = 0.12   # was 0.075 – raised for angle/distance robustness
SIG_UPSERT_THRESHOLD = 0.08  # was 0.060
SIG_WINDOW  = 40             # smoothing frames (was 25)
# Consecutive frame requirement before "different face" is declared
FACE_MISMATCH_HOLD_FRAMES = 45   # ~1.5 s at 30 fps

DIR_TABLE = {
    (-1, -1): "UP-LEFT",   (0, -1): "UP",     (1, -1): "UP-RIGHT",
    (-1,  0): "LEFT",      (0,  0): "CENTER", (1,  0): "RIGHT",
    (-1,  1): "DOWN-LEFT", (0,  1): "DOWN",   (1,  1): "DOWN-RIGHT",
}
DIR_COLOR = {
    "CENTER":     (30, 210, 30),    "LEFT":       (255, 100, 10),
    "RIGHT":      (10, 100, 255),   "UP":         (220, 0, 220),
    "DOWN":       (0, 220, 220),    "UP-LEFT":    (200, 40, 180),
    "UP-RIGHT":   (40, 40, 255),    "DOWN-LEFT":  (200, 200, 0),
    "DOWN-RIGHT": (0, 200, 150),    "LOW_CONF":   (120, 120, 120),
    "NO_FACE":    (60, 60, 200),    "NEW_FACE":   (60, 180, 255),
}

# ── Emotion
EMOTIEFF_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                   "Happiness", "Neutral", "Sadness", "Surprise"]
EMOTIEFF_TO_INTERNAL = {
    "Anger": "angry",   "Contempt": "contempt", "Disgust": "disgust",
    "Fear": "fear",     "Happiness": "happy",   "Neutral": "neutral",
    "Sadness": "sad",   "Surprise": "surprise",
}
EMOTIONS = ["angry", "contempt", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_META = {
    "angry":    {"label": "Angry",    "color": (60,  60,  220)},
    "contempt": {"label": "Contempt", "color": (100, 80,  160)},
    "disgust":  {"label": "Disgust",  "color": (40,  130, 190)},
    "fear":     {"label": "Fear",     "color": (160, 40,  200)},
    "happy":    {"label": "Happy",    "color": (50,  210, 100)},
    "sad":      {"label": "Sad",      "color": (200, 120, 40)},
    "surprise": {"label": "Surprise", "color": (30,  210, 255)},
    "neutral":  {"label": "Neutral",  "color": (150, 150, 155)},
}

# Light adaptor
class LightAdaptor:
    def __init__(self):
        self._clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        self._bright = collections.deque(maxlen=30)
        self.brightness   = 128.0
        self.frame_quality = 1.0
        self.light_label  = "NORMAL"

    def process(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        self._bright.append(float(np.mean(l)))
        self.brightness = float(np.mean(self._bright))

        if self.brightness < 80 or self.brightness > 200:
            lab = cv2.merge([self._clahe.apply(l), a, b])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            out = bgr

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        self.frame_quality = float(np.clip((lap_var - 20.0) / 480.0, 0.0, 1.0))

        if   self.brightness < 60:  self.light_label = "DARK"
        elif self.brightness < 90:  self.light_label = "DIM"
        elif self.brightness > 210: self.light_label = "BRIGHT"
        else:                       self.light_label = "NORMAL"
        return out

    def status(self) -> str:
        return f"LIGHT:{self.light_label} ({self.brightness:.0f})"

# Geometry helpers
def _lm(face, idx):
    p = face[idx]
    return float(p.x), float(p.y)

def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_ear_6pt(face, pts) -> float:
    P = [_lm(face, i) for i in pts]
    return (_dist(P[1], P[5]) + _dist(P[2], P[4])) / (2.0 * max(_dist(P[0], P[3]), 1e-7))

def iris_in_eyebox(face, eye, iris_idx):
    if face is None or max(iris_idx) >= len(face):
        return None
    pts = [_lm(face, eye[k]) for k in ("outer", "inner", "top", "bot")]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    xmn, xmx = min(xs), max(xs); ymn, ymx = min(ys), max(ys)
    pw = (xmx - xmn) * 0.22;  ph = (ymx - ymn) * 0.42
    xmn -= pw; xmx += pw; ymn -= ph; ymx += ph
    bw = max(xmx - xmn, 1e-8); bh = max(ymx - ymn, 1e-8)
    iris_pts = [_lm(face, i) for i in iris_idx]
    icx = sum(p[0] for p in iris_pts) / len(iris_pts)
    icy = sum(p[1] for p in iris_pts) / len(iris_pts)
    return (float(np.clip((icx - xmn) / bw, 0, 1)),
            float(np.clip((icy - ymn) / bh, 0, 1)))

def gaze_pose_degrees(face, iris_l=None, iris_r=None):
    if face is None or (iris_l is None and iris_r is None):
        return None
    if iris_l is not None and iris_r is not None:
        gx = (iris_l[0] + iris_r[0]) * 0.5
        gy = (iris_l[1] + iris_r[1]) * 0.5
    else:
        gx, gy = iris_l if iris_l is not None else iris_r
    yaw   = (gx - 0.5) * 100.0
    pitch = (0.5 - gy) * 70.0
    ax, ay = _lm(face, EM_EYE_A)
    bx, by = _lm(face, EM_EYE_B)
    roll = math.degrees(math.atan2(by - ay, bx - ax))
    return float(pitch), float(yaw), float(roll)

def head_pose_yaw_pitch(result):
    try:
        mats = result.facial_transformation_matrixes
        if not mats:
            return None
        arr = mats[0].data if hasattr(mats[0], "data") else list(mats[0])
        M = np.array(arr, np.float32).reshape(4, 4)
        R = M[:3, :3]
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            yaw   = math.degrees(math.atan2(-R[2, 0], sy))
        else:
            pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
            yaw   = math.degrees(math.atan2(-R[2, 0], sy))
        yaw = -yaw
        return float(pitch), float(yaw)
    except Exception:
        return None

def zone_from_xy(sx, sy, th_h, th_v, edge_margin=EDGE_MARGIN):
    h = v = 0
    if sx < -edge_margin:       h = -1
    elif sx > 1 + edge_margin:  h =  1
    if sy < -edge_margin:       v = -1
    elif sy > 1 + edge_margin:  v =  1
    is_off = (h != 0 or v != 0)
    if not is_off:
        if   sx < 0.5 - th_h: h = -1
        elif sx > 0.5 + th_h: h =  1
        if   sy < 0.5 - th_v: v = -1
        elif sy > 0.5 + th_v: v =  1
    return DIR_TABLE.get((h, v), "CENTER"), is_off

def head_offscreen_fallback(hp, head_center):
    if hp is None:
        return False
    cy = head_center[0] if head_center else hp[1]
    cp = head_center[1] if head_center else hp[0]
    ry = hp[1] - cy;  rp = hp[0] - cp
    return (ry / HEAD_YAW_OFFSCREEN) ** 2 + (rp / HEAD_PITCH_OFFSCREEN) ** 2 > 1.0

def face_size_score(face) -> float:
    if face is None:
        return 0.0
    try:
        return float(np.clip(_dist(_lm(face, 1), _lm(face, 152)) / 0.085, 0.0, 1.0))
    except Exception:
        return 0.0

def confidence_score(face, ear_l, ear_r, hp, head_center, frame_quality=1.0) -> float:
    head_c = 0.20
    if hp is not None:
        cy = head_center[0] if head_center else hp[1]
        cp = head_center[1] if head_center else hp[0]
        yaw_c   = float(np.clip(1.0 - abs(hp[1] - cy) / HEAD_YAW_OFFSCREEN,   0, 1))
        pitch_c = float(np.clip(1.0 - abs(hp[0] - cp) / HEAD_PITCH_OFFSCREEN, 0, 1))
        head_c  = min(yaw_c, pitch_c)
    ear_avg = ((ear_l or 0.0) + (ear_r or 0.0)) * 0.5
    ear_c   = float(np.clip((ear_avg - EAR_BLINK_TH) / 0.18, 0, 1))
    size_c  = face_size_score(face) if face is not None else 0.6
    raw = 0.55 * head_c + 0.25 * ear_c + 0.20 * size_c
    cap = 0.75 + 0.25 * float(np.clip(frame_quality, 0, 1))
    return float(np.clip(raw, 0, cap))

# Face crop (for emotion)
def extract_face_crop(frame_bgr: np.ndarray, landmarks, align: bool = True):
    if landmarks is None or frame_bgr is None:
        return None
    try:
        h, w = frame_bgr.shape[:2]
        xs = np.array([lm.x for lm in landmarks], np.float32)
        ys = np.array([lm.y for lm in landmarks], np.float32)
        bw = max(float(xs.max() - xs.min()), 1e-6)
        bh = max(float(ys.max() - ys.min()), 1e-6)
        x0 = int(np.clip((xs.min() - EM_CROP_MARGIN * bw) * w, 0, w - 1))
        x1 = int(np.clip((xs.max() + EM_CROP_MARGIN * bw) * w, 1, w))
        y0 = int(np.clip((ys.min() - EM_CROP_MARGIN * bh) * h, 0, h - 1))
        y1 = int(np.clip((ys.max() + EM_CROP_MARGIN * bh) * h, 1, h))
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame_bgr[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return None
        if align:
            a = landmarks[EM_EYE_A]; b = landmarks[EM_EYE_B]
            ax_ = a.x * w - x0; ay_ = a.y * h - y0
            bx_ = b.x * w - x0; by_ = b.y * h - y0
            angle = math.degrees(math.atan2(by_ - ay_, bx_ - ax_))
            if abs(angle) > 1.0:
                ch, cw = crop.shape[:2]
                M = cv2.getRotationMatrix2D((cw * 0.5, ch * 0.5), -angle, 1.0)
                crop = cv2.warpAffine(crop, M, (cw, ch),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        return crop
    except Exception:
        return None

# Kalman filter (2D gaze)
class KF2DAdaptive:
    def __init__(self, fps=30.0, q=1e-4, r=5e-3,
                 max_missing=6, max_jump=0.22, max_vel=0.10):
        self.fps = float(max(fps, 1e-6))
        dt = 1.0 / self.fps
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix   = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
        self.kf.errorCovPost       = np.eye(4, dtype=np.float32)
        self._init       = False
        self.missing     = 0
        self.max_missing = int(max_missing)
        self.max_jump    = float(max_jump)
        self.max_vel     = float(max_vel)

    def reset(self):
        self._init = False
        self.missing = 0

    def _init_at(self, x, y):
        st = np.zeros((4, 1), np.float32)
        st[0, 0] = x; st[1, 0] = y
        self.kf.statePost = st; self.kf.statePre = st.copy()
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        self._init = True; self.missing = 0

    def _set_dt(self, dt):
        dt = float(max(dt, 1e-6))
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt

    def step(self, meas_xy, valid: bool, dt=None):
        if dt is not None:
            self._set_dt(dt)
        if not self._init:
            if valid and meas_xy is not None:
                self._init_at(float(meas_xy[0]), float(meas_xy[1]))
                return float(meas_xy[0]), float(meas_xy[1]), False
            return 0.5, 0.5, True

        pred = self.kf.predict()
        px, py = float(pred[0,0]), float(pred[1,0])
        vx, vy = float(pred[2,0]), float(pred[3,0])

        if abs(vx) > self.max_vel * 2.0 or abs(vy) > self.max_vel * 2.0:
            if valid and meas_xy is not None:
                self._init_at(float(meas_xy[0]), float(meas_xy[1]))
                return float(meas_xy[0]), float(meas_xy[1]), False
            self.missing += 1
            if self.missing >= self.max_missing: self.reset()
            return px, py, True

        if not valid or meas_xy is None:
            self.missing += 1
            if self.missing >= self.max_missing: self.reset()
            return px, py, True

        mx, my = float(meas_xy[0]), float(meas_xy[1])
        if math.hypot(mx - px, my - py) > self.max_jump:
            self._init_at(mx, my)
            return mx, my, False

        est = self.kf.correct(np.array([[mx], [my]], np.float32))
        self.missing = 0
        return float(est[0,0]), float(est[1,0]), False

# 1-D Kalman for emotion smoothing
class _EmKalman1D:
    def __init__(self):
        self.x = 0.0; self.P = 1.0

    def update(self, z: float) -> float:
        P_ = self.P + EM_KALMAN_Q
        K  = P_ / (P_ + EM_KALMAN_R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * P_
        return self.x

    def reset(self):
        self.x = 0.0; self.P = 1.0

# Head center tracker (adaptive baseline)
class HeadCenterTracker:
    """
    Builds an initial baseline from stable forward-facing frames,
    then slowly adapts (handles gradual posture change).
    Gating is relaxed so it still works when face is slightly angled.
    """
    def __init__(self, init_frames=60, alpha=0.01,
                 min_conf=0.30, iris_center_th=0.15):
        self.init_frames    = int(init_frames)
        self.alpha          = float(alpha)
        self.min_conf       = float(min_conf)
        self.iris_center_th = float(iris_center_th)  # relaxed from 0.10
        self._buf  = []
        self._ready = False
        self.yaw   = 0.0
        self.pitch = 0.0

    def reset(self):
        self._buf.clear(); self._ready = False
        self.yaw = 0.0;    self.pitch  = 0.0

    def _gate(self, hp, iris_xy, conf, blink_bad) -> bool:
        if hp is None or blink_bad or conf < self.min_conf:
            return False
        if iris_xy is not None:
            if abs(iris_xy[0] - 0.5) > self.iris_center_th:
                return False
            if abs(iris_xy[1] - 0.5) > self.iris_center_th:
                return False
        return True

    def update(self, hp, iris_xy, conf, blink_bad):
        if not self._gate(hp, iris_xy, conf, blink_bad):
            return
        pitch, yaw = float(hp[0]), float(hp[1])
        if not self._ready:
            self._buf.append((yaw, pitch))
            if len(self._buf) >= self.init_frames:
                self.yaw   = float(np.median([v[0] for v in self._buf]))
                self.pitch = float(np.median([v[1] for v in self._buf]))
                self._ready = True
            return
        a = self.alpha
        self.yaw   = (1 - a) * self.yaw   + a * yaw
        self.pitch = (1 - a) * self.pitch + a * pitch

    def center(self):
        return (self.yaw, self.pitch) if self._ready else None

# Head-Eye Fusion
class HeadEyeFusion:
    def __init__(self, head_tracker: HeadCenterTracker):
        self._ht  = head_tracker
        self._sm_x = 0.5;  self._sm_y = 0.5

    def reset(self):
        self._sm_x = 0.5;  self._sm_y = 0.5

    def reset_to(self, x, y):
        self._sm_x = float(x);  self._sm_y = float(y)

    def fuse(self, screen_xy_raw, hp):
        if screen_xy_raw is None:
            return None, 0.0
        ix, iy = screen_xy_raw
        if hp is None:
            self._sm_x = self._sm_x * (1 - FUSION_EXP_ALPHA) + ix * FUSION_EXP_ALPHA
            self._sm_y = self._sm_y * (1 - FUSION_EXP_ALPHA) + iy * FUSION_EXP_ALPHA
            return (self._sm_x, self._sm_y), 0.0

        center = self._ht.center()
        cy = center[0] if center else hp[1]
        cp = center[1] if center else hp[0]
        ry = hp[1] - cy;  rp = hp[0] - cp

        hsx = 0.5 + ry * HEAD_YAW_SCALE
        hsy = 0.5 + rp * HEAD_PITCH_SCALE

        yw = float(np.clip((abs(ry) - HEAD_YAW_SOFT)   / max(HEAD_YAW_HARD   - HEAD_YAW_SOFT,   1), 0, 1))
        pw = float(np.clip((abs(rp) - HEAD_PITCH_SOFT) / max(HEAD_PITCH_HARD - HEAD_PITCH_SOFT, 1), 0, 1))
        hw = max(yw, pw)

        rawx = ix * (1 - hw) + hsx * hw
        rawy = iy * (1 - hw) + hsy * hw

        self._sm_x = self._sm_x * (1 - FUSION_EXP_ALPHA) + rawx * FUSION_EXP_ALPHA
        self._sm_y = self._sm_y * (1 - FUSION_EXP_ALPHA) + rawy * FUSION_EXP_ALPHA
        return (self._sm_x, self._sm_y), hw

# Direction voter
class DirectionVoter:
    def __init__(self, window=VOTE_WINDOW, majority=VOTE_MAJORITY):
        self._q   = collections.deque(maxlen=window)
        self._maj = float(majority)
        self.last = "CENTER"

    def reset(self):
        self._q.clear(); self.last = "CENTER"

    def seed(self, direction):
        self._q.clear()
        for _ in range(max(3, self._q.maxlen // 2)):
            self._q.append(direction)
        self.last = direction

    def push(self, direction) -> str:
        self._q.append(direction)
        cnt = collections.Counter(self._q)
        top, n = cnt.most_common(1)[0]
        if len(self._q) >= 3 and n / len(self._q) >= self._maj:
            self.last = top
        return self.last

# Offscreen monitor (no file IO)
class OffscreenMonitor:
    def __init__(self, hold=OFFSCREEN_HOLD, on_event=None):
        self.hold     = float(hold)
        self.on_event = on_event
        self._since   = None
        self._active  = False
        self._last    = "CENTER"
        self.dur      = 0.0

    def update(self, is_off, direction, conf):
        now = time.time()
        just_returned = False
        if is_off:
            self._last = direction
            if self._since is None: self._since = now
            self.dur = now - self._since
            if self.dur >= self.hold and not self._active:
                self._active = True
                if self.on_event:
                    self.on_event("OFFSCREEN_START", self._last, 0.0, conf, now)
        else:
            if self._active and self._since is not None:
                dur = now - self._since
                if self.on_event:
                    self.on_event("OFFSCREEN_END", self._last, dur, conf, now)
                just_returned = True
            self._active = False; self._since = None; self.dur = 0.0
        return self._active, just_returned

    def force_close(self):
        if self._active and self._since is not None:
            dur = time.time() - self._since
            if self.on_event:
                self.on_event("OFFSCREEN_END", self._last, dur, 0.0, time.time())
            self._active = False; self._since = None; self.dur = 0.0

# Face signature – robust, scale + partial-pose invariant
_SIG_PTS = {
    "L_EYE_OUT": 33,   "R_EYE_OUT":  263,
    "FOREHEAD":  10,   "CHIN":       152,
    "L_CHEEK":   234,  "R_CHEEK":    454,
    "MOUTH_L":   61,   "MOUTH_R":    291,
    "NOSE_TIP":  4,    "NOSE_BASE":  2,
    "L_BROW":    70,   "R_BROW":     300,
    "UPPER_LIP": 13,   "LOWER_LIP":  14,
}

def face_signature(face) -> np.ndarray:
    """
    12-ratio pose-robust signature.
    Each ratio is computed from same-depth landmark pairs to minimise
    perspective distortion when the head is turned or tilted.
    """
    if face is None:
        return None
    try:
        p_le  = _lm(face, _SIG_PTS["L_EYE_OUT"])
        p_re  = _lm(face, _SIG_PTS["R_EYE_OUT"])
        p_top = _lm(face, _SIG_PTS["FOREHEAD"])
        p_ch  = _lm(face, _SIG_PTS["CHIN"])
        p_lc  = _lm(face, _SIG_PTS["L_CHEEK"])
        p_rc  = _lm(face, _SIG_PTS["R_CHEEK"])
        p_ml  = _lm(face, _SIG_PTS["MOUTH_L"])
        p_mr  = _lm(face, _SIG_PTS["MOUTH_R"])
        p_nt  = _lm(face, _SIG_PTS["NOSE_TIP"])
        p_nb  = _lm(face, _SIG_PTS["NOSE_BASE"])
        p_lb  = _lm(face, _SIG_PTS["L_BROW"])
        p_rb  = _lm(face, _SIG_PTS["R_BROW"])
        p_ul  = _lm(face, _SIG_PTS["UPPER_LIP"])
        p_ll  = _lm(face, _SIG_PTS["LOWER_LIP"])

        d_io  = _dist(p_le, p_re)    # inter-ocular
        d_fh  = _dist(p_top, p_ch)   # face height
        d_fw  = _dist(p_lc, p_rc)    # face width
        d_mw  = _dist(p_ml, p_mr)    # mouth width
        d_nw  = _dist(p_nt, p_nb)    # nose length
        d_bw  = _dist(p_lb, p_rb)    # brow span
        d_lip = _dist(p_ul, p_ll)    # lip gap
        # mid-eye to nose tip (vertical)
        eye_mid = ((p_le[0]+p_re[0])*0.5, (p_le[1]+p_re[1])*0.5)
        d_en  = _dist(eye_mid, p_nt)

        eps = 1e-6
        sig = np.array([
            d_io  / max(d_fw,  eps),
            d_io  / max(d_fh,  eps),
            d_mw  / max(d_fw,  eps),
            d_fw  / max(d_fh,  eps),
            d_nw  / max(d_fh,  eps),
            d_bw  / max(d_fw,  eps),
            d_en  / max(d_fh,  eps),
            d_mw  / max(d_io,  eps),
            d_lip / max(d_mw,  eps),
            d_io  / max(d_bw,  eps),
            d_en  / max(d_io,  eps),
            d_nw  / max(d_io,  eps),
        ], np.float32)
        return sig
    except Exception:
        return None


class SignatureTracker:
    """Running mean over a window; returns None until enough samples."""
    def __init__(self, window=SIG_WINDOW):
        self._q = collections.deque(maxlen=int(window))

    def reset(self):
        self._q.clear()

    def push(self, sig: np.ndarray):
        if sig is None:
            return None
        self._q.append(np.asarray(sig, np.float32))
        if len(self._q) < 8:
            return None
        return np.mean(np.stack(self._q, axis=0), axis=0)

# Calibration data store (single .npz, multiple profiles)
class AffineMapper:
    def __init__(self):
        self.A     = None;  self.ready = False
        self.th_h  = ZONE_INNER_H_DEF
        self.th_v  = ZONE_INNER_V_DEF
        self._src  = [];    self._dst  = []

    def reset(self):
        self.A = None;  self.ready = False
        self.th_h = ZONE_INNER_H_DEF
        self.th_v = ZONE_INNER_V_DEF
        self._src.clear();  self._dst.clear()

    def add(self, iris_xy, screen_xy):
        self._src.append(iris_xy);  self._dst.append(screen_xy)

    def fit(self) -> bool:
        if len(self._src) < 6:
            return False
        src = np.array(self._src, np.float32)
        dst = np.array(self._dst, np.float32)
        X   = np.hstack([src, np.ones((len(src), 1), np.float32)])
        A, *_ = np.linalg.lstsq(X, dst, rcond=None)
        self.A = A

        mapped = list(X @ A)
        mapped = [(float(r[0]), float(r[1])) for r in mapped]

        lp = [mapped[i] for i,d in enumerate(self._dst) if d[0] < 0.3]
        rp = [mapped[i] for i,d in enumerate(self._dst) if d[0] > 0.7]
        cp = [mapped[i] for i,d in enumerate(self._dst) if 0.3 <= d[0] <= 0.7 and 0.3 <= d[1] <= 0.7]
        if lp and rp and cp:
            cx = np.mean([p[0] for p in cp])
            lx = np.mean([p[0] for p in lp])
            rx = np.mean([p[0] for p in rp])
            self.th_h = float(np.clip(0.28 * min(cx - lx, rx - cx), 0.07, 0.22))

        tp = [mapped[i] for i,d in enumerate(self._dst) if d[1] < 0.3]
        bp = [mapped[i] for i,d in enumerate(self._dst) if d[1] > 0.7]
        if tp and bp and cp:
            cy = np.mean([p[1] for p in cp])
            ty = np.mean([p[1] for p in tp])
            by = np.mean([p[1] for p in bp])
            self.th_v = float(np.clip(0.28 * min(cy - ty, by - cy), 0.05, 0.20))

        self.ready = True
        return True

    def map(self, iris_xy, clip=True):
        if not self.ready or self.A is None or iris_xy is None:
            return iris_xy
        v = np.array([iris_xy[0], iris_xy[1], 1.0], np.float32)
        r = v @ self.A
        x, y = float(r[0]), float(r[1])
        if clip:
            x = float(np.clip(x, 0, 1))
            y = float(np.clip(y, 0, 1))
        return x, y

    def save_single(self, path=None):
        if path is None:
            path = CALIB_FILE
        if self.A is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, A=self.A, th_h=[self.th_h], th_v=[self.th_v])

    def load_single(self, path=None) -> bool:
        if path is None:
            path = CALIB_FILE
        try:
            d = np.load(path, allow_pickle=True)
            if "A" not in d:
                return False
            self.A = d["A"].astype(np.float32)
            self.th_h = float(d["th_h"][0]) if "th_h" in d else ZONE_INNER_H_DEF
            self.th_v = float(d["th_v"][0]) if "th_v" in d else ZONE_INNER_V_DEF
            self.ready = True
            return True
        except Exception:
            return False



class ProfileStore:
    K_SIG = 12   # must match face_signature() output length

    def __init__(self, path=CALIB_FILE):
        self.path  = path
        self.sigs  = np.zeros((0, self.K_SIG), np.float32)
        self.As    = np.zeros((0, 3, 2),        np.float32)
        self.th_h  = np.zeros((0,),             np.float32)
        self.th_v  = np.zeros((0,),             np.float32)
        self.ts    = np.zeros((0,),             np.float64)
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            d = np.load(self.path, allow_pickle=True)
            if "sigs" in d and "As" in d:
                sigs = d["sigs"].astype(np.float32)
                # Migrate 4-dim signatures from old version
                if sigs.shape[1] != self.K_SIG:
                    # pad / truncate to new size; old profiles become "unknown"
                    n = sigs.shape[0]
                    new_sigs = np.zeros((n, self.K_SIG), np.float32)
                    copy_len = min(sigs.shape[1], self.K_SIG)
                    new_sigs[:, :copy_len] = sigs[:, :copy_len]
                    sigs = new_sigs
                self.sigs = sigs
                self.As   = d["As"].astype(np.float32)
                self.th_h = d["th_h"].astype(np.float32)
                self.th_v = d["th_v"].astype(np.float32)
                self.ts   = d["ts"].astype(np.float64)
                return
            # Fallback: legacy single-profile format
            if "A" in d:
                A    = d["A"].astype(np.float32).reshape(3, 2)
                th_h = float(d["th_h"][0]) if "th_h" in d else ZONE_INNER_H_DEF
                th_v = float(d["th_v"][0]) if "th_v" in d else ZONE_INNER_V_DEF
                self.sigs = np.zeros((1, self.K_SIG), np.float32)
                self.As   = A.reshape(1, 3, 2)
                self.th_h = np.array([th_h], np.float32)
                self.th_v = np.array([th_v], np.float32)
                self.ts   = np.array([time.time()], np.float64)
                self._save()
        except Exception:
            pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        np.savez(self.path, sigs=self.sigs, As=self.As,
                 th_h=self.th_h, th_v=self.th_v, ts=self.ts)

    def count(self) -> int:
        return int(self.As.shape[0])

    def clear(self):
        self.sigs = np.zeros((0, self.K_SIG), np.float32)
        self.As   = np.zeros((0, 3, 2),       np.float32)
        self.th_h = np.zeros((0,),            np.float32)
        self.th_v = np.zeros((0,),            np.float32)
        self.ts   = np.zeros((0,),            np.float64)
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            pass

    def match(self, sig: np.ndarray, threshold=SIG_MATCH_THRESHOLD):
        """Return (idx, dist) or (None, dist)."""
        if sig is None or self.count() == 0:
            return None, None
        valid = np.linalg.norm(self.sigs, axis=1) > 1e-6
        if not np.any(valid):
            return None, None
        dif  = self.sigs[valid] - sig.reshape(1, -1)
        ds   = np.sqrt(np.sum(dif * dif, axis=1))
        j    = int(np.argmin(ds))
        dmin = float(ds[j])
        idxs = np.flatnonzero(valid)
        idx  = int(idxs[j])
        return (idx, dmin) if dmin <= threshold else (None, dmin)

    def load_mapper(self, idx: int) -> AffineMapper:
        m = AffineMapper()
        idx = int(np.clip(idx, 0, max(0, self.count() - 1)))
        m.A    = self.As[idx].astype(np.float32)
        m.th_h = float(self.th_h[idx])
        m.th_v = float(self.th_v[idx])
        m.ready = True
        return m

    def upsert(self, sig: np.ndarray, mapper: AffineMapper,
               threshold=SIG_UPSERT_THRESHOLD):
        if sig is None or mapper is None or not mapper.ready or mapper.A is None:
            return None
        sig = sig.astype(np.float32).reshape(1, -1)
        # Pad to K_SIG if necessary (e.g. old 4-dim sig passed in)
        if sig.shape[1] != self.K_SIG:
            ns = np.zeros((1, self.K_SIG), np.float32)
            ns[:, :min(sig.shape[1], self.K_SIG)] = sig[:, :self.K_SIG]
            sig = ns
        idx, d = self.match(sig[0], threshold=threshold)
        if idx is None:
            self.sigs = np.vstack([self.sigs, sig])
            self.As   = np.concatenate([self.As,   mapper.A.reshape(1,3,2).astype(np.float32)], axis=0)
            self.th_h = np.append(self.th_h, np.array([mapper.th_h], np.float32))
            self.th_v = np.append(self.th_v, np.array([mapper.th_v], np.float32))
            self.ts   = np.append(self.ts,   np.array([time.time()], np.float64))
            self._save()
            return self.count() - 1
        self.sigs[idx]  = sig[0]
        self.As[idx]    = mapper.A.astype(np.float32).reshape(3, 2)
        self.th_h[idx]  = float(mapper.th_h)
        self.th_v[idx]  = float(mapper.th_v)
        self.ts[idx]    = float(time.time())
        self._save()
        return idx


class EmotionTracker:
    def __init__(self, model_name=EM_MODEL_NAME, engine=EM_ENGINE):
        self._lock            = threading.Lock()
        self._kalman          = {e: _EmKalman1D() for e in EMOTIONS}
        self._ema             = {e: 0.0 for e in EMOTIONS}
        self._raw_scores      = {e: 0.0 for e in EMOTIONS}
        self._pending         = None
        self._candidate       = "neutral"
        self._candidate_count = 0
        self._stable          = "neutral"
        self.smoothed         = {e: 0.0 for e in EMOTIONS}
        self.dominant         = "neutral"
        self.available        = False
        self.analysis_fps     = 0.0
        self._running         = True
        self._model_name      = model_name
        self._engine          = engine
        self._thread          = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def feed(self, face_crop_bgr: np.ndarray):
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return
        with self._lock:
            self._pending = face_crop_bgr

    def _loop(self):
        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
            rec = EmotiEffLibRecognizer(engine=self._engine,
                                        model_name=self._model_name,
                                        device="cpu")
            self.available = True
        except Exception:
            return  # library not installed – silent fallback

        while self._running:
            crop = None
            with self._lock:
                if self._pending is not None:
                    crop, self._pending = self._pending, None
            if crop is None:
                time.sleep(0.005)
                continue
            t0 = time.perf_counter()
            try:
                _, scores = rec.predict_emotions(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), logits=False)
                probs = np.asarray(scores, np.float32)
                probs = probs[0] if probs.ndim == 2 else probs.reshape(-1)
                s = float(probs.sum())
                if s > 1e-8:
                    probs /= s
                raw = {}
                for i, lab in enumerate(EMOTIEFF_LABELS):
                    k = EMOTIEFF_TO_INTERNAL.get(lab)
                    if k and i < len(probs):
                        raw[k] = float(probs[i])
                with self._lock:
                    for e in EMOTIONS:
                        self._raw_scores[e] = raw.get(e, 0.0)
                    self.analysis_fps = 1.0 / max(time.perf_counter() - t0, 1e-9)
            except Exception:
                pass

    def tick(self):
        """Call once per video frame; returns (smoothed_dict, dominant_str)."""
        if not self.available:
            return dict(self.smoothed), self.dominant
        with self._lock:
            raw = dict(self._raw_scores)
        total = max(1e-9, sum(raw.values()))
        norm  = {e: raw[e] / total for e in EMOTIONS}
        k_out = {e: self._kalman[e].update(norm[e]) for e in EMOTIONS}
        for e in EMOTIONS:
            self._ema[e] = EM_EMA_ALPHA * k_out[e] + (1 - EM_EMA_ALPHA) * self._ema[e]
        s = max(1e-9, sum(self._ema.values()))
        for e in EMOTIONS:
            self.smoothed[e] = self._ema[e] / s * 100.0

        cand = max(self.smoothed, key=self.smoothed.get)
        if cand == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate       = cand
            self._candidate_count = 1
        if self._candidate_count >= EM_DEBOUNCE:
            self._stable = self._candidate
        self.dominant = self._stable
        return dict(self.smoothed), self.dominant

    def reset(self):
        for e in EMOTIONS:
            self._kalman[e].reset()
            self._ema[e] = 0.0
        self._candidate       = "neutral"
        self._candidate_count = 0
        self._stable          = "neutral"
        with self._lock:
            self._raw_scores = {e: 0.0 for e in EMOTIONS}

    def stop(self):
        self._running = False
        try: self._thread.join(timeout=0.5)
        except Exception: pass

#UI
class CalibratorUI:
    def __init__(self, label="Calibration"):
        self.label = label
        self.reset()

    def reset(self):
        self.idx   = 0;    self.start = None
        self.samps = [];   self.wait  = True
        self.done  = False
        self._mapper = AffineMapper()

    def update(self, iris_xy, conf, frame, face_ok) -> bool:
        pt    = CALIB_POINTS[self.idx]
        label = CALIB_LABELS[self.idx]
        now   = time.time()
        if self.start is None:
            self.start = now;  self.samps = [];  self.wait = True

        if self.wait:
            prog = 0.0
            if now - self.start >= CALIB_WAIT_S:
                self.wait = False;  self.start = now
        else:
            if iris_xy is not None and conf > 0.28:
                self.samps.append(iris_xy)
            prog = min(1.0, len(self.samps) / CALIB_FRAMES)
            if len(self.samps) >= CALIB_FRAMES:
                mx = float(np.mean([p[0] for p in self.samps]))
                my = float(np.mean([p[1] for p in self.samps]))
                self._mapper.add((mx, my), pt)
                self.idx += 1;  self.start = None
                if self.idx >= len(CALIB_POINTS):
                    if self._mapper.fit():
                        self.done = True
                        return True

        _draw_calib(frame, pt, label, self.wait, prog, self.idx, face_ok, self.label)
        return False

    def get_mapper(self) -> AffineMapper:
        return self._mapper


def _draw_calib(frame, pt, label, wait, prog, idx, face_ok, title):
    H, W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (W,H), (0,0,0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    px = int(np.clip(pt[0]*W, 52, W-52))
    py = int(np.clip(pt[1]*H, 52, H-52))
    cv2.circle(frame, (px,py), 52, (80,80,80) if wait else (50,50,50), 2)
    if not wait:
        cv2.ellipse(frame, (px,py), (52,52), -90, 0, int(360*prog), (0,240,80), 4)
        cv2.circle(frame, (px,py), 12, (0,240,80), -1)
    cv2.putText(frame, label,  (px-40, py-66), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    cv2.putText(frame, "Face OK" if face_ok else "NO FACE", (50, H-16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,0) if face_ok else (0,40,220), 2)
    cv2.putText(frame, f"Point {idx+1}/{len(CALIB_POINTS)}", (W//2-120, H-16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 1)
    cv2.putText(frame, title,  (W//2-180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100,220,255), 2)


def draw_hud(frame, stable, conf, gp, fps, off_active, off_dur,
             fusion_w, light_str, mode="RUN", user_text="",
             emotion_label="", emotion_available=False):
    H, W = frame.shape[:2]
    col     = DIR_COLOR.get(stable, (180,180,180))
    top_col = (0, 0, 50) if not off_active else (70, 0, 0)
    cv2.rectangle(frame, (0,0), (W,92), top_col, -1)

    if mode == "KALIB":
        lbl = "KALIBRASYON"
    else:
        lbl = f"OFF-SCREEN ({off_dur:.1f}s)" if off_active else f"{stable} [{conf:.0%}]"

    (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
    cv2.putText(frame, lbl, ((W-tw)//2, 84), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,0,0), 4)
    cv2.putText(frame, lbl, ((W-tw)//2, 84), cv2.FONT_HERSHEY_DUPLEX, 0.75, col, 2)

    cv2.putText(frame, f"FPS:{fps:.0f}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160,160,160), 1)

    # Emotion: sadece "EMOTION: HAPPY" gibi tek satır
    if emotion_available and emotion_label:
        em_col = EMOTION_META.get(emotion_label, {}).get("color", (180,180,180))
        em_lbl = EMOTION_META.get(emotion_label, {}).get("label", emotion_label).upper()
        cv2.putText(frame, f"EMOTION: {em_lbl}", (8,48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, em_col, 2)

    if gp:
        cv2.putText(frame, f"EYE P:{gp[0]:+.0f} Y:{gp[1]:+.0f} R:{gp[2]:+.0f}",
                    (8,72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130,130,130), 1)

    cv2.putText(frame, light_str, (W-230, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,180,100), 1)

    if off_active:
        for t in range(1,6):
            cv2.rectangle(frame, (t,t), (W-t,H-t), (0,0,200), 1)


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    print(f"[gaze] face_landmarker.task bulunamadı, indiriliyor...\n  → {MODEL_URL}")
    try:
        import urllib.request
        tmp = MODEL_PATH + ".tmp"
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                print(f"\r  indiriliyor... {pct}%", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, tmp, _progress)
        print()  # newline after progress
        os.replace(tmp, MODEL_PATH)
        print(f"[gaze] Model indirildi: {MODEL_PATH}")
    except Exception as e:
        # Clean up partial download
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise RuntimeError(
            f"Model indirilemedi: {e}\n"
            f"Manuel olarak şu adresten indirip '{MODEL_PATH}' konumuna koyun:\n"
            f"  {MODEL_URL}"
        ) from e

def build_detector():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.40,
        min_face_presence_confidence=0.40,
        min_tracking_confidence=0.40,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)