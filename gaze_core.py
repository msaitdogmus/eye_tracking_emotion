# Bu satır ilgili ayarın öğrenci tarafından daha kolay anlaşılmasını sağlar.
import os, math, time, threading, collections
import cv2
import numpy as np
from mediapipe.tasks import python as Mp_Python
from mediapipe.tasks.python import vision as Mp_Vision

# Geriye dönük uyumluluk için temel ayarlar tutulur.
Model_İsim  = "enet_b0_8_best_vgaf"
Model_Motor      = "onnx"
Kesme_Payı  = 0.22
Duygu_Besleme_Aralığı  = 1
Duygu_Kalman_Q    = 0.015
Duygu_Kalman_R    = 0.08
Duygu_Ema_Alfa   = 0.40
Duygu_Kararlılık_Karesi    = 4

Temel_Dizin   = os.path.dirname(os.path.abspath(__file__))
Model_Yolu = os.path.join(Temel_Dizin, "face_landmarker.task")
Kalibrasyon_Dosyası = os.path.join(Temel_Dizin, "calib_data.npz")

# Kalibrasyon noktaları ekranda bakılacak yerleri belirler.
Kalibrasyon_Noktaları = [
    (0.05, 0.05), (0.50, 0.05), (0.95, 0.05),
    (0.05, 0.50), (0.50, 0.50), (0.95, 0.50),
    (0.05, 0.95), (0.50, 0.95), (0.95, 0.95),
]
Kalibrasyon_Etiketleri = ["TL", "TC", "TR", "ML", "C", "MR", "BL", "BC", "BR"]
Kalibrasyon_Bekleme_Suresi = 1.5
Kalibrasyon_Kare_Sayısı = 70

# Kamera ve ekran boyut ayarları burada tutulur.
Ekran_Genişliği = 1920; Ekran_Yüksekliği = 1080
Kamera_Genişliği = 1280;     Kamera_Yüksekliği = 720
Çıkarım_Genişliği = 640;    Çıkarım_Yüksekliği = 480
Kamera_Fps = 30;     Kamera_İndeksi = 0

# Yüz işaret noktalarının indeksleri burada tanımlanır.
Sol_İris_İndeksleri = [468, 469, 470, 471, 472]
Sağ_İris_İndeksleri = [473, 474, 475, 476, 477]
Sol_Göz      = {"outer": 33,  "inner": 133, "top": 159, "bot": 145}
Sağ_Göz      = {"outer": 362, "inner": 263, "top": 386, "bot": 374}
Sol_Kulak_Noktaları  = [33, 160, 158, 133, 153, 144]
Sağ_Kulak_Noktaları  = [362, 385, 387, 263, 373, 380]
Duygu_Göz_A   = 33
Duygu_Göz_B   = 263

# Göz kırpma ve güven eşiği ayarları burada tutulur.
Göz_Kırpma_Eşiği = 0.22
En_Az_Güven     = 0.30

# Bakış yönü bölgeleri için eşik değerleri burada tanımlanır.
Kenar_Payı      = 0.08
Bölge_İç_Yatay_Varsayılan = 0.13
Bölge_İç_Dikey_Varsayılan = 0.11
Ekran_Dışı_Bekleme   = 1.0

# Baş ve göz hareketleri tek bakış sonucunda birleştirilir.
Baş_Sapma_Ölçeği   = 0.020
Baş_Eğim_Ölçeği = 0.024
Baş_Sapma_Yumuşak    = 6.0
Baş_Sapma_Sert    = 30.0
Baş_Eğim_Yumuşak  = 4.0
Baş_Eğim_Sert  = 22.0
Birleşim_Ema_Alfa = 0.30

# Baş pozu ekran dışı kontrolü için yedek eşikler burada tutulur.
Baş_Sapma_Ekran_Dışı   = 22.0
Baş_Eğim_Ekran_Dışı = 18.0

# Yön kararını kararlı yapmak için oylama ayarları kullanılır.
Oylama_Penceresi   = 6
Oylama_Çoğunluğu = 0.50

# Işık uyarlaması için görüntü iyileştirme ayarları kullanılır.
Clahe_Kırpma = 3.0
Clahe_Izgara = (8, 8)

# Yüz imzası eşleştirmesi için tolerans değerleri burada tutulur.
# Bu satır ilgili ayarın öğrenci tarafından daha kolay anlaşılmasını sağlar.
İmza_Eşleşme_Eşiği = 0.12   # Eski değer 0.075 idi ve açı ile mesafe dayanıklılığı için yükseltildi.
İmza_Kayıt_Eşiği = 0.08  # Eski değer 0.060 idi.
İmza_Penceresi  = 40             # Eski değer 25 idi ve yumuşatma karesi sayısını belirtir.
# Farklı yüz kararı için art arda gereken kare sayısı belirlenir.
Yüz_Uyuşmazlık_Bekleme_Karesi = 45   # 30 FPS için yaklaşık 1.5 saniyeye karşılık gelir.

Yön_Tablosu = {
    (-1, -1): "UP-LEFT",   (0, -1): "UP",     (1, -1): "UP-RIGHT",
    (-1,  0): "LEFT",      (0,  0): "CENTER", (1,  0): "RIGHT",
    (-1,  1): "DOWN-LEFT", (0,  1): "DOWN",   (1,  1): "DOWN-RIGHT",
}
Yön_Renkleri = {
    "CENTER":     (30, 210, 30),    "LEFT":       (255, 100, 10),
    "RIGHT":      (10, 100, 255),   "UP":         (220, 0, 220),
    "DOWN":       (0, 220, 220),    "UP-LEFT":    (200, 40, 180),
    "UP-RIGHT":   (40, 40, 255),    "DOWN-LEFT":  (200, 200, 0),
    "DOWN-RIGHT": (0, 200, 150),    "LOW_CONF":   (120, 120, 120),
    "NO_FACE":    (60, 60, 200),    "NEW_FACE":   (60, 180, 255),
}

# Duygu modeli etiketleri ve iç eşlemeleri burada tanımlanır.
Duygu_Etiketleri = ["Anger", "Contempt", "Disgust", "Fear",
                   "Happiness", "Neutral", "Sadness", "Surprise"]
Duygu_İç_Eşleme = {
    "Anger": "angry",   "Contempt": "contempt", "Disgust": "disgust",
    "Fear": "fear",     "Happiness": "happy",   "Neutral": "neutral",
    "Sadness": "sad",   "Surprise": "surprise",
}
Duygular = ["angry", "contempt", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
Duygu_Bilgileri = {
    "angry":    {"label": "Angry",    "color": (60,  60,  220)},
    "contempt": {"label": "Contempt", "color": (100, 80,  160)},
    "disgust":  {"label": "Disgust",  "color": (40,  130, 190)},
    "fear":     {"label": "Fear",     "color": (160, 40,  200)},
    "happy":    {"label": "Happy",    "color": (50,  210, 100)},
    "sad":      {"label": "Sad",      "color": (200, 120, 40)},
    "surprise": {"label": "Surprise", "color": (30,  210, 255)},
    "neutral":  {"label": "Neutral",  "color": (150, 150, 155)},
}

# Işık uyarlaması için görüntü iyileştirme ayarları kullanılır.
class Işık_Uyarlayıcı:
    def __init__(self):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self._clahe = cv2.createCLAHE(clipLimit=Clahe_Kırpma, tileGridSize=Clahe_Izgara)
        self._bright = collections.deque(maxlen=30)
        self.brightness   = 128.0
        self.Kare_Kalitesi = 1.0
        self.light_label  = "NORMAL"

    def process(self, bgr: np.ndarray) -> np.ndarray:
        """Bu fonksiyon, kameradan gelen kareyi ışık durumuna göre işler."""
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
        self.Kare_Kalitesi = float(np.clip((lap_var - 20.0) / 480.0, 0.0, 1.0))

        if   self.brightness < 60:  self.light_label = "DARK"
        elif self.brightness < 90:  self.light_label = "DIM"
        elif self.brightness > 210: self.light_label = "BRIGHT"
        else:                       self.light_label = "NORMAL"
        return out

    def status(self) -> str:
        """Bu fonksiyon, mevcut ışık bilgisini okunabilir metin olarak döndürür."""
        return f"LIGHT:{self.light_label} ({self.brightness:.0f})"

# Geometri yardımcıları yüz ve göz ölçümlerini hesaplar.
def _Nokta_Al(Yüz, İndeks):
    """Bu fonksiyon, verilen yüz işaret noktasının x ve y koordinatını alır."""
    p = Yüz[İndeks]
    return float(p.x), float(p.y)

def _Mesafe(a, b) -> float:
    """Bu fonksiyon, iki nokta arasındaki Öklid mesafesini hesaplar."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def Göz_Açıklığı_Hesapla(Yüz, Noktalar) -> float:
    """Bu fonksiyon, altı göz noktasıyla göz açıklık oranını hesaplar."""
    P = [_Nokta_Al(Yüz, i) for i in Noktalar]
    return (_Mesafe(P[1], P[5]) + _Mesafe(P[2], P[4])) / (2.0 * max(_Mesafe(P[0], P[3]), 1e-7))

def İrisi_Göz_Kutusunda_Bul(Yüz, eye, İris_İndeksleri):
    """Bu fonksiyon, irisin göz kutusu içindeki normalize konumunu bulur."""
    if Yüz is None or max(İris_İndeksleri) >= len(Yüz):
        return None
    Noktalar = [_Nokta_Al(Yüz, eye[k]) for k in ("outer", "inner", "top", "bot")]
    xs = [p[0] for p in Noktalar]; ys = [p[1] for p in Noktalar]
    xmn, xmx = min(xs), max(xs); ymn, ymx = min(ys), max(ys)
    pw = (xmx - xmn) * 0.22;  ph = (ymx - ymn) * 0.42
    xmn -= pw; xmx += pw; ymn -= ph; ymx += ph
    bw = max(xmx - xmn, 1e-8); bh = max(ymx - ymn, 1e-8)
    iris_pts = [_Nokta_Al(Yüz, i) for i in İris_İndeksleri]
    icx = sum(p[0] for p in iris_pts) / len(iris_pts)
    icy = sum(p[1] for p in iris_pts) / len(iris_pts)
    return (float(np.clip((icx - xmn) / bw, 0, 1)),
            float(np.clip((icy - ymn) / bh, 0, 1)))

def Bakış_Pozunu_Derece_Hesapla(Yüz, Sol_İris=None, Sağ_İris=None):
    """Bu fonksiyon, iris konumundan bakış eğim, sapma ve dönme değerlerini hesaplar."""
    if Yüz is None or (Sol_İris is None and Sağ_İris is None):
        return None
    if Sol_İris is not None and Sağ_İris is not None:
        gx = (Sol_İris[0] + Sağ_İris[0]) * 0.5
        gy = (Sol_İris[1] + Sağ_İris[1]) * 0.5
    else:
        gx, gy = Sol_İris if Sol_İris is not None else Sağ_İris
    yaw   = (gx - 0.5) * 100.0
    pitch = (0.5 - gy) * 70.0
    ax, ay = _Nokta_Al(Yüz, Duygu_Göz_A)
    bx, by = _Nokta_Al(Yüz, Duygu_Göz_B)
    roll = math.degrees(math.atan2(by - ay, bx - ax))
    return float(pitch), float(yaw), float(roll)

def Baş_Pozu_Sapma_Eğim(Sonuç):
    """Bu fonksiyon, MediaPipe yüz dönüşüm matrisinden baş eğimi ve sapmasını çıkarır."""
    try:
        mats = Sonuç.facial_transformation_matrixes
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

def Koordinattan_Bölge_Bul(sx, sy, th_h, th_v, edge_margin=Kenar_Payı):
    """Bu fonksiyon, normalize ekran koordinatını bakış yönü bölgesine çevirir."""
    h = v = 0
    if sx < -edge_margin:       h = -1
    elif sx > 1 + edge_margin:  h =  1
    if sy < -edge_margin:       v = -1
    elif sy > 1 + edge_margin:  v =  1
    Ekran_Dışı_Mı = (h != 0 or v != 0)
    if not Ekran_Dışı_Mı:
        if   sx < 0.5 - th_h: h = -1
        elif sx > 0.5 + th_h: h =  1
        if   sy < 0.5 - th_v: v = -1
        elif sy > 0.5 + th_v: v =  1
    return Yön_Tablosu.get((h, v), "CENTER"), Ekran_Dışı_Mı

def Baş_Ekran_Dışı_Yedek_Kontrol(hp, Baş_Merkezi):
    """Bu fonksiyon, baş pozu merkeze göre fazla saptığında ekran dışı durumu tahmin eder."""
    if hp is None:
        return False
    cy = Baş_Merkezi[0] if Baş_Merkezi else hp[1]
    cp = Baş_Merkezi[1] if Baş_Merkezi else hp[0]
    ry = hp[1] - cy;  rp = hp[0] - cp
    return (ry / Baş_Sapma_Ekran_Dışı) ** 2 + (rp / Baş_Eğim_Ekran_Dışı) ** 2 > 1.0

def Yüz_Boyut_Puanı(Yüz) -> float:
    """Bu fonksiyon, yüzün görüntüde yeterli büyüklükte olup olmadığını puanlar."""
    if Yüz is None:
        return 0.0
    try:
        return float(np.clip(_Mesafe(_Nokta_Al(Yüz, 1), _Nokta_Al(Yüz, 152)) / 0.085, 0.0, 1.0))
    except Exception:
        return 0.0

def Güven_Puanı(Yüz, Sol_Göz_Açıklığı, Sağ_Göz_Açıklığı, hp, Baş_Merkezi, Kare_Kalitesi=1.0) -> float:
    """Bu fonksiyon, baş pozu, göz açıklığı ve kare kalitesinden güven puanı üretir."""
    head_c = 0.20
    if hp is not None:
        cy = Baş_Merkezi[0] if Baş_Merkezi else hp[1]
        cp = Baş_Merkezi[1] if Baş_Merkezi else hp[0]
        yaw_c   = float(np.clip(1.0 - abs(hp[1] - cy) / Baş_Sapma_Ekran_Dışı,   0, 1))
        pitch_c = float(np.clip(1.0 - abs(hp[0] - cp) / Baş_Eğim_Ekran_Dışı, 0, 1))
        head_c  = min(yaw_c, pitch_c)
    ear_avg = ((Sol_Göz_Açıklığı or 0.0) + (Sağ_Göz_Açıklığı or 0.0)) * 0.5
    ear_c   = float(np.clip((ear_avg - Göz_Kırpma_Eşiği) / 0.18, 0, 1))
    size_c  = Yüz_Boyut_Puanı(Yüz) if Yüz is not None else 0.6
    raw = 0.55 * head_c + 0.25 * ear_c + 0.20 * size_c
    cap = 0.75 + 0.25 * float(np.clip(Kare_Kalitesi, 0, 1))
    return float(np.clip(raw, 0, cap))

# Duygu modeli etiketleri ve iç eşlemeleri burada tanımlanır.
def Yüz_Kırpması_Al(Bgr_Kare: np.ndarray, İşaret_Noktaları, Hizala: bool = True):
    """Bu fonksiyon, duygu analizi için yüz bölgesini kırpar ve gerekirse hizalar."""
    if İşaret_Noktaları is None or Bgr_Kare is None:
        return None
    try:
        h, w = Bgr_Kare.shape[:2]
        xs = np.array([lm.x for lm in İşaret_Noktaları], np.float32)
        ys = np.array([lm.y for lm in İşaret_Noktaları], np.float32)
        bw = max(float(xs.max() - xs.min()), 1e-6)
        bh = max(float(ys.max() - ys.min()), 1e-6)
        x0 = int(np.clip((xs.min() - Kesme_Payı * bw) * w, 0, w - 1))
        x1 = int(np.clip((xs.max() + Kesme_Payı * bw) * w, 1, w))
        y0 = int(np.clip((ys.min() - Kesme_Payı * bh) * h, 0, h - 1))
        y1 = int(np.clip((ys.max() + Kesme_Payı * bh) * h, 1, h))
        if x1 <= x0 or y1 <= y0:
            return None
        crop = Bgr_Kare[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return None
        if Hizala:
            a = İşaret_Noktaları[Duygu_Göz_A]; b = İşaret_Noktaları[Duygu_Göz_B]
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

# İki boyutlu Kalman filtresi bakış koordinatını yumuşatır.
class Uyarlanabilir_Kalman_2B:
    def __init__(self, Fps=30.0, q=1e-4, r=5e-3,
                 En_Fazla_Kayıp=6, En_Fazla_Sıçrama=0.22, En_Fazla_Hız=0.10):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self.Fps = float(max(Fps, 1e-6))
        dt = 1.0 / self.Fps
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix   = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
        self.kf.errorCovPost       = np.eye(4, dtype=np.float32)
        self._init       = False
        self.missing     = 0
        self.En_Fazla_Kayıp = int(En_Fazla_Kayıp)
        self.En_Fazla_Sıçrama    = float(En_Fazla_Sıçrama)
        self.En_Fazla_Hız     = float(En_Fazla_Hız)

    def reset(self):
        """Bu fonksiyon, nesnenin geçici durumunu başlangıç değerlerine döndürür."""
        self._init = False
        self.missing = 0

    def _init_at(self, x, y):
        """Bu fonksiyon, Kalman filtresini verilen koordinatta başlatır."""
        st = np.zeros((4, 1), np.float32)
        st[0, 0] = x; st[1, 0] = y
        self.kf.statePost = st; self.kf.statePre = st.copy()
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        self._init = True; self.missing = 0

    def _set_dt(self, dt):
        """Bu fonksiyon, Kalman filtresinin zaman adımını günceller."""
        dt = float(max(dt, 1e-6))
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt

    def step(self, Ölçüm_Koordinatı, Geçerli_Mi: bool, dt=None):
        """Bu fonksiyon, ölçüm geçerliyse filtreyi düzeltir ve değilse tahminle ilerler."""
        if dt is not None:
            self._set_dt(dt)
        if not self._init:
            if Geçerli_Mi and Ölçüm_Koordinatı is not None:
                self._init_at(float(Ölçüm_Koordinatı[0]), float(Ölçüm_Koordinatı[1]))
                return float(Ölçüm_Koordinatı[0]), float(Ölçüm_Koordinatı[1]), False
            return 0.5, 0.5, True

        pred = self.kf.predict()
        px, py = float(pred[0,0]), float(pred[1,0])
        vx, vy = float(pred[2,0]), float(pred[3,0])

        if abs(vx) > self.En_Fazla_Hız * 2.0 or abs(vy) > self.En_Fazla_Hız * 2.0:
            if Geçerli_Mi and Ölçüm_Koordinatı is not None:
                self._init_at(float(Ölçüm_Koordinatı[0]), float(Ölçüm_Koordinatı[1]))
                return float(Ölçüm_Koordinatı[0]), float(Ölçüm_Koordinatı[1]), False
            self.missing += 1
            if self.missing >= self.En_Fazla_Kayıp: self.reset()
            return px, py, True

        if not Geçerli_Mi or Ölçüm_Koordinatı is None:
            self.missing += 1
            if self.missing >= self.En_Fazla_Kayıp: self.reset()
            return px, py, True

        mx, my = float(Ölçüm_Koordinatı[0]), float(Ölçüm_Koordinatı[1])
        if math.hypot(mx - px, my - py) > self.En_Fazla_Sıçrama:
            self._init_at(mx, my)
            return mx, my, False

        est = self.kf.correct(np.array([[mx], [my]], np.float32))
        self.missing = 0
        return float(est[0,0]), float(est[1,0]), False

# Duygu modeli etiketleri ve iç eşlemeleri burada tanımlanır.
class _Duygu_Kalman_1B:
    def __init__(self):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self.x = 0.0; self.P = 1.0

    def update(self, z: float) -> float:
        """Bu fonksiyon, verilen yeni bilgiye göre nesnenin durumunu günceller."""
        P_ = self.P + Duygu_Kalman_Q
        K  = P_ / (P_ + Duygu_Kalman_R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * P_
        return self.x

    def reset(self):
        """Bu fonksiyon, nesnenin geçici durumunu başlangıç değerlerine döndürür."""
        self.x = 0.0; self.P = 1.0

# Baş merkezi izleyici doğal baş pozunu temel değer olarak öğrenir.
class Baş_Merkez_İzleyici:
    """
    Builds an initial baseline from stable forward-facing frames,
    then slowly adapts (handles gradual posture change).
    Gating is relaxed so it still works when face is slightly angled.
    """
    def __init__(self, Başlangıç_Kareleri=60, Alfa=0.01,
                 En_Az_Güven=0.30, İris_Merkez_Eşiği=0.15):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self.Başlangıç_Kareleri    = int(Başlangıç_Kareleri)
        self.Alfa          = float(Alfa)
        self.En_Az_Güven       = float(En_Az_Güven)
        self.İris_Merkez_Eşiği = float(İris_Merkez_Eşiği)  # Eski değer 0.10 idi ve daha esnek çalışması için artırıldı.
        self._buf  = []
        self._ready = False
        self.yaw   = 0.0
        self.pitch = 0.0

    def reset(self):
        """Bu fonksiyon, nesnenin geçici durumunu başlangıç değerlerine döndürür."""
        self._buf.clear(); self._ready = False
        self.yaw = 0.0;    self.pitch  = 0.0

    def _gate(self, hp, İris_Koordinatı, Güven, Kırpma_Kötü_Mü) -> bool:
        """Bu fonksiyon, baş merkezi öğrenimine uygun kareleri seçer."""
        if hp is None or Kırpma_Kötü_Mü or Güven < self.En_Az_Güven:
            return False
        if İris_Koordinatı is not None:
            if abs(İris_Koordinatı[0] - 0.5) > self.İris_Merkez_Eşiği:
                return False
            if abs(İris_Koordinatı[1] - 0.5) > self.İris_Merkez_Eşiği:
                return False
        return True

    def update(self, hp, İris_Koordinatı, Güven, Kırpma_Kötü_Mü):
        """Bu fonksiyon, verilen yeni bilgiye göre nesnenin durumunu günceller."""
        if not self._gate(hp, İris_Koordinatı, Güven, Kırpma_Kötü_Mü):
            return
        pitch, yaw = float(hp[0]), float(hp[1])
        if not self._ready:
            self._buf.append((yaw, pitch))
            if len(self._buf) >= self.Başlangıç_Kareleri:
                self.yaw   = float(np.median([v[0] for v in self._buf]))
                self.pitch = float(np.median([v[1] for v in self._buf]))
                self._ready = True
            return
        a = self.Alfa
        self.yaw   = (1 - a) * self.yaw   + a * yaw
        self.pitch = (1 - a) * self.pitch + a * pitch

    def center(self):
        """Bu fonksiyon, öğrenilen baş merkezi hazırsa sapma ve eğim değerini döndürür."""
        return (self.yaw, self.pitch) if self._ready else None

# Baş ve göz hareketleri tek bakış sonucunda birleştirilir.
class Baş_Göz_Birleştirici:
    def __init__(self, head_tracker: Baş_Merkez_İzleyici):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self._ht  = head_tracker
        self._sm_x = 0.5;  self._sm_y = 0.5

    def reset(self):
        """Bu fonksiyon, nesnenin geçici durumunu başlangıç değerlerine döndürür."""
        self._sm_x = 0.5;  self._sm_y = 0.5

    def reset_to(self, x, y):
        """Bu fonksiyon, birleşim yumuşatma değerlerini verilen koordinata ayarlar."""
        self._sm_x = float(x);  self._sm_y = float(y)

    def fuse(self, Ham_Ekran_Koordinatı, hp):
        """Bu fonksiyon, göz koordinatı ile baş pozunu tek ekran koordinatında birleştirir."""
        if Ham_Ekran_Koordinatı is None:
            return None, 0.0
        ix, iy = Ham_Ekran_Koordinatı
        if hp is None:
            self._sm_x = self._sm_x * (1 - Birleşim_Ema_Alfa) + ix * Birleşim_Ema_Alfa
            self._sm_y = self._sm_y * (1 - Birleşim_Ema_Alfa) + iy * Birleşim_Ema_Alfa
            return (self._sm_x, self._sm_y), 0.0

        center = self._ht.center()
        cy = center[0] if center else hp[1]
        cp = center[1] if center else hp[0]
        ry = hp[1] - cy;  rp = hp[0] - cp

        hsx = 0.5 + ry * Baş_Sapma_Ölçeği
        hsy = 0.5 + rp * Baş_Eğim_Ölçeği

        yw = float(np.clip((abs(ry) - Baş_Sapma_Yumuşak)   / max(Baş_Sapma_Sert   - Baş_Sapma_Yumuşak,   1), 0, 1))
        pw = float(np.clip((abs(rp) - Baş_Eğim_Yumuşak) / max(Baş_Eğim_Sert - Baş_Eğim_Yumuşak, 1), 0, 1))
        hw = max(yw, pw)

        rawx = ix * (1 - hw) + hsx * hw
        rawy = iy * (1 - hw) + hsy * hw

        self._sm_x = self._sm_x * (1 - Birleşim_Ema_Alfa) + rawx * Birleşim_Ema_Alfa
        self._sm_y = self._sm_y * (1 - Birleşim_Ema_Alfa) + rawy * Birleşim_Ema_Alfa
        return (self._sm_x, self._sm_y), hw

# Yön oylayıcı kısa süreli titreşimleri azaltır.
class Yön_Oylayıcı:
    def __init__(self, Pencere=Oylama_Penceresi, Çoğunluk=Oylama_Çoğunluğu):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self._q   = collections.deque(maxlen=Pencere)
        self._maj = float(Çoğunluk)
        self.last = "CENTER"

    def reset(self):
        """Bu fonksiyon, nesnenin geçici durumunu başlangıç değerlerine döndürür."""
        self._q.clear(); self.last = "CENTER"

    def seed(self, Yön):
        """Bu fonksiyon, yön oylayıcıyı verilen başlangıç yönüyle doldurur."""
        self._q.clear()
        for _ in range(max(3, self._q.maxlen // 2)):
            self._q.append(Yön)
        self.last = Yön

    def push(self, Yön) -> str:
        """Bu fonksiyon, yeni yönü oylamaya ekler ve kararlı yönü döndürür."""
        self._q.append(Yön)
        cnt = collections.Counter(self._q)
        top, n = cnt.most_common(1)[0]
        if len(self._q) >= 3 and n / len(self._q) >= self._maj:
            self.last = top
        return self.last

# Ekran dışı izleyici kullanıcının ekrandan uzak bakmasını takip eder.
class Ekran_Dışı_İzleyici:
    def __init__(self, Bekleme=Ekran_Dışı_Bekleme, Olay_Fonksiyonu=None):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self.Bekleme     = float(Bekleme)
        self.Olay_Fonksiyonu = Olay_Fonksiyonu
        self._since   = None
        self._active  = False
        self._last    = "CENTER"
        self.dur      = 0.0

    def update(self, Ekran_Dışı_Mı, Yön, Güven):
        """Bu fonksiyon, verilen yeni bilgiye göre nesnenin durumunu günceller."""
        now = time.time()
        just_returned = False
        if Ekran_Dışı_Mı:
            self._last = Yön
            if self._since is None: self._since = now
            self.dur = now - self._since
            if self.dur >= self.Bekleme and not self._active:
                self._active = True
                if self.Olay_Fonksiyonu:
                    self.Olay_Fonksiyonu("OFFSCREEN_START", self._last, 0.0, Güven, now)
        else:
            if self._active and self._since is not None:
                dur = now - self._since
                if self.Olay_Fonksiyonu:
                    self.Olay_Fonksiyonu("OFFSCREEN_END", self._last, dur, Güven, now)
                just_returned = True
            self._active = False; self._since = None; self.dur = 0.0
        return self._active, just_returned

    def force_close(self):
        """Bu fonksiyon, açık ekran dışı olayını güvenli şekilde kapatır."""
        if self._active and self._since is not None:
            dur = time.time() - self._since
            if self.Olay_Fonksiyonu:
                self.Olay_Fonksiyonu("OFFSCREEN_END", self._last, dur, 0.0, time.time())
            self._active = False; self._since = None; self.dur = 0.0

# Yüz imzası kullanıcı yüzünü oranlarla temsil eder.
_İmza_Noktaları = {
    "L_EYE_OUT": 33,   "R_EYE_OUT":  263,
    "FOREHEAD":  10,   "CHIN":       152,
    "L_CHEEK":   234,  "R_CHEEK":    454,
    "MOUTH_L":   61,   "MOUTH_R":    291,
    "NOSE_TIP":  4,    "NOSE_BASE":  2,
    "L_BROW":    70,   "R_BROW":     300,
    "UPPER_LIP": 13,   "LOWER_LIP":  14,
}

def Yüz_İmzası(Yüz) -> np.ndarray:
    """Bu fonksiyon, yüzü poz değişimine dayanıklı oranlardan oluşan imzaya dönüştürür."""
    if Yüz is None:
        return None
    try:
        p_le  = _Nokta_Al(Yüz, _İmza_Noktaları["L_EYE_OUT"])
        p_re  = _Nokta_Al(Yüz, _İmza_Noktaları["R_EYE_OUT"])
        p_top = _Nokta_Al(Yüz, _İmza_Noktaları["FOREHEAD"])
        p_ch  = _Nokta_Al(Yüz, _İmza_Noktaları["CHIN"])
        p_lc  = _Nokta_Al(Yüz, _İmza_Noktaları["L_CHEEK"])
        p_rc  = _Nokta_Al(Yüz, _İmza_Noktaları["R_CHEEK"])
        p_ml  = _Nokta_Al(Yüz, _İmza_Noktaları["MOUTH_L"])
        p_mr  = _Nokta_Al(Yüz, _İmza_Noktaları["MOUTH_R"])
        p_nt  = _Nokta_Al(Yüz, _İmza_Noktaları["NOSE_TIP"])
        p_nb  = _Nokta_Al(Yüz, _İmza_Noktaları["NOSE_BASE"])
        p_lb  = _Nokta_Al(Yüz, _İmza_Noktaları["L_BROW"])
        p_rb  = _Nokta_Al(Yüz, _İmza_Noktaları["R_BROW"])
        p_ul  = _Nokta_Al(Yüz, _İmza_Noktaları["UPPER_LIP"])
        p_ll  = _Nokta_Al(Yüz, _İmza_Noktaları["LOWER_LIP"])

        d_io  = _Mesafe(p_le, p_re)    # İki göz dış noktası arasındaki mesafedir.
        d_fh  = _Mesafe(p_top, p_ch)   # Yüz yüksekliğini temsil eder.
        d_fw  = _Mesafe(p_lc, p_rc)    # Yüz genişliğini temsil eder.
        d_mw  = _Mesafe(p_ml, p_mr)    # Ağız genişliğini temsil eder.
        d_nw  = _Mesafe(p_nt, p_nb)    # Burun uzunluğunu temsil eder.
        d_bw  = _Mesafe(p_lb, p_rb)    # Kaş açıklığını temsil eder.
        d_lip = _Mesafe(p_ul, p_ll)    # Dudak açıklığını temsil eder.
        # Bu satır ilgili ayarın öğrenci tarafından daha kolay anlaşılmasını sağlar.
        eye_mid = ((p_le[0]+p_re[0])*0.5, (p_le[1]+p_re[1])*0.5)
        d_en  = _Mesafe(eye_mid, p_nt)

        eps = 1e-6
        İmza = np.array([
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
        return İmza
    except Exception:
        return None


class İmza_İzleyici:
    """Bu sınıf, yüz imzalarını kısa bir pencere boyunca ortalayıp kararlı hale getirir."""
    def __init__(self, Pencere=İmza_Penceresi):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self._q = collections.deque(maxlen=int(Pencere))

    def reset(self):
        """Bu fonksiyon, nesnenin geçici durumunu başlangıç değerlerine döndürür."""
        self._q.clear()

    def push(self, İmza: np.ndarray):
        """Bu fonksiyon, yeni yönü oylamaya ekler ve kararlı yönü döndürür."""
        if İmza is None:
            return None
        self._q.append(np.asarray(İmza, np.float32))
        if len(self._q) < 8:
            return None
        return np.mean(np.stack(self._q, axis=0), axis=0)

# Kalibrasyon verisi ekrandaki bakış haritasını saklar.
class Afin_Haritalayıcı:
    def __init__(self):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self.A     = None;  self.ready = False
        self.th_h  = Bölge_İç_Yatay_Varsayılan
        self.th_v  = Bölge_İç_Dikey_Varsayılan
        self._src  = [];    self._dst  = []

    def reset(self):
        """Bu fonksiyon, nesnenin geçici durumunu başlangıç değerlerine döndürür."""
        self.A = None;  self.ready = False
        self.th_h = Bölge_İç_Yatay_Varsayılan
        self.th_v = Bölge_İç_Dikey_Varsayılan
        self._src.clear();  self._dst.clear()

    def add(self, İris_Koordinatı, Ekran_Koordinatı):
        """Bu fonksiyon, kalibrasyon için iris ve ekran noktası çiftini ekler."""
        self._src.append(İris_Koordinatı);  self._dst.append(Ekran_Koordinatı)

    def fit(self) -> bool:
        """Bu fonksiyon, toplanan kalibrasyon noktalarından afin dönüşüm matrisini öğrenir."""
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

    def map(self, İris_Koordinatı, clip=True):
        """Bu fonksiyon, iris koordinatını öğrenilmiş afin dönüşümle ekran koordinatına çevirir."""
        if not self.ready or self.A is None or İris_Koordinatı is None:
            return İris_Koordinatı
        v = np.array([İris_Koordinatı[0], İris_Koordinatı[1], 1.0], np.float32)
        r = v @ self.A
        x, y = float(r[0]), float(r[1])
        if clip:
            x = float(np.clip(x, 0, 1))
            y = float(np.clip(y, 0, 1))
        return x, y

    def save_single(self, path=None):
        """Bu fonksiyon, tek kalibrasyon profilini dosyaya kaydeder."""
        if path is None:
            path = Kalibrasyon_Dosyası
        if self.A is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, A=self.A, th_h=[self.th_h], th_v=[self.th_v])

    def load_single(self, path=None) -> bool:
        """Bu fonksiyon, tek kalibrasyon profilini dosyadan yükler."""
        if path is None:
            path = Kalibrasyon_Dosyası
        try:
            d = np.load(path, allow_pickle=True)
            if "A" not in d:
                return False
            self.A = d["A"].astype(np.float32)
            self.th_h = float(d["th_h"][0]) if "th_h" in d else Bölge_İç_Yatay_Varsayılan
            self.th_v = float(d["th_v"][0]) if "th_v" in d else Bölge_İç_Dikey_Varsayılan
            self.ready = True
            return True
        except Exception:
            return False



class Profil_Deposu:
    K_SIG = 12   # Yüz_İmzası çıktısının uzunluğuyla aynı olmalıdır.

    def __init__(self, path=Kalibrasyon_Dosyası):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self.path  = path
        self.sigs  = np.zeros((0, self.K_SIG), np.float32)
        self.As    = np.zeros((0, 3, 2),        np.float32)
        self.th_h  = np.zeros((0,),             np.float32)
        self.th_v  = np.zeros((0,),             np.float32)
        self.ts    = np.zeros((0,),             np.float64)
        self._load()

    def _load(self):
        """Bu fonksiyon, kayıtlı profil verilerini dosyadan okumaya çalışır."""
        if not os.path.exists(self.path):
            return
        try:
            d = np.load(self.path, allow_pickle=True)
            if "sigs" in d and "As" in d:
                sigs = d["sigs"].astype(np.float32)
                # Bu satır ilgili ayarın öğrenci tarafından daha kolay anlaşılmasını sağlar.
                if sigs.shape[1] != self.K_SIG:
                    # Bu satır ilgili ayarın öğrenci tarafından daha kolay anlaşılmasını sağlar.
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
            # Bu satır ilgili ayarın öğrenci tarafından daha kolay anlaşılmasını sağlar.
            if "A" in d:
                A    = d["A"].astype(np.float32).reshape(3, 2)
                th_h = float(d["th_h"][0]) if "th_h" in d else Bölge_İç_Yatay_Varsayılan
                th_v = float(d["th_v"][0]) if "th_v" in d else Bölge_İç_Dikey_Varsayılan
                self.sigs = np.zeros((1, self.K_SIG), np.float32)
                self.As   = A.reshape(1, 3, 2)
                self.th_h = np.array([th_h], np.float32)
                self.th_v = np.array([th_v], np.float32)
                self.ts   = np.array([time.time()], np.float64)
                self._save()
        except Exception:
            pass

    def _save(self):
        """Bu fonksiyon, profil verilerini dosyaya yazar."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        np.savez(self.path, sigs=self.sigs, As=self.As,
                 th_h=self.th_h, th_v=self.th_v, ts=self.ts)

    def count(self) -> int:
        """Bu fonksiyon, kayıtlı profil sayısını döndürür."""
        return int(self.As.shape[0])

    def clear(self):
        """Bu fonksiyon, kayıtlı profilleri ve dosyadaki verileri temizler."""
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

    def match(self, İmza: np.ndarray, Eşik=İmza_Eşleşme_Eşiği):
        """Bu fonksiyon, verilen yüz imzasını kayıtlı profillerle karşılaştırır."""
        if İmza is None or self.count() == 0:
            return None, None
        Geçerli_Mi = np.linalg.norm(self.sigs, axis=1) > 1e-6
        if not np.any(Geçerli_Mi):
            return None, None
        dif  = self.sigs[Geçerli_Mi] - İmza.reshape(1, -1)
        ds   = np.sqrt(np.sum(dif * dif, axis=1))
        j    = int(np.argmin(ds))
        dmin = float(ds[j])
        idxs = np.flatnonzero(Geçerli_Mi)
        İndeks  = int(idxs[j])
        return (İndeks, dmin) if dmin <= Eşik else (None, dmin)

    def load_mapper(self, İndeks: int) -> Afin_Haritalayıcı:
        """Bu fonksiyon, seçilen profile ait afin haritalayıcıyı hazırlar."""
        m = Afin_Haritalayıcı()
        İndeks = int(np.clip(İndeks, 0, max(0, self.count() - 1)))
        m.A    = self.As[İndeks].astype(np.float32)
        m.th_h = float(self.th_h[İndeks])
        m.th_v = float(self.th_v[İndeks])
        m.ready = True
        return m

    def upsert(self, İmza: np.ndarray, Haritalayıcı: Afin_Haritalayıcı,
               Eşik=İmza_Kayıt_Eşiği):
        """Bu fonksiyon, profil varsa günceller yoksa yeni profil olarak ekler."""
        if İmza is None or Haritalayıcı is None or not Haritalayıcı.ready or Haritalayıcı.A is None:
            return None
        İmza = İmza.astype(np.float32).reshape(1, -1)
        # Bu satır ilgili ayarın öğrenci tarafından daha kolay anlaşılmasını sağlar.
        if İmza.shape[1] != self.K_SIG:
            ns = np.zeros((1, self.K_SIG), np.float32)
            ns[:, :min(İmza.shape[1], self.K_SIG)] = İmza[:, :self.K_SIG]
            İmza = ns
        İndeks, d = self.match(İmza[0], Eşik=Eşik)
        if İndeks is None:
            self.sigs = np.vstack([self.sigs, İmza])
            self.As   = np.concatenate([self.As,   Haritalayıcı.A.reshape(1,3,2).astype(np.float32)], axis=0)
            self.th_h = np.append(self.th_h, np.array([Haritalayıcı.th_h], np.float32))
            self.th_v = np.append(self.th_v, np.array([Haritalayıcı.th_v], np.float32))
            self.ts   = np.append(self.ts,   np.array([time.time()], np.float64))
            self._save()
            return self.count() - 1
        self.sigs[İndeks]  = İmza[0]
        self.As[İndeks]    = Haritalayıcı.A.astype(np.float32).reshape(3, 2)
        self.th_h[İndeks]  = float(Haritalayıcı.th_h)
        self.th_v[İndeks]  = float(Haritalayıcı.th_v)
        self.ts[İndeks]    = float(time.time())
        self._save()
        return İndeks


class Duygu_İzleyici:
    def __init__(self, Model_İsim=Model_İsim, Motor=Model_Motor):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self._lock            = threading.Lock()
        self._kalman          = {e: _Duygu_Kalman_1B() for e in Duygular}
        self._ema             = {e: 0.0 for e in Duygular}
        self._raw_scores      = {e: 0.0 for e in Duygular}
        self._pending         = None
        self._candidate       = "neutral"
        self._candidate_count = 0
        self._stable          = "neutral"
        self.smoothed         = {e: 0.0 for e in Duygular}
        self.dominant         = "neutral"
        self.available        = False
        self.analysis_fps     = 0.0
        self._running         = True
        self._model_name      = Model_İsim
        self._engine          = Motor
        self._thread          = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def feed(self, Bgr_Yüz_Kırpması: np.ndarray):
        """Bu fonksiyon, duygu analizi kuyruğuna yeni yüz kırpmasını verir."""
        if Bgr_Yüz_Kırpması is None or Bgr_Yüz_Kırpması.size == 0:
            return
        with self._lock:
            self._pending = Bgr_Yüz_Kırpması

    def _loop(self):
        """Bu fonksiyon, arka planda duygu modelini çalıştırıp ham skorları üretir."""
        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
            rec = EmotiEffLibRecognizer(Motor=self._engine,
                                        Model_İsim=self._model_name,
                                        device="cpu")
            self.available = True
        except Exception:
            return  # Kütüphane yoksa sessizce geri dönüş yapılır.

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
                for i, lab in enumerate(Duygu_Etiketleri):
                    k = Duygu_İç_Eşleme.get(lab)
                    if k and i < len(probs):
                        raw[k] = float(probs[i])
                with self._lock:
                    for e in Duygular:
                        self._raw_scores[e] = raw.get(e, 0.0)
                    self.analysis_fps = 1.0 / max(time.perf_counter() - t0, 1e-9)
            except Exception:
                pass

    def tick(self):
        """Bu fonksiyon, her video karesinde duygu skorlarını yumuşatır ve baskın duyguyu seçer."""
        if not self.available:
            return dict(self.smoothed), self.dominant
        with self._lock:
            raw = dict(self._raw_scores)
        total = max(1e-9, sum(raw.values()))
        norm  = {e: raw[e] / total for e in Duygular}
        k_out = {e: self._kalman[e].update(norm[e]) for e in Duygular}
        for e in Duygular:
            self._ema[e] = Duygu_Ema_Alfa * k_out[e] + (1 - Duygu_Ema_Alfa) * self._ema[e]
        s = max(1e-9, sum(self._ema.values()))
        for e in Duygular:
            self.smoothed[e] = self._ema[e] / s * 100.0

        cand = max(self.smoothed, key=self.smoothed.get)
        if cand == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate       = cand
            self._candidate_count = 1
        if self._candidate_count >= Duygu_Kararlılık_Karesi:
            self._stable = self._candidate
        self.dominant = self._stable
        return dict(self.smoothed), self.dominant

    def reset(self):
        """Bu fonksiyon, nesnenin geçici durumunu başlangıç değerlerine döndürür."""
        for e in Duygular:
            self._kalman[e].reset()
            self._ema[e] = 0.0
        self._candidate       = "neutral"
        self._candidate_count = 0
        self._stable          = "neutral"
        with self._lock:
            self._raw_scores = {e: 0.0 for e in Duygular}

    def stop(self):
        """Bu fonksiyon, duygu analiz iş parçacığını durdurur."""
        self._running = False
        try: self._thread.join(timeout=0.5)
        except Exception: pass

# Arayüz sınıfı kalibrasyon ekranını yönetir.
class Kalibrasyon_Arayüzü:
    def __init__(self, Etiket="Calibration"):
        """Bu fonksiyon, sınıfın başlangıç değerlerini kurar."""
        self.Etiket = Etiket
        self.reset()

    def reset(self):
        """Bu fonksiyon, nesnenin geçici durumunu başlangıç değerlerine döndürür."""
        self.İndeks   = 0;    self.start = None
        self.samps = [];   self.wait  = True
        self.done  = False
        self._mapper = Afin_Haritalayıcı()

    def update(self, İris_Koordinatı, Güven, frame, Yüz_Uygun_Mu) -> bool:
        """Bu fonksiyon, verilen yeni bilgiye göre nesnenin durumunu günceller."""
        pt    = Kalibrasyon_Noktaları[self.İndeks]
        Etiket = Kalibrasyon_Etiketleri[self.İndeks]
        now   = time.time()
        if self.start is None:
            self.start = now;  self.samps = [];  self.wait = True

        if self.wait:
            prog = 0.0
            if now - self.start >= Kalibrasyon_Bekleme_Suresi:
                self.wait = False;  self.start = now
        else:
            if İris_Koordinatı is not None and Güven > 0.28:
                self.samps.append(İris_Koordinatı)
            prog = min(1.0, len(self.samps) / Kalibrasyon_Kare_Sayısı)
            if len(self.samps) >= Kalibrasyon_Kare_Sayısı:
                mx = float(np.mean([p[0] for p in self.samps]))
                my = float(np.mean([p[1] for p in self.samps]))
                self._mapper.add((mx, my), pt)
                self.İndeks += 1;  self.start = None
                if self.İndeks >= len(Kalibrasyon_Noktaları):
                    if self._mapper.fit():
                        self.done = True
                        return True

        _Kalibrasyon_Çiz(frame, pt, Etiket, self.wait, prog, self.İndeks, Yüz_Uygun_Mu, self.Etiket)
        return False

    def get_mapper(self) -> Afin_Haritalayıcı:
        """Bu fonksiyon, kalibrasyon sonunda oluşan haritalayıcıyı döndürür."""
        return self._mapper


def _Kalibrasyon_Çiz(frame, pt, Etiket, wait, prog, İndeks, Yüz_Uygun_Mu, Başlık):
    """Bu fonksiyon, kalibrasyon hedefini ve ilerleme bilgisini ekrana çizer."""
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
    cv2.putText(frame, Etiket,  (px-40, py-66), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    cv2.putText(frame, "Face OK" if Yüz_Uygun_Mu else "NO FACE", (50, H-16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,0) if Yüz_Uygun_Mu else (0,40,220), 2)
    cv2.putText(frame, f"Point {İndeks+1}/{len(Kalibrasyon_Noktaları)}", (W//2-120, H-16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 1)
    cv2.putText(frame, Başlık,  (W//2-180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100,220,255), 2)


def Gösterge_Çiz(frame, Kararlı_Yön, Güven, gp, Fps, Ekran_Dışı_Aktif, Ekran_Dışı_Süre,
             Birleşim_Ağırlığı, Işık_Yazısı, Mod="RUN", Kullanıcı_Yazısı="",
             Duygu_Etiketi="", Duygu_Uygun_Mu=False):
    """Bu fonksiyon, kamera görüntüsünün üzerine yön, güven, ışık ve duygu bilgisini çizer."""
    H, W = frame.shape[:2]
    col     = Yön_Renkleri.get(Kararlı_Yön, (180,180,180))
    top_col = (0, 0, 50) if not Ekran_Dışı_Aktif else (70, 0, 0)
    cv2.rectangle(frame, (0,0), (W,92), top_col, -1)

    if Mod == "KALIB":
        lbl = "KALIBRASYON"
    else:
        lbl = f"OFF-SCREEN ({Ekran_Dışı_Süre:.1f}s)" if Ekran_Dışı_Aktif else f"{Kararlı_Yön} [{Güven:.0%}]"

    (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
    cv2.putText(frame, lbl, ((W-tw)//2, 84), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,0,0), 4)
    cv2.putText(frame, lbl, ((W-tw)//2, 84), cv2.FONT_HERSHEY_DUPLEX, 0.75, col, 2)

    cv2.putText(frame, f"FPS:{Fps:.0f}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160,160,160), 1)

    # Duygu modeli etiketleri ve iç eşlemeleri burada tanımlanır.
    if Duygu_Uygun_Mu and Duygu_Etiketi:
        em_col = Duygu_Bilgileri.get(Duygu_Etiketi, {}).get("color", (180,180,180))
        em_lbl = Duygu_Bilgileri.get(Duygu_Etiketi, {}).get("label", Duygu_Etiketi).upper()
        cv2.putText(frame, f"EMOTION: {em_lbl}", (8,48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, em_col, 2)

    if gp:
        cv2.putText(frame, f"EYE P:{gp[0]:+.0f} Y:{gp[1]:+.0f} R:{gp[2]:+.0f}",
                    (8,72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130,130,130), 1)

    cv2.putText(frame, Işık_Yazısı, (W-230, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,180,100), 1)

    if Ekran_Dışı_Aktif:
        for t in range(1,6):
            cv2.rectangle(frame, (t,t), (W-t,H-t), (0,0,200), 1)


Model_Url = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

def Modeli_Hazırla():
    """Bu fonksiyon, yüz işaretleme modeli yoksa indirip kullanılabilir hale getirir."""
    if os.path.exists(Model_Yolu) and os.path.getsize(Model_Yolu) > 1_000_000:
        return
    print(f"[gaze] face_landmarker.task bulunamadı, indiriliyor...\n  → {Model_Url}")
    try:
        import urllib.request
        tmp = Model_Yolu + ".tmp"
        os.makedirs(os.path.dirname(Model_Yolu) or ".", exist_ok=True)

        def _progress(block_num, block_size, total_size):
            """Bu fonksiyon, model indirme yüzdesini ekrana yazdırır."""
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                print(f"\r  indiriliyor... {pct}%", end="", flush=True)

        urllib.request.urlretrieve(Model_Url, tmp, _progress)
        print()  # İndirme ilerlemesinden sonra yeni satır basılır.
        os.replace(tmp, Model_Yolu)
        print(f"[gaze] Model indirildi: {Model_Yolu}")
    except Exception as e:
        # Bu satır ilgili ayarın öğrenci tarafından daha kolay anlaşılmasını sağlar.
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise RuntimeError(
            f"Model indirilemedi: {e}\n"
            f"Manuel olarak şu adresten indirip '{Model_Yolu}' konumuna koyun:\n"
            f"  {Model_Url}"
        ) from e

def Algılayıcı_Kur():
    """Bu fonksiyon, MediaPipe yüz işaretleyici algılayıcısını verilen ayarlarla oluşturur."""
    opts = Mp_Vision.FaceLandmarkerOptions(
        base_options=Mp_Python.BaseOptions(model_asset_path=Model_Yolu),
        running_mode=Mp_Vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.40,
        min_face_presence_confidence=0.40,
        min_tracking_confidence=0.40,
    )
    return Mp_Vision.FaceLandmarker.create_from_options(opts)
