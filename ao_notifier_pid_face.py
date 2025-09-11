# ao_notifier_pid_face.py (DeepSORT embedder = torchreid + OSNet)
import os
import cv2
import time
import json
import argparse
from datetime import datetime
from typing import Optional, Tuple, Dict

import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import google.generativeai as genai
from PIL import Image

# InsightFace (deteksi + embedding wajah)
from insightface.app import FaceAnalysis


# ---------- Args ----------
def parse_args():
    ap = argparse.ArgumentParser(
        "AO Notifier (DeepSORT + TorchReID/OSNet + Stable PID + Face Recognition + Strict Ownership + Paired Snapshots)"
    )
    ap.add_argument("--source", type=str, default="0",
                    help="0/1 untuk webcam, path video, atau RTSP/MJPEG URL")
    ap.add_argument("--model", type=str, default="yolov8n.pt",
                    help="Model YOLO (COCO) – auto-download")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence minimum")

    ap.add_argument("--r_attend", type=int, default=160, help="Radius ATTEND (px)")
    ap.add_argument("--t_attend", type=float, default=1.0, help="Durasi ATTEND (detik)")
    ap.add_argument("--r_unattend", type=int, default=260, help="Radius UNATTEND (px)")
    ap.add_argument("--t_unattend", type=float, default=3.0, help="Durasi UNATTEND (detik)")
    ap.add_argument("--hysteresis_gap", type=int, default=60,
                    help="Jarak minimal antara R_UNATTEND dan R_ATTEND (px)")
    ap.add_argument("--grace_after_attended", type=float, default=1.0,
                    help="Grace time setelah attended sebelum boleh jadi unattended (detik)")
    ap.add_argument("--min_area_ratio", type=float, default=0.0002,
                    help="Buang bbox area < rasio ini terhadap frame")
    ap.add_argument("--max_area_ratio", type=float, default=0.25,
                    help="Buang bbox area > rasio ini terhadap frame")

    ap.add_argument("--out_dir", type=str, default="ao_snaps", help="Folder snapshot")
    ap.add_argument("--save_video", action="store_true", help="Simpan video beranotasi")
    ap.add_argument("--window", type=str, default="Abandon Object (OSNet + Face)", help="Nama window")
    ap.add_argument("--index_file", type=str, default="ao_index.json", help="Nama file index JSON")
    ap.add_argument("--attended_mode", type=str, default="first",
                    choices=["first", "last"],
                    help="Gunakan ATTENDED pertama ('first') atau terakhir ('last') saat menyimpan pasangan.")

    # Face matcher params
    ap.add_argument("--face_device", type=str, default="cpu", choices=["cpu", "gpu"], help="Device untuk insightface")
    ap.add_argument("--face_sample_every", type=int, default=5, help="Sampling identitas tiap N frame")
    ap.add_argument("--face_thresh", type=float, default=0.40, help="Threshold cosine match facebank (0.38–0.45)")

    # PID manager params
    ap.add_argument("--pid_ttl", type=float, default=4.0, help="TTL PID (detik) saat orang hilang sementara")
    ap.add_argument("--pid_max_dist", type=int, default=140, help="Maks jarak pusat bbox untuk re-associate PID (px)")
    ap.add_argument("--pid_min_iou", type=float, default=0.12, help="Min IoU untuk re-associate PID")

    # TorchReID / DeepSORT params
    ap.add_argument("--reid_device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device untuk TorchReID")
    ap.add_argument("--reid_model", type=str, default="osnet_x0_25", help="Nama model TorchReID (mis. osnet_x0_25)")
    ap.add_argument("--reid_wts", type=str, default="", help="Path weight ReID (kosongkan untuk auto)")
    ap.add_argument("--reid_max_age", type=int, default=40, help="DeepSORT max_age (ketahanan track)")
    ap.add_argument("--reid_max_cos", type=float, default=0.20, help="DeepSORT max_cosine_distance (ketat=lebih kecil)")
    ap.add_argument("--reid_nn_budget", type=int, default=100, help="DeepSORT nn_budget (gallery size)")

    return ap.parse_args()


# ---------- Utils ----------
def is_person(name: str) -> bool:
    return name == "person"


def bbox_min_distance(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
    dy = max(0.0, max(by1 - ay2, ay1 - by2)) 
    return (dx*dx + dy*dy) ** 0.5



def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)  # fixed
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clamp_bbox(box, W, H):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    return x1, y1, x2, y2, (x2 > x1 and y2 > y1)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_index(path):
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"pairs": {}}  # key: "ownerPID{pid}_obj{obj_id}"


def save_index(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def analyze_with_gemini(image_path: str):
    """
    Kirim crop objek ke Gemini, return dict JSON
    dengan kunci: {"nama_objek": "...", "kategori": "barang pribadi"}
    NOTE: gunakan GOOGLE_API_KEY dari environment.
    """
    try:
        api_key = "AIzaSyDiMY2xY0N_eOw5vUzk-J3sLVDb81TEfS8"
        if not api_key:
            print("[GEMINI-ERROR] GOOGLE_API_KEY tidak ditemukan (export GOOGLE_API_KEY=...)")
            return None
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        img = Image.open(image_path)

        prompt = (
            "Analisis gambar ini. Tentukan apakah objek adalah barang pribadi. "
            "Barang pribadi meliputi interaksi objek yang jelas dipegang/dipakai/dikenakan orang "
            "(misal: tas digendong, koper ditenteng, botol dibawa, laptop dibawa, jaket dipakai) "
            "atau fasilitas umum (misal: kursi, meja, pagar, tong sampah). "
            "Jawab HANYA JSON valid dengan format: "
            "{\"nama_objek\": \"...\", \"kategori\": \"barang pribadi\"}"
        )

        response = model.generate_content([prompt, img])
        cleaned = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned)
    except Exception as e:
        print(f"[GEMINI-ERROR] {e}")
        return None


# ---------- FaceBank & FaceMatcher ----------
class FaceBank:
    """
    Muat 1 foto per orang dari folder images/,
    filename tanpa ekstensi dipakai sebagai label (identity).
    """
    def __init__(self, app: FaceAnalysis, folder="images"):
        self.app = app
        self.folder = folder
        self.labels = []
        self.embeddings = np.zeros((0, 512), np.float32)
        self._load()

    def _load(self):
        if not os.path.isdir(self.folder):
            print(f"[FACEBANK] Folder '{self.folder}' tidak ada, skip.")
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        embs = []
        labels = []
        for fn in os.listdir(self.folder):
            path = os.path.join(self.folder, fn)
            root, ext = os.path.splitext(fn.lower())
            if ext not in exts or not os.path.isfile(path):
                continue
            label = os.path.splitext(fn)[0]
            try:
                img = cv2.imread(path)
                if img is None:
                    print(f"[FACEBANK] Gagal load {path}")
                    continue
                faces = self.app.get(img)
                if not faces:
                    print(f"[FACEBANK] Tidak ada wajah pada {path}")
                    continue
                face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
                emb = face.normed_embedding
                if emb is None:
                    print(f"[FACEBANK] Embedding gagal untuk {path}")
                    continue
                labels.append(label)
                embs.append(emb.astype(np.float32))
                print(f"[FACEBANK] + {label}")
            except Exception as e:
                print(f"[FACEBANK] Error {path}: {e}")
        if embs:
            self.labels = labels
            self.embeddings = np.stack(embs).astype(np.float32)

    def match(self, emb: np.ndarray, thresh: float = 0.40) -> Optional[Tuple[str, float]]:
        if emb is None or self.embeddings.shape[0] == 0:
            return None
        sims = self.embeddings @ emb  # normed embeddings → dot == cosine
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= thresh:
            return (self.labels[idx], score)
        return None


class FaceMatcher:
    """
    Deteksi wajah dalam bbox person → ambil embedding → cocokin ke FaceBank.
    Untuk efisiensi, sampling tiap N frame per track.
    """
    def __init__(self, device="cpu", sample_every=5, bank_thresh=0.40):
        providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0 if device != "cpu" else -1, det_size=(640, 640))
        self.facebank = FaceBank(self.app, folder="images")
        self.sample_every = max(1, int(sample_every))
        self.bank_thresh = float(bank_thresh)
        self._tick = 0

    def maybe_identify(self, frame: np.ndarray, person_box) -> Optional[Tuple[str, float]]:
        self._tick += 1
        if (self._tick % self.sample_every) != 0:
            return None
        x1, y1, x2, y2 = map(int, person_box)
        x1, y1, x2, y2, ok = clamp_bbox([x1, y1, x2, y2], frame.shape[1], frame.shape[0])
        if not ok:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        faces = self.app.get(crop)
        if not faces:
            return None
        face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
        emb = face.normed_embedding
        if emb is None:
            return None
        return self.facebank.match(emb.astype(np.float32), thresh=self.bank_thresh)


# ---------- Stable Person ID (PID) ----------
class PersonManager:
    """
    PID persisten berbasis IoU+jarak.
    Jika ada 'identity' dari FaceMatcher (string label), PID langsung ‘dikunci’
    ke identity tsb (identity → PID satu-satu). Jadi super stabil.
    """
    def __init__(self, ttl=4.0, max_dist_px=140, min_iou=0.12):
        self.ttl = ttl
        self.max_dist_px = max_dist_px
        self.min_iou = min_iou
        self.pid_next = 1
        self.registry: Dict[int, dict] = {}     # pid -> {"box":..., "last_t":..., "identity": Optional[str]}
        self.identity2pid: Dict[str, int] = {}  # label -> pid
        self.tid2pid = {}

    @staticmethod
    def box_center(box):
        x1, y1, x2, y2 = box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @staticmethod
    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _best_match_pid(self, box, now):
        c_now = self.box_center(box)
        best_pid, best_score = None, -1.0
        for pid, st in list(self.registry.items()):
            if now - st["last_t"] > self.ttl:
                continue
            iou = iou_xyxy(box, st["box"])
            d = self.dist(c_now, self.box_center(st["box"]))
            score = iou - (d / max(self.max_dist_px, 1e-6)) * 0.2
            if iou >= self.min_iou or d <= self.max_dist_px:
                if score > best_score:
                    best_score, best_pid = score, pid
        return best_pid

    def assign(self, people_boxes, now, identities: Dict[int, Optional[str]]):
        out = []

        # 1) Dahulukan yang punya identity → map ke PID tetap via identity2pid
        for p in people_boxes:
            tid = p["id"]
            box = p["box"]
            ident = identities.get(tid) if identities else None
            if not ident:
                continue
            if ident in self.identity2pid:
                pid = self.identity2pid[ident]
            else:
                pid = self.pid_next
                self.pid_next += 1
                self.identity2pid[ident] = pid
            self.registry[pid] = {"box": box, "last_t": now, "identity": ident}
            self.tid2pid[tid] = pid
            out.append({**p, "pid": pid, "identity": ident})

        # 2) Sisanya (tanpa identity) → match spatio-temporal
        for p in people_boxes:
            if any(pp["id"] == p["id"] for pp in out):
                continue
            tid = p["id"]
            box = p["box"]
            pid = self._best_match_pid(box, now)
            if pid is None:
                pid = self.pid_next
                self.pid_next += 1
            st = self.registry.get(pid, {"identity": None})
            st.update({"box": box, "last_t": now})
            self.registry[pid] = st
            self.tid2pid[tid] = pid
            out.append({**p, "pid": pid, "identity": st.get("identity")})

        # 3) Bersihkan PID kadaluarsa (identity2pid tetap nempel agar re-entry lama stabil)
        for pid in list(self.registry.keys()):
            if now - self.registry[pid]["last_t"] > self.ttl:
                self.registry.pop(pid, None)
        return out


# ---------- Main ----------
def main():
    args = parse_args()
    SOURCE = int(args.source) if args.source.isdigit() else args.source
    ensure_dir(args.out_dir)

    # Hysteresis: pastikan r_unattend > r_attend + gap
    if args.r_unattend < args.r_attend + args.hysteresis_gap:
        args.r_unattend = args.r_attend + args.hysteresis_gap
        print(f"[INFO] Menyetel R_UNATTEND -> {args.r_unattend} (histeresis aktif)")

    # Load YOLOv8
    model = YOLO(args.model)

    # Init DeepSORT (TorchReID + OSNet)
    tracker_kwargs = dict(
        max_age=args.reid_max_age,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=args.reid_max_cos,
        nn_budget=args.reid_nn_budget,
        embedder="torchreid",
        half=True,
        bgr=True,
        # NOTE: beberapa versi lib belum support 'embedder_device'
        # Device bakal otomatis: CUDA kalau tersedia, else CPU.
    )
    if args.reid_model:
        tracker_kwargs["embedder_model_name"] = args.reid_model
    if args.reid_wts:
        tracker_kwargs["embedder_wts"] = args.reid_wts

    tracker = DeepSort(**tracker_kwargs)

    # NEW: Face matcher & Person manager
    face_matcher = FaceMatcher(device=args.face_device,
                            sample_every=args.face_sample_every,
                            bank_thresh=args.face_thresh)
    person_manager = PersonManager(ttl=args.pid_ttl,
                                max_dist_px=args.pid_max_dist,
                                min_iou=args.pid_min_iou)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit("Gagal membuka --source.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25
    FRAME_AREA = float(max(W * H, 1))

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(args.out_dir, f"annotated_{datetime.now():%Y%m%d_%H%M%S}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))
        print(f"[INFO] Simpan video anotasi: {out_path}")

    # State per object id (track id DeepSORT)
    obj_state: Dict[int, dict] = {}
    index_path = os.path.join(args.out_dir, args.index_file)
    index_db = load_index(index_path)

    print("[INFO] Running… tekan Q untuk berhenti.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi YOLO
        results = model(frame, conf=args.conf)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            # DeepSORT expects tlwh (x, y, w, h)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        # Tracking pakai DeepSORT
        tracks = tracker.update_tracks(detections, frame=frame)

        # Kumpulkan people & objects
        raw_people, objects = [], []
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cls = t.det_class
            name = model.names.get(cls, "object")

            if is_person(name):
                raw_people.append({"id": tid, "box": [x1, y1, x2, y2]})
            else:
                # filter ukuran objek
                area = max(0, x2 - x1) * max(0, y2 - y1)
                ratio = area / FRAME_AREA
                if args.min_area_ratio <= ratio <= args.max_area_ratio:
                    objects.append({"id": tid, "box": [x1, y1, x2, y2]})

        now = time.time()

        # Identifikasi wajah (sampling)
        identities: Dict[int, Optional[str]] = {}
        for p in raw_people:
            hit = face_matcher.maybe_identify(frame, p["box"])  # -> (label, score) or None
            if hit:
                label, score = hit
                identities[p["id"]] = label

        # Assign PID menggabungkan identity (jika ada)
        people = person_manager.assign(raw_people, now, identities)

        # Proses objek per track id
        for ob in objects:
            oid = ob["id"]
            st = obj_state.get(oid, {
                "status": "unknown",
                "owner_pid": None,
                "owner_name": None,  # kalau face match ada
                "attend_start": None,
                "last_attended_time": None,
                "unattend_start": None,
                "snapshot_done": False,
                # buffers:
                "attended_first": None,  # set sekali (ATTENDED pertama)
                "attended_last": None,   # overwrite terus (ATTENDED terakhir)
            })

            # jarak terdekat ke siapapun (untuk fase attend awal)
            if people:
                dists = [bbox_min_distance(ob["box"], p["box"]) for p in people]
                dmin = min(dists)
                near = people[dists.index(dmin)]
                near_pid = near["pid"]
                near_name = near.get("identity")
            else:
                dmin, near, near_pid, near_name = float("inf"), None, None, None

            # === Strict ownership: set owner sekali saat pertama kali attend lengkap ===
            if dmin <= args.r_attend:
                if st["attend_start"] is None:
                    st["attend_start"] = now
                if (now - st["attend_start"]) >= args.t_attend:
                    st["status"] = "attended"
                    st["last_attended_time"] = now
                    if st["owner_pid"] is None:
                        st["owner_pid"] = near_pid
                        st["owner_name"] = near_name  # bisa None kalau belum dikenal

                    # ---- Buffer ATTENDED (tidak menulis file) ----
                    ts_buf = datetime.now().strftime("%Y%m%d_%H%M%S")
                    x1, y1, x2, y2, ok = clamp_bbox(ob["box"], W, H)
                    buf_crop = frame[y1:y2, x1:x2].copy() if ok else None
                    buf_pack = {"timestamp": ts_buf, "frame": frame.copy(), "crop": buf_crop}

                    if st["attended_first"] is None:
                        st["attended_first"] = buf_pack
                    st["attended_last"] = buf_pack

                st["unattend_start"] = None
            else:
                st["attend_start"] = None

            # === Unattended hanya melihat jarak ke OWNER LAMA, bukan orang lain ===
            if st["status"] == "attended" and st["owner_pid"] is not None:
                owner = next((p for p in people if p.get("pid") == st["owner_pid"]), None)
                d_owner = bbox_min_distance(ob["box"], owner["box"]) if owner else float("inf")

                # grace time setelah terakhir attended (oleh owner)
                if (st["last_attended_time"] is not None) and \
                (now - st["last_attended_time"] < args.grace_after_attended):
                    st["unattend_start"] = None
                else:
                    if d_owner > args.r_unattend:
                        if st["unattend_start"] is None:
                            st["unattend_start"] = now
                        elif (now - st["unattend_start"]) >= args.t_unattend:
                            st["status"] = "unattended"

            # === Saat status UNATTENDED, lakukan snapshot pasangan ===
            if st["status"] == "unattended" and not st["snapshot_done"]:
                ts_un = datetime.now().strftime("%Y%m%d_%H%M%S")
                pair_key = f"ownerPID{st['owner_pid']}_obj{oid}"
                pair_dir = os.path.join(args.out_dir, pair_key)
                ensure_dir(pair_dir)

                # ambil crop untuk analisis
                x1, y1, x2, y2, ok = clamp_bbox(ob["box"], W, H)
                if not ok:
                    print(f"[WARN] UNATTENDED crop invalid for obj {oid}")
                    st["snapshot_done"] = True
                    obj_state[oid] = st
                    continue

                crop_path = os.path.join(pair_dir, f"TMP_{ts_un}_crop.jpg")
                cv2.imwrite(crop_path, frame[y1:y2, x1:x2].copy())

                # --- cek Gemini ---
                gemini_result = analyze_with_gemini(crop_path)
                if not gemini_result or gemini_result.get("kategori") != "barang pribadi":
                    print(f"[SKIP] Objek {oid} dilewati (Gemini kategorikan: {gemini_result})")
                    st["snapshot_done"] = True
                    obj_state[oid] = st
                    continue

                print(f"[CONFIRMED] Objek {oid} adalah barang pribadi → simpan snapshot")

                # Pastikan entry index tersedia
                pair = index_db["pairs"].setdefault(pair_key, {
                    "owner_pid": st["owner_pid"],
                    "owner_name": st["owner_name"],
                    "object_id": oid,
                    "attended": [],
                    "unattended": []
                })

                # simpan ATTENDED
                selected = st["attended_first"] if args.attended_mode == "first" else st["attended_last"]
                if selected is not None and selected.get("frame") is not None:
                    ts_att = selected["timestamp"]
                    att_frame = selected["frame"].copy()
                    title = "ATTENDED (first)" if args.attended_mode == "first" else "ATTENDED (last)"
                    cv2.putText(att_frame, title, (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
                    att_frame_path = os.path.join(pair_dir, f"ATTENDED_{ts_att}_frame.jpg")
                    cv2.imwrite(att_frame_path, att_frame)

                    if selected.get("crop") is not None:
                        att_crop_path = os.path.join(pair_dir, f"ATTENDED_{ts_att}_crop.jpg")
                        cv2.imwrite(att_crop_path, selected["crop"])
                    else:
                        att_crop_path = None

                    pair["attended"].append({
                        "timestamp": ts_att,
                        "frame": os.path.relpath(att_frame_path, args.out_dir),
                        "crop": (os.path.relpath(att_crop_path, args.out_dir) if att_crop_path else None)
                    })

                # simpan UNATTENDED
                un_frame = frame.copy()
                owner_txt = f"PID {st['owner_pid']}" + (f" ({st['owner_name']})" if st.get("owner_name") else "")
                cv2.putText(un_frame, f"UNATTENDED by {owner_txt}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
                un_frame_path = os.path.join(pair_dir, f"UNATTENDED_{ts_un}_frame.jpg")
                cv2.imwrite(un_frame_path, un_frame)

                un_crop_path = os.path.join(pair_dir, f"UNATTENDED_{ts_un}_crop.jpg")
                cv2.imwrite(un_crop_path, frame[y1:y2, x1:x2].copy())

                pair["unattended"].append({
                    "timestamp": ts_un,
                    "frame": os.path.relpath(un_frame_path, args.out_dir),
                    "crop": os.path.relpath(un_crop_path, args.out_dir)
                })
                save_index(index_path, index_db)

                st["snapshot_done"] = True
                st["attended_first"] = None
                st["attended_last"] = None

            obj_state[oid] = st

        # Overlay
        for p in people:
            x1, y1, x2, y2 = map(int, p["box"])
            name = p.get("identity")
            label = f"PID {p['pid']}" + (f" | {name}" if name else "")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (30, 30 if y1 < 40 else y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for ob in objects:
            x1, y1, x2, y2 = map(int, ob["box"])
            st = obj_state.get(ob["id"], {})
            color = (0, 165, 255) if st.get("status") != "unattended" else (0, 0, 255)
            owner_txt = ""
            if st.get("owner_pid") is not None:
                owner_txt = f"PID {st['owner_pid']}"
                if st.get("owner_name"):
                    owner_txt += f" ({st['owner_name']})"
            txt = f"object {ob['id']}"
            if st.get("status") == "attended":
                txt += f" [attended by {owner_txt}]"
            if st.get("status") == "unattended":
                txt += f" [UNATTENDED by {owner_txt}]"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                        3 if st.get("status") == "unattended" else 2)
            cv2.putText(frame, txt, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow(args.window, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
