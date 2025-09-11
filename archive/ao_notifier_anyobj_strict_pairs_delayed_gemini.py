import os
import cv2
import time
import json
import argparse
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import google.generativeai as genai
from PIL import Image

# ---------- Args ----------
def parse_args():
    ap = argparse.ArgumentParser("AO Notifier (DeepSORT + Strict Ownership + Paired Snapshots, Attended first/last)")
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
    ap.add_argument("--window", type=str, default="Abandon Object (Strict Pairs)", help="Nama window")
    ap.add_argument("--index_file", type=str, default="ao_index.json", help="Nama file index JSON")
    ap.add_argument("--attended_mode", type=str, default="first",
                    choices=["first", "last"],
                    help="Gunakan ATTENDED pertama ('first') atau terakhir ('last') saat menyimpan pasangan.")
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
    return {"pairs": {}}  # key: "owner{owner_id}_obj{obj_id}"

def save_index(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def analyze_with_gemini(image_path: str):
    """
    Kirim crop objek ke Gemini, return dict JSON
    dengan kunci: {"nama_objek": "...", "kategori": "barang pribadi/fasilitas umum"}
    """
    try:
        api_key = "AIzaSyDiMY2xY0N_eOw5vUzk-J3sLVDb81TEfS8"
        if not api_key:
            print("[GEMINI-ERROR] GOOGLE_API_KEY tidak ditemukan.")
            return None
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        img = Image.open(image_path)

        prompt = (
            "Analisis gambar ini. Tentukan apakah objek adalah barang pribadi "
            "Barang pribadi meliputi interaksi objek yang jelas dipegang/dipakai/dikenakan oleh orang tersebut"
            "(misal: tas yang digendong, koper yang ditenteng, botol yang dibawa, laptop yang dibawa, jaket yang dipakai) atau fasilitas umum "
            "(misal: kursi, meja, pagar, tong sampah). "
            "Jawab HANYA JSON valid dengan format: "
            "{\"nama_objek\": \"...\", \"kategori\": \"barang pribadi\"}"
        )

        response = model.generate_content([prompt, img])
        cleaned = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned)
    except Exception as e:
        print(f"[GEMINI-ERROR] {e}")
        return None
    
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

    # Init DeepSORT
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0,
                       max_cosine_distance=0.5, embedder="mobilenet",
                       half=True)

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
    obj_state = {}
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
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        # Tracking pakai DeepSORT
        tracks = tracker.update_tracks(detections, frame=frame)

        # Kumpulkan people & objects
        people, objects = [], []
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cls = t.det_class
            name = model.names.get(cls, "object")

            if is_person(name):
                people.append({"id": tid, "box": [x1, y1, x2, y2]})
            else:
                # filter ukuran objek
                area = max(0, x2 - x1) * max(0, y2 - y1)
                ratio = area / FRAME_AREA
                if args.min_area_ratio <= ratio <= args.max_area_ratio:
                    objects.append({"id": tid, "box": [x1, y1, x2, y2]})

        now = time.time()
        for ob in objects:
            oid = ob["id"]
            st = obj_state.get(oid, {
                "status": "unknown",
                "owner_id": None,
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
                near_id = people[dists.index(dmin)]["id"]
            else:
                dmin, near_id = float("inf"), None

            # === Strict ownership: set owner sekali saat pertama kali attend lengkap ===
            if dmin <= args.r_attend:
                if st["attend_start"] is None:
                    st["attend_start"] = now
                if (now - st["attend_start"]) >= args.t_attend:
                    st["status"] = "attended"
                    st["last_attended_time"] = now
                    if st["owner_id"] is None:
                        st["owner_id"] = near_id  # tetapkan owner sekali

                    # ---- Buffer ATTENDED (tidak menulis file) ----
                    ts_buf = datetime.now().strftime("%Y%m%d_%H%M%S")
                    x1, y1, x2, y2, ok = clamp_bbox(ob["box"], W, H)
                    buf_crop = frame[y1:y2, x1:x2].copy() if ok else None
                    buf_pack = {"timestamp": ts_buf, "frame": frame.copy(), "crop": buf_crop}

                    # FIRST: hanya isi sekali
                    if st["attended_first"] is None:
                        st["attended_first"] = buf_pack
                    # LAST: selalu update
                    st["attended_last"] = buf_pack

                st["unattend_start"] = None
            else:
                st["attend_start"] = None

            # === Unattended hanya melihat jarak ke OWNER LAMA, bukan orang lain ===
            if st["status"] == "attended" and st["owner_id"] is not None:
                owner = next((p for p in people if p["id"] == st["owner_id"]), None)
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

            if st["status"] == "unattended" and not st["snapshot_done"]:
                ts_un = datetime.now().strftime("%Y%m%d_%H%M%S")
                pair_key = f"owner{st['owner_id']}_obj{oid}"
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
                    "owner_id": st["owner_id"],
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
                cv2.putText(un_frame, "UNATTENDED", (30, 40),
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"person {p['id']}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for ob in objects:
            x1, y1, x2, y2 = map(int, ob["box"])
            st = obj_state.get(ob["id"], {})
            color = (0, 165, 255) if st.get("status") != "unattended" else (0, 0, 255)
            txt = f"object {ob['id']}"
            if st.get("status") == "attended":
                txt += f" [attended by {st['owner_id']}]"
            if st.get("status") == "unattended":
                txt += f" [UNATTENDED by {st['owner_id']}]"
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
