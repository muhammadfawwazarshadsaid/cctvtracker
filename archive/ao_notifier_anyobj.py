# ao_notifier_anyobj_v2.py
# Abandon/Unattended Object Notifier (class-agnostic) dengan:
# - jarak bbox-to-bbox (bukan center) -> lebih akurat saat objek bersentuhan
# - histeresis antara R_ATTEND & R_UNATTEND agar stabil
# - grace time setelah attended sebelum boleh jadi unattended
# - snapshot frame + crop saat trigger

import os
import cv2
import time
import json
import argparse
from datetime import datetime
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image
import json
import os
import cv2
import time
import argparse
from datetime import datetime
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser("Abandon/Unattended Object Notifier v2 (class-agnostic)")
    ap.add_argument("--source", type=str, default="0",
                    help="0/1 untuk webcam, path video, atau RTSP/MJPEG URL")
    ap.add_argument("--model", type=str, default="yolov8n.pt",
                    help="Model YOLO (COCO) – auto-download")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence minimum")
    ap.add_argument("--r_attend", type=int, default=160, help="Radius ATTEND (px)")
    ap.add_argument("--t_attend", type=float, default=2.0, help="Durasi ATTEND (detik)")
    ap.add_argument("--r_unattend", type=int, default=260, help="Radius UNATTEND (px)")
    ap.add_argument("--t_unattend", type=float, default=5.0, help="Durasi UNATTEND (detik)")
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
    ap.add_argument("--window", type=str, default="Abandon Object (v2)", help="Nama window")
    return ap.parse_args()

# ---------- Utils ----------
def is_person(name: str) -> bool:
    return name == "person"

def bbox_min_distance(a, b):
    """Jarak Euclidean terpendek antara dua bbox. 0 jika overlap/menyentuh."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0.0, max(bx1 - ax2, ax1 - bx2))   # horizontal gap
    dy = max(0.0, max(by1 - ay2, ay1 - by2))   # vertical gap
    return (dx*dx + dy*dy) ** 0.5

# ---------- Main ----------
def main():
    args = parse_args()
    SOURCE = int(args.source) if args.source.isdigit() else args.source
    os.makedirs(args.out_dir, exist_ok=True)

    # enforce histeresis: R_UNATTEND >= R_ATTEND + hysteresis_gap
    if args.r_unattend < args.r_attend + args.hysteresis_gap:
        args.r_unattend = args.r_attend + args.hysteresis_gap
        print(f"[INFO] Menyetel R_UNATTEND -> {args.r_unattend} (histeresis aktif)")

    model = YOLO(args.model)

    # probe ukuran frame
    cap_probe = cv2.VideoCapture(SOURCE)
    if not cap_probe.isOpened():
        raise SystemExit("Gagal membuka --source. Cek alamat/RTSP/ID webcam.")
    W = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    FPS = cap_probe.get(cv2.CAP_PROP_FPS) or 25
    FRAME_AREA = float(max(W * H, 1))
    cap_probe.release()

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(args.out_dir, f"annotated_{datetime.now():%Y%m%d_%H%M%S}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))
        print(f"[INFO] Simpan video anotasi: {out_path}")

    # State per object (berdasarkan tracker ID)
    # obj_state[obj_id] = {
    #   "status": "unknown|attended|unattended",
    #   "owner_id": int|None,
    #   "attend_start": float|None,
    #   "last_attended_time": float|None,
    #   "unattend_start": float|None,
    #   "snapshot_done": bool
    # }
    obj_state = {}

    print("[INFO] Running… tekan Q untuk berhenti.")
    while True:
        try:
            track_stream = model.track(
                source=SOURCE, conf=args.conf, stream=True, persist=True,
                tracker="bytetrack.yaml", verbose=False
            )
            for res in track_stream:
                frame = res.orig_img.copy()
                people, objects = [], []

                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None
                    if ids is None:
                        continue  # perlu ID tracker

                    for i in range(len(xyxy)):
                        name = model.names[cls[i]]
                        box = xyxy[i].tolist()
                        tid = int(ids[i])

                        if is_person(name):
                            people.append({"id": tid, "box": box})
                        else:
                            # filter ukuran
                            x1, y1, x2, y2 = map(int, box)
                            area = max(0, x2 - x1) * max(0, y2 - y1)
                            ratio = area / FRAME_AREA
                            if args.min_area_ratio <= ratio <= args.max_area_ratio:
                                objects.append({"id": tid, "box": box})

                # update state
                now = time.time()
                for ob in objects:
                    oid = ob["id"]
                    st = obj_state.get(oid, {
                        "status": "unknown",
                        "owner_id": None,
                        "attend_start": None,
                        "last_attended_time": None,
                        "unattend_start": None,
                        "snapshot_done": False
                    })

                    # cari orang terdekat dgn jarak bbox-to-bbox
                    if people:
                        dists = [bbox_min_distance(ob["box"], p["box"]) for p in people]
                        dmin = min(dists)
                        near_person = people[dists.index(dmin)]
                        near_id = near_person["id"]
                    else:
                        dmin, near_id = float("inf"), None

                    # fase attended (jarak <= R_ATTEND selama T_ATTEND)
                    if dmin <= args.r_attend:
                        if st["attend_start"] is None:
                            st["attend_start"] = now
                        if (now - st["attend_start"]) >= args.t_attend:
                            st["status"] = "attended"
                            st["owner_id"] = near_id
                            st["last_attended_time"] = now
                        st["unattend_start"] = None
                    else:
                        st["attend_start"] = None  # reset hitung attend bila keluar radius

                    # kandidat unattended hanya jika pernah attended
                    if st["status"] == "attended":
                        # respect grace setelah attended
                        if (st["last_attended_time"] is not None) and (now - st["last_attended_time"] < args.grace_after_attended):
                            st["unattend_start"] = None
                        else:
                            if dmin > args.r_unattend:
                                if st["unattend_start"] is None:
                                    st["unattend_start"] = now
                                elif (now - st["unattend_start"]) >= args.t_unattend:
                                    st["status"] = "unattended"

                    # trigger sekali saat unattended
                    if st["status"] == "unattended" and not st["snapshot_done"]:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        file_id = f"UNATTENDED_id{oid}_{ts}" # <<< MODIFIKASI: Gunakan ini sebagai ID
                        label = f"UNATTENDED OBJECT! ({file_id})"
                        cv2.putText(frame, "UNATTENDED OBJECT!", (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

                        frame_path = os.path.join(args.out_dir, f"{file_id}_frame.jpg")
                        cv2.imwrite(frame_path, frame)

                        x1, y1, x2, y2 = map(int, ob["box"])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W - 1, x2), min(H - 1, y2)
                        crop = frame[y1:y2, x1:x2].copy()
                        crop_path = os.path.join(args.out_dir, f"{file_id}_crop.jpg")
                        cv2.imwrite(crop_path, crop)

                        print(f"[ALERT] UNATTENDED OBJECT @ {ts} | obj_id={oid} owner_id={st['owner_id']}\n"
                              f"   ID File: {file_id}\n"
                              f"   Frame: {frame_path}\n   Crop : {crop_path}")
                        
                        # --- PANGGIL GEMINI DI SINI ---
                        print("[INFO] Menganalisis dengan Gemini...")
                        gemini_analysis = analyze_with_gemini(crop_path)
                        print("--- HASIL ANALISIS GEMINI ---")
                        print(f"ID: {file_id}")
                        print(gemini_analysis)
                        print("-----------------------------\n")
                        # --------------------------------

                        st["snapshot_done"] = True

                    obj_state[oid] = st

                # --- overlay
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
                        txt += " [attended]"
                    if st.get("status") == "unattended":
                        txt += " [UNATTENDED]"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                                  3 if st.get("status") == "unattended" else 2)
                    cv2.putText(frame, txt, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if writer is not None:
                    writer.write(frame)

                cv2.imshow(args.window, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[WARN] Stream error: {e}. Reconnect 2s…")
            time.sleep(2.0)
            continue

    if writer is not None: writer.release()
    cv2.destroyAllWindows()

def analyze_with_gemini(image_path: str):
    """
    Mengirim crop gambar objek ke Gemini dan meminta analisis dalam format JSON.
    Mengembalikan dictionary hasil parsing atau None jika gagal.
    """
    try:
        api_key = "AIzaSyDiMY2xY0N_eOw5vUzk-J3sLVDb81TEfS8"
        if not api_key:
            print("[GEMINI-ERROR] GOOGLE_API_KEY tidak ditemukan.")
            return None
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        img = Image.open(image_path)

        prompt = (
            "Analisis gambar objek ini yang terdeteksi sebagai 'barang tak bertuan'. Analisis seakan-akan kamu adalah manusia yang dapat mendeteksi dan mengenali objek sehari-hari termasuk nama spesifik dan kategorinya. Jika tidak yakin, jangan menebak, cukup kategorikan sebagai 'barang umum'. Kamu bisa eksplorasi nama spesifik berdasarkan bentuk, warna, dan fungsi yang terlihat.\n"
            "Jawab HANYA dengan format JSON yang valid, tanpa teks tambahan atau markdown. "
            "JSON harus memiliki kunci berikut:\n"
            "- 'nama_objek': (string) Nama spesifik dari objek (misal: 'tas ransel biru', 'koper silver', 'botol minum').\n"
            "- 'kategori': (string) Tentukan apakah ini 'barang pribadi' atau 'fasilitas umum'.\n"
            "\n"
            "Contoh JSON: {\"nama_objek\": \"tas ransel hitam\", \"kategori\": \"barang pribadi\"}"
        )

        response = model.generate_content([prompt, img])
        
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_text)

    except Exception as e:
        print(f"[GEMINI-ERROR] Gagal menganalisis atau parsing JSON: {e}")
        return None
    
if __name__ == "__main__":
    main()
