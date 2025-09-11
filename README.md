# AO Notifier (OSNet + Face ID + Stable PID)

Sistem pendeteksi **abandoned object** dengan:
- **YOLOv8** deteksi objek (COCO).
- **DeepSORT** + **TorchReID/OSNet** untuk tracking ID orang yang “lengket”.
- **Stable PID** (persistent ID) di atas track_id agar tidak gonta‑ganti walau track putus.
- **Face Recognition** (InsightFace + FaceBank) untuk mengunci identitas (opsional).
- **Gemini** untuk memvalidasi bahwa objek termasuk **barang pribadi** sebelum menyimpan snapshot pasangan *ATTENDED ↔ UNATTENDED*.

> Tested on macOS (CPU).

---

## 1) Setup Python env

Disarankan pakai virtualenv/pyenv/conda.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

### Apple Silicon (M1/M2/M3)
InsightFace memakai ONNX Runtime. Jika `onnxruntime` biasa bermasalah di Apple Silicon, gunakan:
```bash
pip uninstall -y onnxruntime
pip install onnxruntime-silicon
```

---

## 2) TorchReID compatibility (fix `torchreid.utils`)

Versi terbaru `torchreid` (0.2.x) **mengubah struktur paket** sehingga `deep-sort-realtime` lama gagal import:
```
ModuleNotFoundError: No module named 'torchreid.utils'
```

### Opsi cepat (shim) — *yang kamu pakai sekarang*
Buat shim agar import lama tetap jalan:
```bash
python - <<'PY'
import os, torchreid
pkg_dir = os.path.dirname(torchreid.__file__)
shim_dir = os.path.join(pkg_dir, 'utils')
os.makedirs(shim_dir, exist_ok=True)
with open(os.path.join(shim_dir, '__init__.py'), 'w') as f:
    f.write('from torchreid.reid.utils.feature_extractor import FeatureExtractor\n')
print('Shim created at:', os.path.join(shim_dir, '__init__.py'))
PY
```

### Alternatif (legacy)
Pakai versi legacy dari GitHub (punya API lama):
```bash
pip uninstall -y torchreid
pip install 'git+https://github.com/KaiyangZhou/deep-person-reid.git@v1.4.0'
```

> **Catatan:** Jangan campur Opsi cepat dan alternatif sekaligus. Pilih salah satu.

---

## 3) Konfigurasi Gemini API Key

**Jangan** hardcode API key. Pakai environment variable:
```bash
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

Di kode, gunakan `os.getenv("GOOGLE_API_KEY")` (sudah disiapkan pada template sebelumnya).

> Di file kamu saat ini, masih ada key tersisip. **Hapus/ubah** menjadi pembacaan env var untuk keamanan.

---

## 4) Struktur data & direktori

```
project-root/
├── ao_notifier_pid_face.py        # skrip utama
├── requirements.txt
├── README.md
├── images/                        # FaceBank (1 foto per orang, nama file = label, ex: Budi.jpg)
└── ao_snaps/                      # output snapshot (dibuat otomatis)
```

- Folder `images/` berisi **1 foto/orang**; nama file menjadi **label** identitas.
- Folder `ao_snaps/` akan berisi pasangan snapshot ATTENDED/UNATTENDED beserta `ao_index.json`.

---

## 5) Cara menjalankan

Webcam (index 0) + simpan video anotasi:
```bash
python ao_notifier_pid_face.py --source 0 --save_video
```

RTSP/Video file:
```bash
python ao_notifier_pid_face.py --source "rtsp://user:pass@ip/..." --save_video
# atau
python ao_notifier_pid_face.py --source /path/to/video.mp4 --save_video
```

Parameter penting (default aman):
- `--reid_model osnet_x0_25` (OSNet ringan, stabil)
- `--reid_max_age 40` (perpanjang umur track saat occlusion → bisa naik ke 50–60)
- `--reid_max_cos 0.20` (ketat; kalau ID sering putus, naikkan ke 0.25–0.28)
- `--pid_ttl 4.0` (sinkron dengan `max_age`; bisa 5–6 detik)
- `--face_sample_every 5` (sampling identitas tiap N frame; naikkan ke 8–10 jika CPU berat)
- `--face_thresh 0.40` (threshold kemiripan; 0.38–0.45 aman)

---

## 6) Troubleshooting cepat

- **`TypeError: DeepSort.__init__() got an unexpected keyword argument 'embedder_device'`**  
  → Hapus argumen `embedder_device` dari inisialisasi `DeepSort` (tidak disupport pada versi kamu).

- **`ImportError: torchreid is not installed` padahal sudah install**  
  → Biasanya ada dependency kurang saat import. Install:
  ```bash
  pip install gdown yacs pyyaml prettytable tabulate tensorboard
  ```
  Lalu terapkan **shim** di atas.

- **`ModuleNotFoundError: No module named 'tensorboard'`**  
  → `pip install tensorboard`

- **InsightFace/ONNX runtime error di Apple Silicon**  
  → Ganti ke `onnxruntime-silicon` (lihat bagian Setup).

- **FPS drop/CPU berat**  
  → Naikkan `--face_sample_every` dan/atau turunkan resolusi input.  
  → Kurangi `--model` ke `yolov8n.pt` (sudah default).

- **ID masih sering ganti**  
  → Naikkan `--reid_max_age`, `--pid_ttl`, dan sedikit longgarkan `--reid_max_cos`.

---

## 7) Keamanan & Privasi

- Simpan API key di env, bukan di source control.  
- Hati‑hati menyimpan snapshot yang memuat wajah; pastikan izin & kebijakan privasi terpenuhi.

---

## 8) Contoh command praktis

```bash
# Mode default (webcam 0) + video anotasi + param yang lebih lengket
python ao_notifier_pid_face.py --source 0 --save_video \
  --reid_max_age 50 --pid_ttl 5 --reid_max_cos 0.25 \
  --face_sample_every 8
```

Selamat mencoba! Kalau butuh, saya bisa bikinin `Makefile` kecil buat bootstrap (`make setup`, `make run`).

