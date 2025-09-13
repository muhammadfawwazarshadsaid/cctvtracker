package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/google/generative-ai-go/genai"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
	"google.golang.org/api/option"
)

// Server struct sekarang menyimpan koneksi DB dan klien Gemini
type Server struct {
	DB     *pgxpool.Pool
	Gemini *genai.GenerativeModel
}

// ----- Structs untuk Payload dan Response -----

type NotifyPayload struct {
	GroupKey  string      `json:"group_key"`
	OwnerPID  int         `json:"owner_pid"`
	OwnerName string      `json:"owner_name,omitempty"`
	ItemName  string      `json:"item_name,omitempty"`
	Location  string      `json:"location,omitempty"`
	Snapshot  SnapPayload `json:"snapshot"`
}

type SnapPayload struct {
	Type      string    `json:"type"`
	TS        time.Time `json:"ts"`
	FrameData string    `json:"frame_data,omitempty"`
	CropData  string    `json:"crop_data,omitempty"`
	Message   string    `json:"message,omitempty"`
	Meta      any       `json:"meta,omitempty"`
}

type GroupRow struct {
	ID               int64     `json:"id"`
	GroupKey         string    `json:"group_key"`
	OwnerPID         int       `json:"owner_pid"`
	OwnerName        *string   `json:"owner_name,omitempty"`
	ItemName         *string   `json:"item_name,omitempty"`
	LocationLabel    *string   `json:"location,omitempty"`
	Status           string    `json:"status"`
	PreviewFrameData *string   `json:"preview_frame_data,omitempty"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
}

type EventRow struct {
	ID         int64     `json:"id"`
	GroupID    int64     `json:"group_id"`
	Kind       string    `json:"kind"`
	Message    string    `json:"message"`
	OccurredAt time.Time `json:"occurred_at"`
	FrameData  *string   `json:"frame_data,omitempty"`
	CropData   *string   `json:"crop_data,omitempty"`
	Meta       any       `json:"meta,omitempty"`
}

// [BARU] Struct untuk pesan chat
type ChatMessage struct {
	ID        int64     `json:"id"`
	GroupID   int64     `json:"group_id"`
	Sender    string    `json:"sender"` // "user" atau "ai"
	Message   string    `json:"message"`
	CreatedAt time.Time `json:"created_at"`
}

// [BARU] Struct untuk payload request chat
type ChatRequestPayload struct {
	Message string `json:"message"`
}

func main() {
	_ = godotenv.Load()

	// --- Koneksi Database ---
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		log.Fatal("DATABASE_URL env kosong. Contoh: postgres://user:pass@localhost:5432/ao?sslmode=disable")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		log.Fatal("Gagal konek database: ", err)
	}
	defer pool.Close()

	if err := runMigrations(ctx, pool); err != nil {
		log.Fatal("Gagal migrasi: ", err)
	}

	// --- Inisialisasi Klien Gemini AI ---
	geminiAPIKey := "AIzaSyDiMY2xY0N_eOw5vUzk-J3sLVDb81TEfS8"
	if geminiAPIKey == "" {
		log.Fatal("GOOGLE_API_KEY env kosong.")
	}
	geminiClient, err := genai.NewClient(ctx, option.WithAPIKey(geminiAPIKey))
	if err != nil {
		log.Fatal("Gagal membuat klien Gemini: ", err)
	}
	defer geminiClient.Close()
	// Gunakan model yang sesuai, 'gemini-1.5-flash' adalah pilihan yang cepat dan efisien.
	geminiModel := geminiClient.GenerativeModel("gemini-1.5-flash")

	s := &Server{
		DB:     pool,
		Gemini: geminiModel,
	}

	// --- Routing / Endpoints ---
	mux := http.NewServeMux()
	mux.HandleFunc("POST /notify", s.handleNotify)
	mux.HandleFunc("GET /groups", s.handleGroups)
	mux.HandleFunc("GET /groups/{id}", s.handleGroupDetail)
	mux.HandleFunc("POST /groups/{id}/resolve", s.handleResolveGroup)

	// [BARU] Endpoint untuk fungsionalitas chat
	mux.HandleFunc("POST /groups/{id}/chat", s.handleChat)

	port := os.Getenv("PORT")
	if port == "" {
		port = "3000"
	}
	addr := ":" + port
	log.Println("Server berjalan di", addr)
	log.Fatal(http.ListenAndServe(addr, logRequest(mux)))
}

func runMigrations(ctx context.Context, db *pgxpool.Pool) error {
	sql := `
    CREATE TABLE IF NOT EXISTS snapshot_groups (
        id BIGSERIAL PRIMARY KEY,
        group_key TEXT UNIQUE NOT NULL,
        owner_pid INT NOT NULL,
        owner_name TEXT,
        item_name TEXT,
        location_label TEXT,
        status TEXT NOT NULL DEFAULT 'ongoing',
        preview_frame_data BYTEA,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS snapshot_events (
        id BIGSERIAL PRIMARY KEY,
        group_id BIGINT REFERENCES snapshot_groups(id) ON DELETE CASCADE,
        kind TEXT NOT NULL,
        message TEXT NOT NULL,
        occurred_at TIMESTAMP WITH TIME ZONE NOT NULL,
        frame_data BYTEA,
        crop_data BYTEA,
        meta JSONB DEFAULT '{}'::jsonb
    );
    -- [BARU] Tabel untuk menyimpan riwayat chat per insiden
    CREATE TABLE IF NOT EXISTS chat_messages (
        id BIGSERIAL PRIMARY KEY,
        group_id BIGINT REFERENCES snapshot_groups(id) ON DELETE CASCADE,
        sender TEXT NOT NULL, -- 'user' or 'ai'
        message TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_snapshot_events_group_time ON snapshot_events(group_id, occurred_at);
    CREATE INDEX IF NOT EXISTS idx_snapshot_groups_updated_at ON snapshot_groups(updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_chat_messages_group_id ON chat_messages(group_id, created_at);
    `
	_, err := db.Exec(ctx, sql)
	return err
}

// -------- Handlers --------

// (handleNotify tidak berubah)
func (s *Server) handleNotify(w http.ResponseWriter, r *http.Request) {
    var p NotifyPayload
    if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
        http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
        return
    }
    if p.GroupKey == "" || p.OwnerPID == 0 || p.Snapshot.Type == "" || p.Snapshot.TS.IsZero() {
        http.Error(w, "missing required fields", http.StatusBadRequest)
        return
    }
    frameBytes, err := decodeBase64(p.Snapshot.FrameData)
    if err != nil {
        http.Error(w, "invalid frame_data base64: "+err.Error(), http.StatusBadRequest)
        return
    }
    cropBytes, err := decodeBase64(p.Snapshot.CropData)
    if err != nil {
        http.Error(w, "invalid crop_data base64: "+err.Error(), http.StatusBadRequest)
        return
    }

    ctx := r.Context()
    tx, err := s.DB.Begin(ctx)
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }
    defer tx.Rollback(ctx)

    var groupID int64
    upsert := `
    INSERT INTO snapshot_groups (group_key, owner_pid, owner_name, item_name, location_label, status, preview_frame_data)
    VALUES ($1,$2,$3,$4,$5,'ongoing',$6)
    ON CONFLICT (group_key) DO UPDATE SET
        owner_pid = EXCLUDED.owner_pid,
        owner_name = COALESCE(EXCLUDED.owner_name, snapshot_groups.owner_name),
        item_name = COALESCE(EXCLUDED.item_name, snapshot_groups.item_name),
        location_label = COALESCE(EXCLUDED.location_label, snapshot_groups.location_label),
        updated_at = now()
    RETURNING id;
    `
    previewBytes := frameBytes
    err = tx.QueryRow(ctx, upsert, p.GroupKey, p.OwnerPID, nullify(p.OwnerName), nullify(p.ItemName), nullify(p.Location), nullifyBytes(previewBytes)).Scan(&groupID)
    if err != nil {
        http.Error(w, "upsert group: "+err.Error(), 500)
        return
    }
    msg := p.Snapshot.Message
    if msg == "" {
        switch p.Snapshot.Type {
        case "attended":
            msg = "CCTV: Kamu membawa " + fallback(p.ItemName, "barang")
        case "distance_gt_2m":
            msg = "CCTV: Posisi kamu dengan " + fallback(p.ItemName, "barang") + " > 2 meter"
        case "person_left_frame":
            msg = "CCTV: Kamu sudah tidak di area kamera"
        case "unattended":
            msg = "CCTV: Kamu meninggalkan " + fallback(p.ItemName, "barang")
        default:
            msg = "CCTV: " + p.Snapshot.Type
        }
    }
    insEv := `
    INSERT INTO snapshot_events (group_id, kind, message, occurred_at, frame_data, crop_data, meta)
    VALUES ($1,$2,$3,$4,$5,$6,$7)
    RETURNING id;`
    var evID int64
    metaJSON, _ := json.Marshal(p.Snapshot.Meta)
    if err := tx.QueryRow(ctx, insEv, groupID, p.Snapshot.Type, msg, p.Snapshot.TS, nullifyBytes(frameBytes), nullifyBytes(cropBytes), metaJSON).Scan(&evID); err != nil {
        http.Error(w, "insert event: "+err.Error(), 500)
        return
    }

    newStatus := ""
    var newPreviewBytes []byte
    switch p.Snapshot.Type {
    case "unattended":
        newStatus = "unattended"
        newPreviewBytes = frameBytes
    }
    if newStatus != "" || len(newPreviewBytes) > 0 {
        sb := strings.Builder{}
        sb.WriteString("UPDATE snapshot_groups SET updated_at=now()")
        args := []any{}
        argCount := 1
        if newStatus != "" {
            sb.WriteString(", status=$" + strconv.Itoa(argCount))
            args = append(args, newStatus)
            argCount++
        }
        if len(newPreviewBytes) > 0 {
            sb.WriteString(", preview_frame_data=$" + strconv.Itoa(argCount))
            args = append(args, newPreviewBytes)
            argCount++
        }
        if len(args) > 0 {
            sb.WriteString(" WHERE id=$" + strconv.Itoa(argCount))
            args = append(args, groupID)
            if _, err := tx.Exec(ctx, sb.String(), args...); err != nil {
                http.Error(w, "update group status: "+err.Error(), 500)
                return
            }
        }
    }

    if err := tx.Commit(ctx); err != nil {
        http.Error(w, err.Error(), 500)
        return
    }

    writeJSON(w, http.StatusCreated, map[string]any{
        "group_id":  groupID,
        "event_id":  evID,
        "message":   msg,
        "newStatus": ifnz(newStatus),
    })
}

// (handleGroups tidak berubah)
func (s *Server) handleGroups(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    limit := 20
    if q := r.URL.Query().Get("limit"); q != "" {
        if v, err := strconv.Atoi(q); err == nil && v > 0 && v <= 200 {
            limit = v
        }
    }
    rows, err := s.DB.Query(ctx, `
    SELECT id, group_key, owner_pid, owner_name, item_name, location_label, status, preview_frame_data, created_at, updated_at
    FROM snapshot_groups
    ORDER BY updated_at DESC
    LIMIT $1`, limit)
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }
    defer rows.Close()

    var out []GroupRow
    for rows.Next() {
        var g GroupRow
        var previewFrameData []byte
        if err := rows.Scan(&g.ID, &g.GroupKey, &g.OwnerPID, &g.OwnerName, &g.ItemName, &g.LocationLabel, &g.Status, &previewFrameData, &g.CreatedAt, &g.UpdatedAt); err != nil {
            http.Error(w, err.Error(), 500)
            return
        }
        g.PreviewFrameData = encodeBase64(previewFrameData)
        out = append(out, g)
    }
    writeJSON(w, 200, out)
}

// [DIUBAH] handleGroupDetail sekarang juga mengambil riwayat chat
func (s *Server) handleGroupDetail(w http.ResponseWriter, r *http.Request) {
	idStr := r.PathValue("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		http.Error(w, "bad id", 400)
		return
	}
	ctx := r.Context()

	// 1. Ambil detail grup
	var g GroupRow
	var previewFrameData []byte
	err = s.DB.QueryRow(ctx, `
    SELECT id, group_key, owner_pid, owner_name, item_name, location_label, status, preview_frame_data, created_at, updated_at
    FROM snapshot_groups WHERE id=$1`, id,
	).Scan(&g.ID, &g.GroupKey, &g.OwnerPID, &g.OwnerName, &g.ItemName, &g.LocationLabel, &g.Status, &previewFrameData, &g.CreatedAt, &g.UpdatedAt)
	if err != nil {
		http.Error(w, "group not found", 404)
		return
	}
	g.PreviewFrameData = encodeBase64(previewFrameData)

	// 2. Ambil event/kejadian terkait
	erows, err := s.DB.Query(ctx, `
    SELECT id, group_id, kind, message, occurred_at, frame_data, crop_data, meta
    FROM snapshot_events
    WHERE group_id=$1
    ORDER BY occurred_at ASC`, id)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer erows.Close()

	events := []EventRow{}
	for erows.Next() {
		var e EventRow
		var frameData, cropData, metaRaw []byte
		if err := erows.Scan(&e.ID, &e.GroupID, &e.Kind, &e.Message, &e.OccurredAt, &frameData, &cropData, &metaRaw); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		e.FrameData = encodeBase64(frameData)
		e.CropData = encodeBase64(cropData)
		if len(metaRaw) > 0 {
			var anyMeta any
			_ = json.Unmarshal(metaRaw, &anyMeta)
			e.Meta = anyMeta
		}
		events = append(events, e)
	}

	// 3. [BARU] Ambil riwayat chat
	crows, err := s.DB.Query(ctx, `
    SELECT id, group_id, sender, message, created_at
    FROM chat_messages
    WHERE group_id=$1
    ORDER BY created_at ASC`, id)
	if err != nil {
		http.Error(w, "gagal fetch chat: "+err.Error(), 500)
		return
	}
	defer crows.Close()
	chatHistory := []ChatMessage{}
	for crows.Next() {
		var c ChatMessage
		if err := crows.Scan(&c.ID, &c.GroupID, &c.Sender, &c.Message, &c.CreatedAt); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		chatHistory = append(chatHistory, c)
	}

	// 4. Kirim semua data dalam satu response
	writeJSON(w, 200, map[string]any{
		"group":  g,
		"events": events,
		"chat":   chatHistory,
	})
}

// (handleResolveGroup tidak berubah)
func (s *Server) handleResolveGroup(w http.ResponseWriter, r *http.Request) {
    idStr := r.PathValue("id")
    id, err := strconv.ParseInt(idStr, 10, 64)
    if err != nil {
        http.Error(w, "bad id", 400)
        return
    }
    ctx := r.Context()
    cmdTag, err := s.DB.Exec(ctx, `UPDATE snapshot_groups SET status='resolved', updated_at=now() WHERE id=$1`, id)
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }
    if cmdTag.RowsAffected() == 0 {
        http.Error(w, "not found", 404)
        return
    }
    writeJSON(w, 200, map[string]string{"status": "resolved"})
}

// [BARU] Handler untuk fungsionalitas chat dengan Gemini AI
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	idStr := r.PathValue("id")
	groupID, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		http.Error(w, "bad id", 400)
		return
	}
	ctx := r.Context()

	// 1. Decode pesan dari user
	var p ChatRequestPayload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(p.Message) == "" {
		http.Error(w, "pesan tidak boleh kosong", http.StatusBadRequest)
		return
	}

	// 2. Simpan pesan user ke database
	_, err = s.DB.Exec(ctx,
		`INSERT INTO chat_messages (group_id, sender, message) VALUES ($1, 'user', $2)`,
		groupID, p.Message,
	)
	if err != nil {
		http.Error(w, "gagal simpan pesan user: "+err.Error(), 500)
		return
	}

	// 3. Bangun Konteks untuk AI (prompt engineering)
	prompt, err := s.buildChatPrompt(ctx, groupID, p.Message)
	if err != nil {
		http.Error(w, "gagal bangun prompt: "+err.Error(), 500)
		return
	}

	// 4. Panggil Gemini API
	resp, err := s.Gemini.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		log.Printf("Error Gemini API: %v", err)
		http.Error(w, "gagal komunikasi dengan AI: "+err.Error(), 502)
		return
	}

	// Ekstrak teks balasan dari AI
	aiMessage := ""
	if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
		for _, part := range resp.Candidates[0].Content.Parts {
			if txt, ok := part.(genai.Text); ok {
				aiMessage += string(txt)
			}
		}
	}
	if aiMessage == "" {
		aiMessage = "Maaf, saya tidak bisa memberikan balasan saat ini."
	}

	// 5. Simpan balasan AI ke database
	var newAiMsg ChatMessage
	err = s.DB.QueryRow(ctx,
		`INSERT INTO chat_messages (group_id, sender, message) VALUES ($1, 'ai', $2) RETURNING id, group_id, sender, message, created_at`,
		groupID, aiMessage,
	).Scan(&newAiMsg.ID, &newAiMsg.GroupID, &newAiMsg.Sender, &newAiMsg.Message, &newAiMsg.CreatedAt)
	if err != nil {
		http.Error(w, "gagal simpan pesan AI: "+err.Error(), 500)
		return
	}

	// 6. Kirim balasan AI ke user
	writeJSON(w, http.StatusCreated, newAiMsg)
}

// [BARU] Helper untuk membangun prompt yang kaya konteks untuk Gemini
func (s *Server) buildChatPrompt(ctx context.Context, groupID int64, newUserMessage string) (string, error) {
	var b strings.Builder

	// Peran dan instruksi untuk AI
	b.WriteString("Anda adalah Akai, asisten AI yang ramah dan membantu untuk layanan Lost & Found. Tugas Anda adalah berinteraksi dengan pengguna mengenai insiden barang yang tertinggal atau ditemukan berdasarkan data dari sistem CCTV. Gunakan informasi yang diberikan untuk menjawab pertanyaan pengguna. Berikan jawaban yang jelas, ringkas, dan empatik. Jangan mengarang informasi di luar data yang diberikan.\n\n")

	// Ambil data insiden (grup)
	var g GroupRow
	err := s.DB.QueryRow(ctx, `SELECT owner_name, item_name, location_label, status FROM snapshot_groups WHERE id=$1`, groupID).Scan(&g.OwnerName, &g.ItemName, &g.LocationLabel, &g.Status)
	if err != nil {
		return "", err
	}
	b.WriteString(fmt.Sprintf("--- DATA INSIDEN (ID: %d) ---\n", groupID))
	b.WriteString(fmt.Sprintf("- Nama Pemilik: %s\n", fallbackPtr(g.OwnerName, "Belum diketahui")))
	b.WriteString(fmt.Sprintf("- Nama Barang: %s\n", fallbackPtr(g.ItemName, "Belum diketahui")))
	b.WriteString(fmt.Sprintf("- Lokasi Terakhir: %s\n", fallbackPtr(g.LocationLabel, "Belum diketahui")))
	b.WriteString(fmt.Sprintf("- Status Saat Ini: %s\n\n", g.Status))

	// Ambil kronologi kejadian
	erows, err := s.DB.Query(ctx, `SELECT occurred_at, message FROM snapshot_events WHERE group_id=$1 ORDER BY occurred_at ASC`, groupID)
	if err != nil {
		return "", err
	}
	defer erows.Close()
	b.WriteString("--- KRONOLOGI KEJADIAN (DARI CCTV) ---\n")
	for erows.Next() {
		var ts time.Time
		var msg string
		if err := erows.Scan(&ts, &msg); err == nil {
			b.WriteString(fmt.Sprintf("- %s: %s\n", ts.In(time.FixedZone("WIB", 7*3600)).Format("15:04:05"), msg))
		}
	}
	b.WriteString("\n")

	// Ambil riwayat chat sebelumnya
	crows, err := s.DB.Query(ctx, `SELECT sender, message FROM chat_messages WHERE group_id=$1 ORDER BY created_at ASC`, groupID)
	if err != nil {
		return "", err
	}
	defer crows.Close()
	b.WriteString("--- RIWAYAT PERCAKAPAN SEBELUMNYA ---\n")
	for crows.Next() {
		var sender, msg string
		if err := crows.Scan(&sender, &msg); err == nil {
			if sender == "user" {
				b.WriteString(fmt.Sprintf("Pengguna: %s\n", msg))
			} else {
				b.WriteString(fmt.Sprintf("Akai: %s\n", msg))
			}
		}
	}
	b.WriteString("\n")

	// Tambahkan pesan baru dari user dan instruksi akhir
	b.WriteString("--- PESAN BARU DARI PENGGUNA ---\n")
	b.WriteString(newUserMessage)
	b.WriteString("\n\n---\n")
	b.WriteString("Tugas: Berdasarkan semua informasi di atas, berikan balasan yang sesuai sebagai Akai.")

	return b.String(), nil
}

// -------- Helpers --------
func nullify(s string) any {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return s
}
func nullifyBytes(b []byte) any {
	if len(b) == 0 {
		return nil
	}
	return b
}
func decodeBase64(s string) ([]byte, error) {
	if s == "" {
		return nil, nil
	}
	return base64.StdEncoding.DecodeString(s)
}
func encodeBase64(b []byte) *string {
	if len(b) == 0 {
		return nil
	}
	s := base64.StdEncoding.EncodeToString(b)
	return &s
}
func ifnz(s string) any {
	if s == "" {
		return nil
	}
	return s
}
func fallback(s string, def string) string {
	if strings.TrimSpace(s) == "" {
		return def
	}
	return s
}
func fallbackPtr(s *string, def string) string {
	if s == nil || strings.TrimSpace(*s) == "" {
		return def
	}
	return *s
}
func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}
func logRequest(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s -> %s in %v", r.Method, r.URL.Path, r.RemoteAddr, time.Since(start))
	})
}