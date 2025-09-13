package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/generative-ai-go/genai"
	"github.com/gorilla/handlers"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
	"google.golang.org/api/option"
)

// Server struct
type Server struct {
	DB     *pgxpool.Pool
	Gemini *genai.GenerativeModel
}

// ----- Structs -----

type Laporan struct {
	ID                int64     `json:"id"`
	JenisLaporan      string    `json:"jenis_laporan"`
	NamaPelapor       string    `json:"nama_pelapor"`
	NamaBarang        *string   `json:"nama_barang,omitempty"`
	Deskripsi         *string   `json:"deskripsi,omitempty"`
	Lokasi            *string   `json:"lokasi,omitempty"`
	GambarBarangB64   *string   `json:"gambar_barang_b64,omitempty"`
	Status            string    `json:"status"`
	LaporanPasanganID *int64    `json:"laporan_pasangan_id,omitempty"`
	WaktuLaporan      time.Time `json:"waktu_laporan"`
	UpdatedAt         time.Time `json:"updated_at"`
}

type ChatMessage struct {
	ID                int64     `json:"id"`
	LaporanID         int64     `json:"laporan_id"`
	Sender            string    `json:"sender"`
	Message           string    `json:"message"`
	CreatedAt         time.Time `json:"created_at"`
	AttachmentLaporan *Laporan  `json:"attachment_laporan,omitempty"`
}

type CCTVIncident struct {
	ID               int64     `json:"id"`
	GroupKey         string    `json:"group_key"`
	OwnerName        *string   `json:"owner_name,omitempty"`
	ItemName         *string   `json:"item_name,omitempty"`
	Status           string    `json:"status"` // 'unattended', 'taken', 'resolved_owner', 'resolved_secured'
	LastSnapshotB64  *string   `json:"last_snapshot_b64,omitempty"`
	LaporanTerkaitID *int64    `json:"laporan_terkait_id,omitempty"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
}

type CCTVEvent struct {
	ID         int64     `json:"id"`
	IncidentID int64     `json:"incident_id"`
	EventType  string    `json:"event_type"`
	Message    string    `json:"message"`
	OccurredAt time.Time `json:"occurred_at"`
}

// Payloads
type LaporanPayload struct {
	JenisLaporan    string `json:"jenis_laporan"`
	NamaPelapor     string `json:"nama_pelapor"`
	NamaBarang      string `json:"nama_barang"`
	Deskripsi       string `json:"deskripsi"`
	Lokasi          string `json:"lokasi"`
	GambarBarangB64 string `json:"gambar_barang_b64,omitempty"`
}
type ChatPayload struct {
	Message  string `json:"message"`
	ImageB64 string `json:"image_b64,omitempty"`
}

type NotifyPayload struct {
	GroupKey    string    `json:"group_key"`
	EventType   string    `json:"event_type"`
	OwnerName   string    `json:"owner_name,omitempty"`
	ItemName    string    `json:"item_name,omitempty"`
	Message     string    `json:"message,omitempty"`
	SnapshotB64 string    `json:"snapshot_b64,omitempty"`
	Timestamp   time.Time `json:"timestamp"`
}

type StatusUpdatePayload struct {
	Status string `json:"status"`
}

func main() {
	_ = godotenv.Load()

	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		log.Fatal("DATABASE_URL env kosong. Contoh: postgres://user:pass@localhost:5432/db_lostfound?sslmode=disable")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		log.Fatalf("Gagal konek database: %v", err)
	}
	defer pool.Close()

	if err := runMigrations(ctx, pool); err != nil {
		log.Fatalf("Gagal migrasi: %v", err)
	}

	geminiAPIKey := os.Getenv("GOOGLE_API_KEY")
	if geminiAPIKey == "" {
		log.Fatal("GOOGLE_API_KEY env kosong.")
	}
	geminiClient, err := genai.NewClient(ctx, option.WithAPIKey(geminiAPIKey))
	if err != nil {
		log.Fatalf("Gagal membuat klien Gemini: %v", err)
	}
	defer geminiClient.Close()
	geminiModel := geminiClient.GenerativeModel("gemini-1.5-flash")

	s := &Server{
		DB:     pool,
		Gemini: geminiModel,
	}

	mux := http.NewServeMux()

	mux.HandleFunc("POST /laporan", s.handleBuatLaporan)
	mux.HandleFunc("GET /laporan", s.handleGetLaporan)
	mux.HandleFunc("GET /laporan/{id}", s.handleGetDetailLaporan)
	mux.HandleFunc("POST /laporan/{id}/chat", s.handleChat)

	mux.HandleFunc("POST /notify", s.handleNotify)
	mux.HandleFunc("GET /incidents", s.handleGetIncidents)
	mux.HandleFunc("GET /incidents/{id}", s.handleGetIncidentDetail)
	mux.HandleFunc("POST /incidents/{id}/create-report", s.handleCreateReportFromIncident)
	mux.HandleFunc("PUT /incidents/{id}/status", s.handleUpdateIncidentStatus)

	port := os.Getenv("PORT")
	if port == "" {
		port = "3000"
	}
	addr := ":" + port
	log.Println("Server berjalan di", addr)

	allowedOrigins := handlers.AllowedOrigins([]string{"*"})
	allowedMethods := handlers.AllowedMethods([]string{"GET", "POST", "PUT", "DELETE", "OPTIONS"})
	allowedHeaders := handlers.AllowedHeaders([]string{"Content-Type", "Authorization"})
	handler := logRequest(handlers.CORS(allowedOrigins, allowedMethods, allowedHeaders)(mux))

	log.Fatal(http.ListenAndServe(addr, handler))
}

func runMigrations(ctx context.Context, db *pgxpool.Pool) error {
	laporanSQL := `
    CREATE TABLE IF NOT EXISTS laporan (
        id BIGSERIAL PRIMARY KEY,
        jenis_laporan TEXT NOT NULL,
        nama_pelapor TEXT NOT NULL,
        nama_barang TEXT,
        deskripsi TEXT,
        lokasi TEXT,
        gambar_barang_b64 TEXT,
        status TEXT NOT NULL DEFAULT 'terbuka',
        laporan_pasangan_id BIGINT,
        waktu_laporan TIMESTAMP WITH TIME ZONE DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS chat_messages (
        id BIGSERIAL PRIMARY KEY,
        laporan_id BIGINT REFERENCES laporan(id) ON DELETE CASCADE,
        sender TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    );`
	_, err := db.Exec(ctx, laporanSQL)
	if err != nil {
		return err
	}

	cctvSQL := `
    CREATE TABLE IF NOT EXISTS cctv_incidents (
        id BIGSERIAL PRIMARY KEY,
        group_key TEXT UNIQUE NOT NULL,
        owner_name TEXT,
        item_name TEXT,
        status TEXT NOT NULL,
        last_snapshot_b64 TEXT,
        laporan_terkait_id BIGINT REFERENCES laporan(id) ON DELETE SET NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS cctv_events (
        id BIGSERIAL PRIMARY KEY,
        incident_id BIGINT REFERENCES cctv_incidents(id) ON DELETE CASCADE,
        event_type TEXT NOT NULL,
        message TEXT NOT NULL,
        occurred_at TIMESTAMP WITH TIME ZONE NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_cctv_incidents_status ON cctv_incidents(status);
    `
	_, err = db.Exec(ctx, cctvSQL)
	return err
}

func (s *Server) handleBuatLaporan(w http.ResponseWriter, r *http.Request) {
	var p LaporanPayload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if p.JenisLaporan == "" || p.NamaPelapor == "" || p.NamaBarang == "" {
		http.Error(w, "jenis_laporan, nama_pelapor, dan nama_barang wajib diisi", http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	var newLaporan Laporan
	err := s.DB.QueryRow(ctx, `
        INSERT INTO laporan (jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64)
        VALUES ($1, $2, $3, $4, $5, $6) 
        RETURNING id, jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64, status, laporan_pasangan_id, waktu_laporan, updated_at
    `, p.JenisLaporan, p.NamaPelapor, p.NamaBarang, p.Deskripsi, p.Lokasi, nullify(p.GambarBarangB64)).Scan(
		&newLaporan.ID, &newLaporan.JenisLaporan, &newLaporan.NamaPelapor, &newLaporan.NamaBarang, &newLaporan.Deskripsi, &newLaporan.Lokasi, &newLaporan.GambarBarangB64, &newLaporan.Status, &newLaporan.LaporanPasanganID, &newLaporan.WaktuLaporan, &newLaporan.UpdatedAt,
	)

	if err != nil {
		http.Error(w, "gagal menyimpan laporan: "+err.Error(), 500)
		return
	}

	writeJSON(w, http.StatusCreated, newLaporan)
}

func (s *Server) handleGetLaporan(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	rows, err := s.DB.Query(ctx, `
        SELECT id, jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64, status, laporan_pasangan_id, waktu_laporan, updated_at
        FROM laporan ORDER BY waktu_laporan DESC LIMIT 50
    `)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer rows.Close()

	laporans := []Laporan{}
	for rows.Next() {
		var l Laporan
		if err := rows.Scan(&l.ID, &l.JenisLaporan, &l.NamaPelapor, &l.NamaBarang, &l.Deskripsi, &l.Lokasi, &l.GambarBarangB64, &l.Status, &l.LaporanPasanganID, &l.WaktuLaporan, &l.UpdatedAt); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		laporans = append(laporans, l)
	}

	writeJSON(w, http.StatusOK, laporans)
}

func (s *Server) handleGetDetailLaporan(w http.ResponseWriter, r *http.Request) {
	id, _ := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if id == 0 {
		http.Error(w, "ID laporan tidak valid", 400)
		return
	}

	ctx := r.Context()
	var l Laporan
	err := s.DB.QueryRow(ctx, `
        SELECT id, jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64, status, laporan_pasangan_id, waktu_laporan, updated_at
        FROM laporan WHERE id=$1
    `, id).Scan(&l.ID, &l.JenisLaporan, &l.NamaPelapor, &l.NamaBarang, &l.Deskripsi, &l.Lokasi, &l.GambarBarangB64, &l.Status, &l.LaporanPasanganID, &l.WaktuLaporan, &l.UpdatedAt)
	if err != nil {
		http.Error(w, "laporan tidak ditemukan", 404)
		return
	}

	rows, err := s.DB.Query(ctx, `SELECT id, laporan_id, sender, message, created_at FROM chat_messages WHERE laporan_id=$1 ORDER BY created_at ASC`, id)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer rows.Close()

	chatHistory := []ChatMessage{}
	for rows.Next() {
		var msg ChatMessage
		if err := rows.Scan(&msg.ID, &msg.LaporanID, &msg.Sender, &msg.Message, &msg.CreatedAt); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		chatHistory = append(chatHistory, msg)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"laporan":       l,
		"log_aktivitas": chatHistory,
	})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	laporanID, _ := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if laporanID == 0 {
		http.Error(w, "id laporan tidak valid", 400)
		return
	}

	ctx := r.Context()
	var p ChatPayload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(p.Message) == "" && p.ImageB64 == "" {
		http.Error(w, "pesan atau gambar tidak boleh kosong", http.StatusBadRequest)
		return
	}

	userMessage := p.Message
	if p.ImageB64 != "" {
		userMessage = strings.TrimSpace(userMessage + " [pengguna melampirkan sebuah gambar]")
	}
	_, err := s.DB.Exec(ctx, `INSERT INTO chat_messages (laporan_id, sender, message) VALUES ($1, 'user', $2)`, laporanID, userMessage)
	if err != nil {
		http.Error(w, "gagal simpan pesan user: "+err.Error(), 500)
		return
	}

	promptParts, err := s.buildAdvancedChatPrompt(ctx, laporanID, p.Message, p.ImageB64)
	if err != nil {
		http.Error(w, "gagal bangun prompt: "+err.Error(), 500)
		return
	}

	resp, err := s.Gemini.GenerateContent(ctx, promptParts...)
	if err != nil {
		log.Printf("Error Gemini API: %v", err)
		http.Error(w, "gagal komunikasi dengan AI: "+err.Error(), 502)
		return
	}

	aiMessageText := extractTextFromResponse(resp)
	if aiMessageText == "" {
		aiMessageText = "Maaf, saya tidak bisa memberikan balasan saat ini."
	}

	var newAiMsg ChatMessage
	err = s.DB.QueryRow(ctx,
		`INSERT INTO chat_messages (laporan_id, sender, message) VALUES ($1, 'ai', $2) 
         RETURNING id, laporan_id, sender, message, created_at`,
		laporanID, aiMessageText,
	).Scan(&newAiMsg.ID, &newAiMsg.LaporanID, &newAiMsg.Sender, &newAiMsg.Message, &newAiMsg.CreatedAt)
	if err != nil {
		http.Error(w, "gagal simpan pesan AI: "+err.Error(), 500)
		return
	}

	matchID := findReportIDInText(aiMessageText)
	if matchID > 0 {
		attachedLaporan, err := s.fetchLaporanByID(ctx, matchID)
		if err == nil {
			newAiMsg.AttachmentLaporan = attachedLaporan
		}
	}

	writeJSON(w, http.StatusCreated, newAiMsg)
}

func (s *Server) buildAdvancedChatPrompt(ctx context.Context, laporanID int64, newUserMessage string, newUserImageB64 string) ([]genai.Part, error) {
	var parts []genai.Part
	var b strings.Builder

	b.WriteString("Anda adalah Akai, asisten AI spesialis Lost & Found yang cerdas dan proaktif. Aturan Anda:\n")
	b.WriteString("1. Analisis pesan & gambar baru dari pengguna.\n")
	b.WriteString("2. Bandingkan detail laporan saat ini dengan laporan lain di 'Konteks Tambahan'.\n")
	b.WriteString("3. Jika ada kemungkinan KECOCOKAN (match), beritahu pengguna dengan format WAJIB: 'Saya menemukan kemungkinan kecocokan dengan laporan #${ID_LAPORAN_MATCH}. Berikut detailnya...'. Jangan gunakan format lain.\n")
	b.WriteString("4. Jika tidak ada match, balas pertanyaan pengguna secara normal dan ramah.\n\n")

	var l Laporan
	err := s.DB.QueryRow(ctx, `SELECT jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi FROM laporan WHERE id=$1`, laporanID).Scan(
		&l.JenisLaporan, &l.NamaPelapor, &l.NamaBarang, &l.Deskripsi, &l.Lokasi)
	if err != nil {
		return nil, err
	}

	b.WriteString(fmt.Sprintf("--- LAPORAN UTAMA (ID: %d) ---\n", laporanID))
	b.WriteString(fmt.Sprintf("- Jenis: %s\n- Pelapor: %s\n- Barang: %s\n- Deskripsi: %s\n\n", l.JenisLaporan, l.NamaPelapor, strFallback(l.NamaBarang, "N/A"), strFallback(l.Deskripsi, "N/A")))

	jenisLaporanUntukDicari := "kehilangan"
	if l.JenisLaporan == "kehilangan" {
		jenisLaporanUntukDicari = "penemuan"
	}

	b.WriteString(fmt.Sprintf("--- KONTEKS TAMBAHAN: 5 LAPORAN '%s' TERBARU ---\n", jenisLaporanUntukDicari))
	matchRows, _ := s.DB.Query(ctx,
		`SELECT id, nama_barang, deskripsi, lokasi, gambar_barang_b64 FROM laporan WHERE jenis_laporan=$1 AND status='terbuka' ORDER BY waktu_laporan DESC LIMIT 5`,
		jenisLaporanUntukDicari)

	foundMatches := false
	for matchRows.Next() {
		foundMatches = true
		var m struct {
			ID                            int64
			NamaBarang, Deskripsi, Lokasi *string
			GambarB64                     *string
		}
		if err := matchRows.Scan(&m.ID, &m.NamaBarang, &m.Deskripsi, &m.Lokasi, &m.GambarB64); err == nil {
			b.WriteString(fmt.Sprintf("- Laporan #%d: Barang: %s, Deskripsi: %s.", m.ID, strFallback(m.NamaBarang, "?"), strFallback(m.Deskripsi, "?")))
			if m.GambarB64 != nil && *m.GambarB64 != "" {
				b.WriteString(" [ADA GAMBAR]\n")
			} else {
				b.WriteString(" [TIDAK ADA GAMBAR]\n")
			}
		}
	}
	if !foundMatches {
		b.WriteString("Tidak ada laporan relevan yang ditemukan saat ini.\n")
	}
	b.WriteString("\n")

	b.WriteString("--- PESAN BARU DARI PENGGUNA ---\n")
	b.WriteString(newUserMessage + "\n\n---\n")
	b.WriteString("Tugas Anda: Berikan balasan sebagai Akai berdasarkan semua data di atas.")

	parts = append(parts, genai.Text(b.String()))
	if newUserImageB64 != "" {
		imgBytes, err := base64.StdEncoding.DecodeString(newUserImageB64)
		if err == nil {
			parts = append(parts, genai.ImageData("jpeg", imgBytes))
		}
	}
	return parts, nil
}

func (s *Server) handleNotify(w http.ResponseWriter, r *http.Request) {
	var p NotifyPayload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if p.GroupKey == "" || p.EventType == "" {
		http.Error(w, "group_key dan event_type wajib ada", http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	tx, err := s.DB.Begin(ctx)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer tx.Rollback(ctx)

	var incidentID int64
	var currentStatus string

	err = tx.QueryRow(ctx, "SELECT id, status FROM cctv_incidents WHERE group_key=$1", p.GroupKey).Scan(&incidentID, &currentStatus)

	if err != nil {
		if p.EventType != "unattended" {
			http.Error(w, "insiden belum ada, hanya event 'unattended' yang diterima", http.StatusBadRequest)
			return
		}
		err = tx.QueryRow(ctx, `
            INSERT INTO cctv_incidents (group_key, owner_name, item_name, status, last_snapshot_b64)
            VALUES ($1, $2, $3, 'unattended', $4) RETURNING id
        `, p.GroupKey, nullify(p.OwnerName), nullify(p.ItemName), nullify(p.SnapshotB64)).Scan(&incidentID)
		if err != nil {
			http.Error(w, "gagal buat insiden baru: "+err.Error(), 500)
			return
		}
	} else {
		updateQuery := "UPDATE cctv_incidents SET updated_at=now()"
		args := []interface{}{}
		argID := 1

		if p.EventType == "item_taken_by_other" && currentStatus == "unattended" {
			updateQuery += ", status=$" + strconv.Itoa(argID)
			args = append(args, "taken")
			argID++
		}
		if p.SnapshotB64 != "" {
			updateQuery += ", last_snapshot_b64=$" + strconv.Itoa(argID)
			args = append(args, p.SnapshotB64)
			argID++
		}

		updateQuery += " WHERE id=$" + strconv.Itoa(argID)
		args = append(args, incidentID)

		_, err = tx.Exec(ctx, updateQuery, args...)
		if err != nil {
			http.Error(w, "gagal update insiden: "+err.Error(), 500)
			return
		}
	}

	finalMessage := p.Message
	if finalMessage == "" {
		finalMessage = p.EventType
	}

	_, err = tx.Exec(ctx, `
        INSERT INTO cctv_events (incident_id, event_type, message, occurred_at) VALUES ($1, $2, $3, $4)
    `, incidentID, p.EventType, finalMessage, p.Timestamp)
	if err != nil {
		http.Error(w, "gagal catat event: "+err.Error(), 500)
		return
	}

	if err := tx.Commit(ctx); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "incident_id": strconv.FormatInt(incidentID, 10)})
}

func (s *Server) handleGetIncidents(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	rows, err := s.DB.Query(ctx, "SELECT id, group_key, owner_name, item_name, status, created_at, updated_at, laporan_terkait_id FROM cctv_incidents ORDER BY updated_at DESC")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer rows.Close()

	incidents := []CCTVIncident{}
	for rows.Next() {
		var i CCTVIncident
		if err := rows.Scan(&i.ID, &i.GroupKey, &i.OwnerName, &i.ItemName, &i.Status, &i.CreatedAt, &i.UpdatedAt, &i.LaporanTerkaitID); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		incidents = append(incidents, i)
	}
	writeJSON(w, http.StatusOK, incidents)
}

func (s *Server) handleGetIncidentDetail(w http.ResponseWriter, r *http.Request) {
	id, _ := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if id == 0 {
		http.Error(w, "ID insiden tidak valid", 400)
		return
	}
	ctx := r.Context()
	var incident CCTVIncident
	err := s.DB.QueryRow(ctx, "SELECT id, group_key, owner_name, item_name, status, created_at, updated_at, last_snapshot_b64, laporan_terkait_id FROM cctv_incidents WHERE id=$1", id).Scan(
		&incident.ID, &incident.GroupKey, &incident.OwnerName, &incident.ItemName, &incident.Status, &incident.CreatedAt, &incident.UpdatedAt, &incident.LastSnapshotB64, &incident.LaporanTerkaitID)
	if err != nil {
		http.Error(w, "insiden tidak ditemukan", 404)
		return
	}

	rows, err := s.DB.Query(ctx, "SELECT id, incident_id, event_type, message, occurred_at FROM cctv_events WHERE incident_id=$1 ORDER BY occurred_at ASC", id)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer rows.Close()

	events := []CCTVEvent{}
	for rows.Next() {
		var e CCTVEvent
		if err := rows.Scan(&e.ID, &e.IncidentID, &e.EventType, &e.Message, &e.OccurredAt); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		events = append(events, e)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"incident": incident,
		"events":   events,
	})
}

func (s *Server) handleCreateReportFromIncident(w http.ResponseWriter, r *http.Request) {
	incidentID, _ := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if incidentID == 0 {
		http.Error(w, "ID insiden tidak valid", 400)
		return
	}

	ctx := r.Context()
	tx, err := s.DB.Begin(ctx)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer tx.Rollback(ctx)

	var incident CCTVIncident
	err = tx.QueryRow(ctx, "SELECT owner_name, item_name, last_snapshot_b64, status FROM cctv_incidents WHERE id=$1", incidentID).Scan(
		&incident.OwnerName, &incident.ItemName, &incident.LastSnapshotB64, &incident.Status)
	if err != nil {
		http.Error(w, "insiden tidak ditemukan", 404)
		return
	}

	if incident.Status != "taken" {
		http.Error(w, "hanya insiden dengan status 'taken' (Hilang) yang bisa dibuat laporan", http.StatusBadRequest)
		return
	}

	var newLaporan Laporan
	err = tx.QueryRow(ctx, `
        INSERT INTO laporan (jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64)
        VALUES ('kehilangan', $1, $2, $3, $4, $5)
        RETURNING id, jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64, status, waktu_laporan, updated_at
    `,
		incident.OwnerName, incident.ItemName, "Barang hilang terdeteksi oleh CCTV", "Lokasi terakhir dari CCTV", incident.LastSnapshotB64,
	).Scan(&newLaporan.ID, &newLaporan.JenisLaporan, &newLaporan.NamaPelapor, &newLaporan.NamaBarang, &newLaporan.Deskripsi, &newLaporan.Lokasi, &newLaporan.GambarBarangB64, &newLaporan.Status, &newLaporan.WaktuLaporan, &newLaporan.UpdatedAt)
	if err != nil {
		http.Error(w, "gagal membuat laporan baru: "+err.Error(), 500)
		return
	}

	_, err = tx.Exec(ctx, "UPDATE cctv_incidents SET laporan_terkait_id=$1 WHERE id=$2", newLaporan.ID, incidentID)
	if err != nil {
		http.Error(w, "gagal menautkan insiden ke laporan: "+err.Error(), 500)
		return
	}

	if err := tx.Commit(ctx); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	writeJSON(w, http.StatusCreated, newLaporan)
}

func (s *Server) handleUpdateIncidentStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		http.Error(w, "Method tidak diizinkan", http.StatusMethodNotAllowed)
		return
	}

	incidentID, _ := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if incidentID == 0 {
		http.Error(w, "ID insiden tidak valid", 400)
		return
	}

	var p StatusUpdatePayload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, "Payload tidak valid: "+err.Error(), http.StatusBadRequest)
		return
	}

	validStatuses := map[string]bool{
		"resolved_owner":   true,
		"resolved_secured": true,
	}
	if !validStatuses[p.Status] {
		http.Error(w, "Nilai status tidak valid", http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	cmdTag, err := s.DB.Exec(ctx,
		"UPDATE cctv_incidents SET status=$1, updated_at=now() WHERE id=$2",
		p.Status, incidentID,
	)

	if err != nil {
		http.Error(w, "Gagal update status: "+err.Error(), 500)
		return
	}
	if cmdTag.RowsAffected() == 0 {
		http.Error(w, "Insiden tidak ditemukan", 404)
		return
	}

	updatedIncident, err := s.fetchIncidentByID(ctx, incidentID)
	if err != nil {
		http.Error(w, "Gagal mengambil data terbaru: "+err.Error(), 500)
		return
	}

	writeJSON(w, http.StatusOK, updatedIncident)
}

func (s *Server) fetchLaporanByID(ctx context.Context, id int64) (*Laporan, error) {
	var l Laporan
	err := s.DB.QueryRow(ctx, `
        SELECT id, jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64, status, laporan_pasangan_id, waktu_laporan, updated_at
        FROM laporan WHERE id=$1
    `, id).Scan(&l.ID, &l.JenisLaporan, &l.NamaPelapor, &l.NamaBarang, &l.Deskripsi, &l.Lokasi, &l.GambarBarangB64, &l.Status, &l.LaporanPasanganID, &l.WaktuLaporan, &l.UpdatedAt)
	if err != nil {
		return nil, err
	}
	return &l, nil
}

func (s *Server) fetchIncidentByID(ctx context.Context, id int64) (*CCTVIncident, error) {
	var i CCTVIncident
	err := s.DB.QueryRow(ctx, "SELECT id, group_key, owner_name, item_name, status, created_at, updated_at, last_snapshot_b64, laporan_terkait_id FROM cctv_incidents WHERE id=$1", id).Scan(
		&i.ID, &i.GroupKey, &i.OwnerName, &i.ItemName, &i.Status, &i.CreatedAt, &i.UpdatedAt, &i.LastSnapshotB64, &i.LaporanTerkaitID)
	if err != nil {
		return nil, err
	}
	return &i, nil
}

var reportIDRegex = regexp.MustCompile(`laporan #(\d+)`)

func findReportIDInText(text string) int64 {
	matches := reportIDRegex.FindStringSubmatch(text)
	if len(matches) > 1 {
		id, _ := strconv.ParseInt(matches[1], 10, 64)
		return id
	}
	return 0
}

func extractTextFromResponse(resp *genai.GenerateContentResponse) string {
	var text strings.Builder
	if resp != nil && len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
		for _, part := range resp.Candidates[0].Content.Parts {
			if txt, ok := part.(genai.Text); ok {
				text.WriteString(string(txt))
			}
		}
	}
	return text.String()
}

func nullify(s string) any {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return s
}

func strFallback(s *string, def string) string {
	if s == nil || *s == "" {
		return def
	}
	return *s
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func logRequest(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s -> took %v", r.Method, r.URL.Path, time.Since(start))
	})
}