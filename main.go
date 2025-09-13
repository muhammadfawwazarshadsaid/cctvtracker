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
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
	"golang.org/x/crypto/bcrypt"
	"google.golang.org/api/option"
)

// Server struct
type Server struct {
	DB     *pgxpool.Pool
	Gemini *genai.GenerativeModel
}

// ----- Structs -----

// User struct
type User struct {
	ID           int64     `json:"id"`
	Name         string    `json:"name"`
	Email        *string   `json:"email,omitempty"`
	PasswordHash string    `json:"-"` // Disembunyikan dari output JSON
	CreatedAt    time.Time `json:"created_at"`
}

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
	ID                int64     `json:"id"`
	GroupKey          string    `json:"group_key"`
	OwnerID           *int64    `json:"owner_id,omitempty"`
	OwnerName         *string   `json:"owner_name,omitempty"`
	ItemName          *string   `json:"item_name,omitempty"`
	LastKnownLocation *string   `json:"last_known_location,omitempty"`
	Status            string    `json:"status"`
	LastSnapshotB64   *string   `json:"last_snapshot_b64,omitempty"`
	LaporanTerkaitID  *int64    `json:"laporan_terkait_id,omitempty"`
	CreatedAt         time.Time `json:"created_at"`
	UpdatedAt         time.Time `json:"updated_at"`
}

type CCTVEvent struct {
	ID         int64     `json:"id"`
	IncidentID int64     `json:"incident_id"`
	EventType  string    `json:"event_type"`
	Message    string    `json:"message"`
	OccurredAt time.Time `json:"occurred_at"`
}

// ----- Payloads -----

type LaporanPayload struct {
	JenisLaporan    string `json:"jenis_laporan"`
	NamaPelapor     string `json:"nama_pelapor"`
	NamaBarang      string `json:"nama_barang"`
	Deskripsi       string `json:"deskripsi"`
	Lokasi          string `json:"lokasi"`
	GambarBarangB64 string `json:"gambar_barang_b64,omitempty"`
}
type ChatPayload struct {
	Message        string `json:"message"`
	ImageB64       string `json:"image_b64,omitempty"`
	CctvIncidentID *int64 `json:"cctv_incident_id,omitempty"`
}

type NotifyPayload struct {
	GroupKey    string    `json:"group_key"`
	EventType   string    `json:"event_type"`
	OwnerName   string    `json:"owner_name,omitempty"`
	ItemName    string    `json:"item_name,omitempty"`
	Message     string    `json:"message,omitempty"`
	Location    string    `json:"location,omitempty"`
	SnapshotB64 string    `json:"snapshot_b64,omitempty"`
	Timestamp   time.Time `json:"timestamp"`
}

type StatusUpdatePayload struct {
	Status string `json:"status"`
}

type UserPayload struct {
	Name string `json:"name"`
}

type LoginPayload struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

func main() {
	_ = godotenv.Load()

	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		log.Fatal("DATABASE_URL env kosong.")
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

	// Endpoints Laporan & Chat
	mux.HandleFunc("POST /laporan", s.handleBuatLaporan)
	mux.HandleFunc("GET /laporan", s.handleGetLaporan)
	mux.HandleFunc("GET /laporan/{id}", s.handleGetDetailLaporan)
	mux.HandleFunc("POST /laporan/{id}/chat", s.handleChat)

	// Endpoints Insiden CCTV
	mux.HandleFunc("POST /notify", s.handleNotify)
	mux.HandleFunc("GET /incidents", s.handleGetIncidents)
	mux.HandleFunc("GET /incidents/{id}", s.handleGetIncidentDetail)
	mux.HandleFunc("POST /incidents/{id}/create-report", s.handleCreateReportFromIncident)
	mux.HandleFunc("PUT /incidents/{id}/status", s.handleUpdateIncidentStatus)

	// Endpoints User
	mux.HandleFunc("POST /login", s.handleLogin)
	mux.HandleFunc("POST /users", s.handleCreateUser)
	mux.HandleFunc("GET /users", s.handleGetUsers)

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
	if _, err := db.Exec(ctx, laporanSQL); err != nil {
		return fmt.Errorf("migrasi tabel laporan gagal: %w", err)
	}

	usersSQL := `
    CREATE TABLE IF NOT EXISTS users (
        id BIGSERIAL PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        email TEXT UNIQUE,
        password_hash TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    );`
	if _, err := db.Exec(ctx, usersSQL); err != nil {
		return fmt.Errorf("migrasi tabel users gagal: %w", err)
	}

	cctvSQL := `
    CREATE TABLE IF NOT EXISTS cctv_incidents (
        id BIGSERIAL PRIMARY KEY,
        group_key TEXT UNIQUE NOT NULL,
        owner_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
        item_name TEXT,
        last_known_location TEXT,
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
	_, err := db.Exec(ctx, cctvSQL)
	if err != nil {
		return fmt.Errorf("migrasi tabel cctv gagal: %w", err)
	}
	return nil
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
		http.Error(w, "gagal memulai transaksi: "+err.Error(), 500)
		return
	}
	defer tx.Rollback(ctx)

	var ownerID *int64
	if p.OwnerName != "" {
		var id int64
		// Coba cari user berdasarkan nama
		err := tx.QueryRow(ctx, "SELECT id FROM users WHERE name = $1", p.OwnerName).Scan(&id)

		// Jika user tidak ditemukan, coba buat baru
		if err != nil {
			if err == pgx.ErrNoRows {
				defaultPassword := "12345678"
				hashedPassword, _ := bcrypt.GenerateFromPassword([]byte(defaultPassword), bcrypt.DefaultCost)

				safeName := strings.ReplaceAll(strings.ToLower(p.OwnerName), " ", "")
				email := fmt.Sprintf("%s@track.ai", safeName)

				// Coba INSERT user baru
				errInsert := tx.QueryRow(ctx, `
                    INSERT INTO users (name, email, password_hash)
                    VALUES ($1, $2, $3)
                    RETURNING id
                `, p.OwnerName, email, string(hashedPassword)).Scan(&id)

				if errInsert != nil {
					// Jika INSERT gagal karena duplikat (baik nama atau email)
					if strings.Contains(errInsert.Error(), "duplicate key value") {
						log.Printf("Gagal insert user duplikat '%s', mencoba mencari ulang...", p.OwnerName)
						// Coba cari lagi berdasarkan nama, karena mungkin sudah dibuat oleh proses lain
						errSearchAgain := tx.QueryRow(ctx, "SELECT id FROM users WHERE name = $1", p.OwnerName).Scan(&id)
						if errSearchAgain != nil {
							http.Error(w, "gagal mencari user setelah gagal insert: "+errSearchAgain.Error(), 500)
							return
						}
						log.Printf("User '%s' ditemukan dengan ID %d pada percobaan kedua.", p.OwnerName, id)
					} else {
						// Error lain saat insert
						http.Error(w, "gagal membuat user baru secara otomatis: "+errInsert.Error(), 500)
						return
					}
				} else {
					log.Printf("Akun login baru untuk '%s' dibuat dengan email '%s' dan ID %d", p.OwnerName, email, id)
				}
			} else {
				// Error lain saat SELECT awal
				http.Error(w, "gagal mencari user: "+err.Error(), 500)
				return
			}
		}
		ownerID = &id
	}

	var incidentID int64
	var currentStatus string
	err = tx.QueryRow(ctx, "SELECT id, status FROM cctv_incidents WHERE group_key=$1 FOR UPDATE", p.GroupKey).Scan(&incidentID, &currentStatus)

	if err != nil { // Insiden belum ada
		newStatus := "created"
		if p.EventType == "attended" {
			newStatus = "attended"
		} else if p.EventType == "unattended" {
			newStatus = "unattended"
		} else {
			http.Error(w, "event pertama harus 'attended' atau 'unattended'", http.StatusBadRequest)
			return
		}
		err = tx.QueryRow(ctx, `
            INSERT INTO cctv_incidents (group_key, owner_id, item_name, last_known_location, status, last_snapshot_b64)
            VALUES ($1, $2, $3, $4, $5, $6) RETURNING id
        `, p.GroupKey, ownerID, nullify(p.ItemName), nullify(p.Location), newStatus, nullify(p.SnapshotB64)).Scan(&incidentID)
		if err != nil {
			http.Error(w, "gagal buat insiden baru: "+err.Error(), 500)
			return
		}

	} else { // Insiden sudah ada
		newStatus := currentStatus
		if currentStatus == "attended" && p.EventType == "unattended" {
			newStatus = "unattended"
		} else if currentStatus == "unattended" && p.EventType == "item_taken_by_other" {
			newStatus = "taken"
		}

		query := "UPDATE cctv_incidents SET updated_at=now()"
		args := []interface{}{}
		argID := 1

		if newStatus != currentStatus {
			query += ", status=$" + strconv.Itoa(argID)
			args = append(args, newStatus)
			argID++
		}
		if p.SnapshotB64 != "" {
			query += ", last_snapshot_b64=$" + strconv.Itoa(argID)
			args = append(args, p.SnapshotB64)
			argID++
		}
		if ownerID != nil {
			query += ", owner_id=$" + strconv.Itoa(argID)
			args = append(args, *ownerID)
			argID++
		}
		if p.Location != "" {
			query += ", last_known_location=$" + strconv.Itoa(argID)
			args = append(args, p.Location)
			argID++
		}
		if p.ItemName != "" {
			query += ", item_name=$" + strconv.Itoa(argID)
			args = append(args, p.ItemName)
			argID++
		}

		query += " WHERE id=$" + strconv.Itoa(argID)
		args = append(args, incidentID)

		if len(args) > 1 {
			_, err = tx.Exec(ctx, query, args...)
			if err != nil {
				http.Error(w, "gagal update insiden: "+err.Error(), 500)
				return
			}
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
		http.Error(w, "gagal commit transaksi: "+err.Error(), 500)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "incident_id": strconv.FormatInt(incidentID, 10)})
}
func (s *Server) handleGetIncidents(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	rows, err := s.DB.Query(ctx, `
        SELECT i.id, i.group_key, i.owner_id, u.name as owner_name, i.item_name, i.status, i.created_at, i.updated_at, i.laporan_terkait_id
        FROM cctv_incidents i
        LEFT JOIN users u ON i.owner_id = u.id
        ORDER BY i.updated_at DESC
    `)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer rows.Close()

	incidents := []CCTVIncident{}
	for rows.Next() {
		var i CCTVIncident
		if err := rows.Scan(&i.ID, &i.GroupKey, &i.OwnerID, &i.OwnerName, &i.ItemName, &i.Status, &i.CreatedAt, &i.UpdatedAt, &i.LaporanTerkaitID); err != nil {
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
	err := s.DB.QueryRow(ctx, `
        SELECT i.id, i.group_key, i.owner_id, u.name as owner_name, i.item_name, i.last_known_location, i.status, i.created_at, i.updated_at, i.last_snapshot_b64, i.laporan_terkait_id
        FROM cctv_incidents i
        LEFT JOIN users u ON i.owner_id = u.id
        WHERE i.id=$1
    `, id).Scan(
		&incident.ID, &incident.GroupKey, &incident.OwnerID, &incident.OwnerName, &incident.ItemName, &incident.LastKnownLocation, &incident.Status, &incident.CreatedAt, &incident.UpdatedAt, &incident.LastSnapshotB64, &incident.LaporanTerkaitID)
	if err != nil {
		if err == pgx.ErrNoRows {
			http.Error(w, "insiden tidak ditemukan", 404)
			return
		}
		http.Error(w, "gagal mengambil detail insiden: "+err.Error(), 500)
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
	err = tx.QueryRow(ctx, `
        SELECT i.owner_id, u.name as owner_name, i.item_name, i.last_snapshot_b64, i.status
        FROM cctv_incidents i
        LEFT JOIN users u ON i.owner_id = u.id
        WHERE i.id=$1
    `, incidentID).Scan(
		&incident.OwnerID, &incident.OwnerName, &incident.ItemName, &incident.LastSnapshotB64, &incident.Status)
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
		strFallback(incident.OwnerName, "Sistem CCTV"), incident.ItemName, "Barang hilang terdeteksi oleh CCTV", "Lokasi terakhir dari CCTV", incident.LastSnapshotB64,
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

// ----- Handler User -----

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	var p LoginPayload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, "Payload JSON tidak valid: "+err.Error(), http.StatusBadRequest)
		return
	}
	if p.Email == "" || p.Password == "" {
		http.Error(w, "Email dan password wajib diisi", http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	var user User
	err := s.DB.QueryRow(ctx,
		`SELECT id, name, email, password_hash, created_at FROM users WHERE email = $1`,
		p.Email).Scan(&user.ID, &user.Name, &user.Email, &user.PasswordHash, &user.CreatedAt)

	if err != nil {
		if err == pgx.ErrNoRows {
			http.Error(w, "Kombinasi email dan password salah", http.StatusUnauthorized)
			return
		}
		http.Error(w, "Gagal mengambil data user: "+err.Error(), 500)
		return
	}

	err = bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(p.Password))
	if err != nil {
		http.Error(w, "Kombinasi email dan password salah", http.StatusUnauthorized)
		return
	}

	writeJSON(w, http.StatusOK, user)
}

func (s *Server) handleCreateUser(w http.ResponseWriter, r *http.Request) {
	var p UserPayload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if p.Name == "" {
		http.Error(w, "nama wajib diisi", http.StatusBadRequest)
		return
	}

	var newUser User
	ctx := r.Context()
	err := s.DB.QueryRow(ctx, `
        INSERT INTO users (name) VALUES ($1)
        ON CONFLICT (name) DO NOTHING
        RETURNING id, name, created_at
    `, p.Name).Scan(&newUser.ID, &newUser.Name, &newUser.CreatedAt)

	if err != nil {
		if err == pgx.ErrNoRows {
			http.Error(w, "user dengan nama tersebut sudah ada", http.StatusConflict)
			return
		}
		http.Error(w, "gagal menyimpan user: "+err.Error(), 500)
		return
	}
	writeJSON(w, http.StatusCreated, newUser)
}

func (s *Server) handleGetUsers(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	rows, err := s.DB.Query(ctx, "SELECT id, name, email, created_at FROM users ORDER BY name ASC")
	if err != nil {
		http.Error(w, "gagal mengambil data user: "+err.Error(), 500)
		return
	}
	defer rows.Close()

	users := []User{}
	for rows.Next() {
		var u User
		if err := rows.Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt); err != nil {
			http.Error(w, "gagal scan data user: "+err.Error(), 500)
			return
		}
		users = append(users, u)
	}
	writeJSON(w, http.StatusOK, users)
}
func (s *Server) handleBuatLaporan(w http.ResponseWriter, r *http.Request) {
    var p LaporanPayload
    if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
        http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
        return
    }
    if p.JenisLaporan == "" || p.NamaPelapor == "" {
        http.Error(w, "jenis_laporan dan nama_pelapor wajib diisi", http.StatusBadRequest)
        return
    }

    ctx := r.Context()
    var newLaporan Laporan
    
    err := s.DB.QueryRow(ctx, `
        INSERT INTO laporan (jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64, status)
        VALUES ($1, $2, $3, $4, $5, $6, 'draft')
        RETURNING id, jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64, status, laporan_pasangan_id, waktu_laporan, updated_at
    `, p.JenisLaporan, p.NamaPelapor, p.NamaBarang, p.Deskripsi, p.Lokasi, nullify(p.GambarBarangB64)).Scan(
        &newLaporan.ID, &newLaporan.JenisLaporan, &newLaporan.NamaPelapor, &newLaporan.NamaBarang, &newLaporan.Deskripsi, &newLaporan.Lokasi, &newLaporan.GambarBarangB64, &newLaporan.Status, &newLaporan.LaporanPasanganID, &newLaporan.WaktuLaporan, &newLaporan.UpdatedAt,
    )

    if err != nil {
        // === ADD THIS LOGGING ===
        log.Printf("FATAL: Gagal membuat laporan draft: %v", err) 
        // ========================
        http.Error(w, "gagal menyimpan laporan awal: "+err.Error(), 500)
        return
    }

    // Log the success for debugging
    log.Printf("Berhasil membuat laporan draft dengan ID: %d", newLaporan.ID)
    
    writeJSON(w, http.StatusCreated, newLaporan)
}

func (s *Server) handleGetLaporan(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
    
    // PERUBAHAN: Menambahkan "WHERE status != 'draft'" untuk menyaring hasil
	rows, err := s.DB.Query(ctx, `
        SELECT id, jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64, status, laporan_pasangan_id, waktu_laporan, updated_at
        FROM laporan
        WHERE status != 'draft'
        ORDER BY waktu_laporan DESC LIMIT 50
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

type AiAction struct {
	Action       string `json:"action"`
	JenisLaporan string `json:"jenis_laporan"`
	NamaPelapor  string `json:"nama_pelapor"`
	NamaBarang   string `json:"nama_barang"`
	Lokasi       string `json:"lokasi"`
	Deskripsi    string `json:"deskripsi"`
}
func (s *Server) createLaporanFromAI(ctx context.Context, action AiAction, imageB64 string) (*Laporan, error) {
	var newLaporan Laporan
	err := s.DB.QueryRow(ctx, `
        INSERT INTO laporan (jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id, jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64, status, laporan_pasangan_id, waktu_laporan, updated_at
    `, action.JenisLaporan, action.NamaPelapor, action.NamaBarang, action.Deskripsi, action.Lokasi, nullify(imageB64)).Scan(
		&newLaporan.ID, &newLaporan.JenisLaporan, &newLaporan.NamaPelapor, &newLaporan.NamaBarang, &newLaporan.Deskripsi, &newLaporan.Lokasi, &newLaporan.GambarBarangB64, &newLaporan.Status, &newLaporan.LaporanPasanganID, &newLaporan.WaktuLaporan, &newLaporan.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}
	return &newLaporan, nil
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
	err := s.DB.QueryRow(ctx, `
        SELECT i.id, i.group_key, i.owner_id, u.name AS owner_name, i.item_name, i.status, i.created_at, i.updated_at, i.last_snapshot_b64, i.laporan_terkait_id
        FROM cctv_incidents i
        LEFT JOIN users u ON i.owner_id = u.id
        WHERE i.id=$1
    `, id).Scan(
		&i.ID, &i.GroupKey, &i.OwnerID, &i.OwnerName, &i.ItemName, &i.Status, &i.CreatedAt, &i.UpdatedAt, &i.LastSnapshotB64, &i.LaporanTerkaitID)
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
// AiToolCall merepresentasikan format JSON yang kita harapkan dari AI.
type AiToolCall struct {
	ToolCode  string `json:"tool_code"`
	Arguments struct {
		JenisLaporan string `json:"jenis_laporan"`
		NamaBarang   string `json:"nama_barang"`
		Lokasi       string `json:"lokasi"`
	} `json:"arguments"`
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	laporanID, _ := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if laporanID == 0 {
		http.Error(w, "ID laporan tidak valid", 400)
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

	_, err := s.DB.Exec(ctx, `INSERT INTO chat_messages (laporan_id, sender, message) VALUES ($1, 'user', $2)`, laporanID, p.Message)
	if err != nil {
		http.Error(w, "gagal simpan pesan user: "+err.Error(), 500)
		return
	}

	promptParts, err := s.buildAdvancedChatPrompt(ctx, laporanID, p.Message, p.ImageB64, p.CctvIncidentID)
	if err != nil {
		http.Error(w, "gagal bangun prompt: "+err.Error(), 500)
		return
	}

	resp, err := s.Gemini.GenerateContent(ctx, promptParts...)
	if err != nil {
		http.Error(w, "gagal komunikasi dengan AI: "+err.Error(), 502)
		return
	}

	aiMessageText := extractTextFromResponse(resp)
	if aiMessageText == "" {
		aiMessageText = "Maaf, saya tidak bisa memberikan balasan saat ini."
	}

	re := regexp.MustCompile(`(?s)\{.*\}`)
	jsonString := re.FindString(aiMessageText)

	var newAiMsg ChatMessage
	var toolCall AiToolCall
	err = json.Unmarshal([]byte(jsonString), &toolCall)

	if err == nil && toolCall.ToolCode == "generateLaporan" {
		
        // PASTIKAN BARIS INI MEMANGGIL "finalizeLaporanFromAI"
		finalLaporan, err := s.finalizeLaporanFromAI(ctx, laporanID, toolCall, p.ImageB64)
		
        if err != nil {
			// Ini yang menyebabkan pesan error muncul di Flutter
			http.Error(w, "gagal memfinalisasi laporan dari AI: "+err.Error(), 500)
			return
		}
		aiMessageText = fmt.Sprintf("Baik, laporan %s untuk '%s' atas nama %s telah berhasil dibuat dan sekarang dapat dilihat di riwayat.", finalLaporan.JenisLaporan, *finalLaporan.NamaBarang, finalLaporan.NamaPelapor)
		newAiMsg.AttachmentLaporan = finalLaporan
	} else {
		aiMessageText = strings.Trim(aiMessageText, "` \n")
	}
	err = s.DB.QueryRow(ctx,
		`INSERT INTO chat_messages (laporan_id, sender, message) VALUES ($1, 'ai', $2) 
         RETURNING id, laporan_id, sender, message, created_at`,
		laporanID, aiMessageText,
	).Scan(&newAiMsg.ID, &newAiMsg.LaporanID, &newAiMsg.Sender, &newAiMsg.Message, &newAiMsg.CreatedAt)
	if err != nil {
		http.Error(w, "gagal simpan pesan AI: "+err.Error(), 500)
		return
	}
	writeJSON(w, http.StatusCreated, newAiMsg)
}

// ====================================================================================
// LOGIKA AI & PROMPT
// ====================================================================================

func (s *Server) buildAdvancedChatPrompt(ctx context.Context, laporanID int64, newUserMessage string, newUserImageB64 string, cctvIncidentID *int64) ([]genai.Part, error) {
	var parts []genai.Part
	var b strings.Builder
	var defaultPelapor string

	err := s.DB.QueryRow(ctx, "SELECT nama_pelapor FROM laporan WHERE id=$1", laporanID).Scan(&defaultPelapor)
	if err != nil {
		defaultPelapor = "Pengguna" // Fallback jika gagal
	}

	b.WriteString("Anda adalah Akai, asisten AI spesialis Lost & Found yang proaktif dan cerdas.\n")
	b.WriteString("Tugas Anda adalah memahami percakapan dengan pengguna dan memanggil fungsi `generateLaporan` ketika informasi sudah cukup.\n\n")

	b.WriteString("--- ATURAN WAJIB ---\n")
	b.WriteString("1. TUJUAN UTAMA: Identifikasi `jenis_laporan` ('kehilangan' atau 'penemuan'), `nama_barang`, dan `lokasi` dari percakapan.\n")
	b.WriteString("2. JIKA INFORMASI CUKUP: Segera panggil fungsi `generateLaporan` dengan memberikan respon HANYA dalam format JSON TUNGGAL berikut, tanpa teks tambahan:\n")
	b.WriteString("   `{\"tool_code\": \"generateLaporan\", \"arguments\": {\"jenis_laporan\": \"kehilangan/penemuan\", \"nama_barang\": \"NAMA BARANG LENGKAP\", \"lokasi\": \"LOKASI SPESIFIK\"}}`\n")
	b.WriteString("3. JIKA INFORMASI KURANG: Ajukan pertanyaan spesifik dan sopan untuk melengkapi data yang kurang. Contoh: 'Boleh tahu lokasi lebih detailnya di sebelah mana?'.\n")
	b.WriteString(fmt.Sprintf("4. Gunakan nama pelapor yang sudah ada: '%s'. Jangan tanya nama lagi.\n", defaultPelapor))
	b.WriteString("5. JANGAN PERNAH menjawab hanya dengan kalimat umum seperti 'Terima kasih atas informasinya'. Anda HARUS proaktif: memanggil fungsi (aturan #2) atau bertanya (aturan #3).\n")
	b.WriteString("6. **VALIDASI DATA:** Jangan panggil fungsi jika `nama_barang` atau `lokasi` masih umum atau tidak diketahui (contoh: 'barang', 'di sana', 'belum diketahui'). Tanyakan kembali untuk detail yang lebih spesifik.\n\n")

	b.WriteString("--- RIWAYAT PERCAKAPAN ---\n")
	chatRows, _ := s.DB.Query(ctx, "SELECT sender, message FROM chat_messages WHERE laporan_id=$1 ORDER BY created_at DESC LIMIT 6", laporanID)
	defer chatRows.Close()
	var history string
	for chatRows.Next() {
		var sender, message string
		chatRows.Scan(&sender, &message)
		history = fmt.Sprintf("%s: %s\n%s", sender, message, history)
	}
	if history == "" {
		b.WriteString("Belum ada riwayat percakapan.\n\n")
	} else {
		b.WriteString(history + "\n")
	}

	b.WriteString("--- PESAN BARU DARI PENGGUNA ---\n")
	b.WriteString(newUserMessage)

	parts = append(parts, genai.Text(b.String()))
	if newUserImageB64 != "" {
		imgBytes, err := base64.StdEncoding.DecodeString(newUserImageB64)
		if err == nil {
			parts = append(parts, genai.ImageData("jpeg", imgBytes))
		}
	}
	return parts, nil
}
func (s *Server) finalizeLaporanFromAI(ctx context.Context, laporanID int64, toolCall AiToolCall, imageB64 string) (*Laporan, error) {
	var deskripsi string
	rows, err := s.DB.Query(ctx, "SELECT message FROM chat_messages WHERE laporan_id = $1 AND sender = 'user' ORDER BY created_at DESC LIMIT 3", laporanID)
	if err == nil {
		defer rows.Close()
		var messages []string
		for rows.Next() {
			var msg string
			rows.Scan(&msg)
			messages = append([]string{msg}, messages...)
		}
		deskripsi = "Laporan dibuat berdasarkan percakapan: " + strings.Join(messages, ". ")
	} else {
		deskripsi = "Laporan dibuat oleh Akai AI."
	}

	var updatedLaporan Laporan
	err = s.DB.QueryRow(ctx, `
        UPDATE laporan
        SET
            jenis_laporan = $1,
            nama_barang = $2,
            lokasi = $3,
            deskripsi = $4,
            gambar_barang_b64 = COALESCE($5, gambar_barang_b64),
            status = 'terbuka', 
            updated_at = now()
        WHERE id = $6
        RETURNING id, jenis_laporan, nama_pelapor, nama_barang, deskripsi, lokasi, gambar_barang_b64, status, laporan_pasangan_id, waktu_laporan, updated_at
    `,
		toolCall.Arguments.JenisLaporan,
		toolCall.Arguments.NamaBarang,
		toolCall.Arguments.Lokasi,
		deskripsi,
		nullify(imageB64),
		laporanID,
	).Scan(
		&updatedLaporan.ID, &updatedLaporan.JenisLaporan, &updatedLaporan.NamaPelapor, &updatedLaporan.NamaBarang,
		&updatedLaporan.Deskripsi, &updatedLaporan.Lokasi, &updatedLaporan.GambarBarangB64, &updatedLaporan.Status,
		&updatedLaporan.LaporanPasanganID, &updatedLaporan.WaktuLaporan, &updatedLaporan.UpdatedAt,
	)

	if err != nil {
		return nil, fmt.Errorf("gagal update laporan dari draft: %w", err)
	}
	return &updatedLaporan, nil
}