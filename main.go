package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
)

type Server struct {
	DB *pgxpool.Pool
}

// Payload dari Python sekarang mengirim data gambar sebagai base64 string
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
	FrameData string    `json:"frame_data,omitempty"` // Base64 encoded string
	CropData  string    `json:"crop_data,omitempty"`  // Base64 encoded string
	Message   string    `json:"message,omitempty"`
	Meta      any       `json:"meta,omitempty"`
}

// Struct untuk response JSON, gambar dikirim sebagai base64 string
type GroupRow struct {
	ID               int64     `json:"id"`
	GroupKey         string    `json:"group_key"`
	OwnerPID         int       `json:"owner_pid"`
	OwnerName        *string   `json:"owner_name,omitempty"`
	ItemName         *string   `json:"item_name,omitempty"`
	LocationLabel    *string   `json:"location,omitempty"`
	Status           string    `json:"status"`
	PreviewFrameData *string   `json:"preview_frame_data,omitempty"` // Base64 encoded string
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
}

type EventRow struct {
	ID         int64     `json:"id"`
	GroupID    int64     `json:"group_id"`
	Kind       string    `json:"kind"`
	Message    string    `json:"message"`
	OccurredAt time.Time `json:"occurred_at"`
	FrameData  *string   `json:"frame_data,omitempty"`  // Base64 encoded string
	CropData   *string   `json:"crop_data,omitempty"`   // Base64 encoded string
	Meta       any       `json:"meta,omitempty"`
}

func main() {
	_ = godotenv.Load()

	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		log.Fatal("DATABASE_URL env kosong. Contoh: postgres://user:pass@localhost:5432/ao?sslmode=disable")
	}
	ctx := context.Background()

	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()

	if err := runMigrations(ctx, pool); err != nil {
		log.Fatal(err)
	}

	s := &Server{DB: pool}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /notify", s.handleNotify)
	mux.HandleFunc("GET /groups", s.handleGroups)
	mux.HandleFunc("GET /groups/{id}", s.handleGroupDetail)
	mux.HandleFunc("POST /groups/{id}/resolve", s.handleResolveGroup)

	port := os.Getenv("PORT")
	if port == "" {
		port = "3000"
	}
	addr := ":" + port

	log.Println("listening on", addr)
	log.Fatal(http.ListenAndServe(addr, logRequest(mux)))
}

func runMigrations(ctx context.Context, db *pgxpool.Pool) error {
	// Mengubah kolom dari TEXT ke BYTEA untuk menyimpan data gambar
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
	CREATE INDEX IF NOT EXISTS idx_snapshot_events_group_time ON snapshot_events(group_id, occurred_at);
	CREATE INDEX IF NOT EXISTS idx_snapshot_groups_updated_at ON snapshot_groups(updated_at DESC);
	`
	_, err := db.Exec(ctx, sql)
	return err
}

// -------- Handlers --------

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

	// Decode base64 gambar menjadi byte slice
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

func (s *Server) handleGroupDetail(w http.ResponseWriter, r *http.Request) {
	idStr := r.PathValue("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		http.Error(w, "bad id", 400)
		return
	}
	ctx := r.Context()

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

	writeJSON(w, 200, map[string]any{
		"group":  g,
		"events": events,
	})
}

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

// -------- helpers --------
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