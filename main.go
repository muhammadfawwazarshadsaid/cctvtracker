package main

import (
	"context"
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

type NotifyPayload struct {
	GroupKey   string `json:"group_key"`             // "ownerPID101_obj501"
	OwnerPID   int    `json:"owner_pid"`             // 101
	OwnerName  string `json:"owner_name,omitempty"`  // "Annisa"
	ItemName   string `json:"item_name,omitempty"`   // "tumbler"
	Location   string `json:"location,omitempty"`    // "kantin"
	Snapshot   Snap   `json:"snapshot"`
}

type Snap struct {
	Type      string    `json:"type"`                 // attended | distance_gt_2m | person_left_frame | unattended
	TS        time.Time `json:"ts"`                   // ISO time
	FramePath string    `json:"frame_path,omitempty"`
	CropPath  string    `json:"crop_path,omitempty"`
	Message   string    `json:"message,omitempty"`    // optional override message
	Meta      any       `json:"meta,omitempty"`       // optional
}

type GroupRow struct {
	ID             int64     `json:"id"`
	GroupKey       string    `json:"group_key"`
	OwnerPID       int       `json:"owner_pid"`
	OwnerName      *string   `json:"owner_name,omitempty"`
	ItemName       *string   `json:"item_name,omitempty"`
	LocationLabel  *string   `json:"location,omitempty"`
	Status         string    `json:"status"`
	PreviewFrame   *string   `json:"preview_frame_path,omitempty"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

type EventRow struct {
	ID         int64     `json:"id"`
	GroupID    int64     `json:"group_id"`
	Kind       string    `json:"kind"`
	Message    string    `json:"message"`
	OccurredAt time.Time `json:"occurred_at"`
	FramePath  *string   `json:"frame_path,omitempty"`
	CropPath   *string   `json:"crop_path,omitempty"`
	Meta       any       `json:"meta,omitempty"`
}

func main() {
	_ = godotenv.Load()

    dsn := os.Getenv("DATABASE_URL")
    if dsn == "" {
        log.Fatal("DATABASE_URL env kosong")
    }
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
	sql := `
CREATE TABLE IF NOT EXISTS snapshot_groups (
	id                 BIGSERIAL PRIMARY KEY,
	group_key          TEXT UNIQUE NOT NULL,
	owner_pid          INT NOT NULL,
	owner_name         TEXT,
	item_name          TEXT,
	location_label     TEXT,
	status             TEXT NOT NULL DEFAULT 'ongoing',
	preview_frame_path TEXT,
	created_at         TIMESTAMP WITH TIME ZONE DEFAULT now(),
	updated_at         TIMESTAMP WITH TIME ZONE DEFAULT now()
);
CREATE TABLE IF NOT EXISTS snapshot_events (
	id            BIGSERIAL PRIMARY KEY,
	group_id      BIGINT REFERENCES snapshot_groups(id) ON DELETE CASCADE,
	kind          TEXT NOT NULL,
	message       TEXT NOT NULL,
	occurred_at   TIMESTAMP WITH TIME ZONE NOT NULL,
	frame_path    TEXT,
	crop_path     TEXT,
	meta          JSONB DEFAULT '{}'::jsonb
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

	ctx := r.Context()
	tx, err := s.DB.Begin(ctx)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	defer tx.Rollback(ctx)

	// upsert group by group_key
	var groupID int64
	upsert := `
INSERT INTO snapshot_groups (group_key, owner_pid, owner_name, item_name, location_label, status, preview_frame_path)
VALUES ($1,$2,$3,$4,$5,'ongoing',$6)
ON CONFLICT (group_key) DO UPDATE SET
owner_pid = EXCLUDED.owner_pid,
owner_name = COALESCE(EXCLUDED.owner_name, snapshot_groups.owner_name),
item_name  = COALESCE(EXCLUDED.item_name,  snapshot_groups.item_name),
location_label = COALESCE(EXCLUDED.location_label, snapshot_groups.location_label),
updated_at = now()
RETURNING id;
`
	preview := p.Snapshot.FramePath
	err = tx.QueryRow(ctx, upsert, p.GroupKey, p.OwnerPID, nullify(p.OwnerName), nullify(p.ItemName), nullify(p.Location), nullify(preview)).Scan(&groupID)
	if err != nil {
		http.Error(w, "upsert group: "+err.Error(), 500)
		return
	}

	// build default message if not provided
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

	// insert event
	insEv := `
INSERT INTO snapshot_events (group_id, kind, message, occurred_at, frame_path, crop_path, meta)
VALUES ($1,$2,$3,$4,$5,$6,$7)
RETURNING id;`
	var evID int64
	metaJSON, _ := json.Marshal(p.Snapshot.Meta)
	if err := tx.QueryRow(ctx, insEv, groupID, p.Snapshot.Type, msg, p.Snapshot.TS, nullify(p.Snapshot.FramePath), nullify(p.Snapshot.CropPath), metaJSON).Scan(&evID); err != nil {
		http.Error(w, "insert event: "+err.Error(), 500)
		return
	}

	// update group status/preview on meaningful types
	newStatus := ""
	newPreview := ""
	switch p.Snapshot.Type {
	case "unattended":
		newStatus = "unattended"
		newPreview = p.Snapshot.FramePath
	}
	if newStatus != "" || newPreview != "" {
		sb := strings.Builder{}
		sb.WriteString("UPDATE snapshot_groups SET updated_at=now()")
		args := []any{}
		if newStatus != "" {
			sb.WriteString(", status=$1")
			args = append(args, newStatus)
		}
		if newPreview != "" {
			if len(args) == 0 {
				sb.WriteString(", preview_frame_path=$1")
			} else {
				sb.WriteString(", preview_frame_path=$2")
			}
			args = append(args, newPreview)
		}
		if len(args) > 0 {
			if len(args) == 1 {
				sb.WriteString(" WHERE id=$2")
				args = append(args, groupID)
			} else { // 2 args
				sb.WriteString(" WHERE id=$3")
				args = append(args, groupID)
			}
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
SELECT id, group_key, owner_pid, owner_name, item_name, location_label, status, preview_frame_path, created_at, updated_at
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
		if err := rows.Scan(&g.ID, &g.GroupKey, &g.OwnerPID, &g.OwnerName, &g.ItemName, &g.LocationLabel, &g.Status, &g.PreviewFrame, &g.CreatedAt, &g.UpdatedAt); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		out = append(out, g)
	}
	writeJSON(w, 200, out)
}

func (s *Server) handleGroupDetail(w http.ResponseWriter, r *http.Request) {
	idStr := r.PathValue("id")
	if idStr == "" {
		http.Error(w, "missing id", 400)
		return
	}
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		http.Error(w, "bad id", 400)
		return
	}
	ctx := r.Context()

	var g GroupRow
	err = s.DB.QueryRow(ctx, `
SELECT id, group_key, owner_pid, owner_name, item_name, location_label, status, preview_frame_path, created_at, updated_at
FROM snapshot_groups WHERE id=$1`, id,
	).Scan(&g.ID, &g.GroupKey, &g.OwnerPID, &g.OwnerName, &g.ItemName, &g.LocationLabel, &g.Status, &g.PreviewFrame, &g.CreatedAt, &g.UpdatedAt)
	if err != nil {
		http.Error(w, "group not found", 404)
		return
	}

	erows, err := s.DB.Query(ctx, `
SELECT id, group_id, kind, message, occurred_at, frame_path, crop_path, meta
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
		var metaRaw []byte
		if err := erows.Scan(&e.ID, &e.GroupID, &e.Kind, &e.Message, &e.OccurredAt, &e.FramePath, &e.CropPath, &metaRaw); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
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
