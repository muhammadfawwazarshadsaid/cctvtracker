# Step 1: Build
FROM golang:1.22 AS builder
WORKDIR /app

# Copy mod files and download deps
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build binary
RUN go build -o app .

# Step 2: Run
FROM debian:bookworm-slim
WORKDIR /app

# Copy binary
COPY --from=builder /app/app .

# Copy .env kalau mau (opsional)
# COPY .env .env

# Expose Render's dynamic port
EXPOSE 3000

CMD ["./app"]
