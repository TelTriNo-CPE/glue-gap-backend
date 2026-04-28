# Glue-Gap Backend

Two-microservice backend for geological thin-section image processing.

## Architecture

| Service | Technology | Port | Purpose |
|---|---|---|---|
| `api-gateway` | NestJS (TS) | 3030 | Entry point, file uploads, MinIO streaming |
| `image-processor` | FastAPI (Python) | 8080 | Image tiling (DZI), gap analysis, OpenCV processing |
| `minio` | S3-compatible | 9090/91 | Object storage for raw images, tiles, and results |

---

## 🛠 Prerequisites

Before you begin, ensure you have the following installed:

- **Docker & Docker Compose** (Recommended for easiest setup)
- **Node.js 20+** (For local `api-gateway` dev)
- **Python 3.11+** (For local `image-processor` dev)
- **libvips** (Required for local `image-processor` - [Installation Guide](https://www.libvips.org/install.html))

---

## 🚀 Quick Start (Docker - Recommended)

The fastest way to get everything running in a development environment:

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd glue-gap-backend
   ```

2. **Start all services:**
   ```bash
   docker compose up --build
   ```

3. **Verify:**
   - API Gateway: [http://localhost:3030/health](http://localhost:3030/health)
   - Image Processor: [http://localhost:8080/health](http://localhost:8080/health)
   - MinIO Console: [http://localhost:9091](http://localhost:9091) (User/Pass: `minioadmin` / `minioadmin`)

---

## 💻 Local Development Setup

If you prefer to run services natively for faster debugging:

### 1. Start Infrastructure (MinIO)
The backend requires MinIO. You can run just MinIO using Docker:
```bash
docker compose up minio -d
```

### 2. API Gateway Setup
```bash
cd api-gateway
pnpm install  # or npm install
npm run start:dev
```
**Default Environment Variables:**
- `PORT`: 3030
- `MINIO_ENDPOINT`: http://localhost:9090
- `MINIO_ACCESS_KEY`: minioadmin
- `MINIO_SECRET_KEY`: minioadmin
- `MINIO_BUCKET`: glue-analysis

### 3. Image Processor Setup
```bash
cd image-processor
python -m venv venv
# Windows: venv\Scripts\activate | Unix: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```
**Important:** Ensure `libvips` is installed on your system and available in your PATH.

---

## 🧪 Testing the Pipeline

1. **Upload an image:**
   ```bash
   curl -X POST http://localhost:3030/upload/image -F "file=@./sample.tiff"
   ```

2. **Check Processing Status:**
   The `image-processor` automatically polls MinIO for new images. Watch the logs to see tiling and gap analysis in progress.

3. **Get Results:**
   ```bash
   # List processed files
   curl http://localhost:3030/results
   
   # Get specific analysis (replace <stem> with filename without extension)
   curl http://localhost:3030/results/<stem>
   ```

## 🧹 Cleanup
To stop and remove all Docker containers and volumes:
```bash
docker compose down -v
```
