# glue-gap-backend

Two-microservice backend for geological thin-section image processing.

## Services

| Service | Port | Purpose |
|---|---|---|
| `api-gateway` | 3030 | Receives image uploads, streams to MinIO |
| `image-processor` | 8080 | Downloads images from MinIO for processing |
| `minio` | 9090 (API) / 9091 (Console) | S3-compatible object store |

**Bucket:** `glue-analysis`

## Quick Start

```bash
docker compose up --build
```

## Verification

```bash
# Health checks
curl http://localhost:3030/health
curl http://localhost:8080/health

# Upload an image (up to 1 GB)
curl -X POST http://localhost:3030/upload/image \
  -F "file=@./sample.tiff"

# MinIO web console
open http://localhost:9091   # minioadmin / minioadmin

# List uploaded files
curl http://localhost:8080/files
```
