FROM python:3.11-slim

WORKDIR /app

# Install native libraries required by OpenCV and libvips
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libvips-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Restrict libvips resource usage inside the container.
# VIPS_CONCURRENCY=1 keeps exactly one worker thread active so only one
# source strip is held in RAM at a time (prevents N-thread × strip_size spikes).
# The CACHE_MAX* variables mirror the pyvips.cache_set_max* calls in main.py and
# ensure the C library itself never retains decoded tiles after evaluation.
ENV VIPS_CONCURRENCY=1
ENV VIPS_CACHE_MAX=0
ENV VIPS_CACHE_MAX_MEM=0
ENV VIPS_CACHE_MAX_FILES=0

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
