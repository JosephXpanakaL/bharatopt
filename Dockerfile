# ==============================================================================
# Multi-stage production build for BharatOpt High-Performance Optimization Engine
# ==============================================================================

# Stage 1: Build C++20 optimization binaries
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY CMakeLists.txt .
COPY include/ ./include/
COPY src/ ./src/
COPY tests/ ./tests/

RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBHARATOPT_ENABLE_CUDA=OFF \
    -DBHARATOPT_BUILD_TESTS=ON \
    && cmake --build build -j$(nproc) \
    && ctest --test-dir build --output-on-failure

# Stage 2: Minimal, secure Python production runtime
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BHARATOPT_BIN=/opt/bharatopt/bin/bharatopt

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/bash bharatopt

WORKDIR /app

# Copy binaries and headers from builder
COPY --from=builder /workspace/build/bin/ /opt/bharatopt/bin/
RUN chmod +x /opt/bharatopt/bin/* && \
    ln -s /opt/bharatopt/bin/bharatopt /usr/local/bin/bharatopt || true

# Install Python microservice dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application layers
COPY api.py app.py ./
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY web/ ./web/

USER bharatopt

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
