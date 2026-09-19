FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBHARATOPT_ENABLE_CUDA=OFF && cmake --build build -j2 && ctest --test-dir build --output-on-failure
FROM python:3.12-slim
WORKDIR /app
COPY --from=build /src/build/bharatopt_cli /app/build/bharatopt_cli
COPY api.py requirements.txt /app/
COPY web /app/web
RUN pip install --no-cache-dir -r requirements.txt
ENV BHARATOPT_BIN=/app/build/bharatopt_cli
EXPOSE 8000
CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8000"]
