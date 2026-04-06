# ── Stage 1: Build ──────────────────────────────────────────────────────────
# Uses a full Ubuntu image with build tools. Compiles everything, runs tests.
# This stage is large (~1 GB) but is only used during CI/build.

FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc)

RUN cd build && ctest --output-on-failure

# ── Stage 2: Runtime ───────────────────────────────────────────────────────
# Copies only the compiled binary into a minimal image.
# Final image is ~10 MB instead of ~1 GB.

FROM ubuntu:22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/build/lattice /usr/local/bin/lattice

ENTRYPOINT ["lattice"]
