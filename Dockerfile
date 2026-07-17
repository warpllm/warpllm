# syntax=docker/dockerfile:1
FROM rust:1.85-slim-bookworm AS builder
WORKDIR /app
COPY . .
# Cache mounts persist the registry and build artifacts across builds; the
# binary is copied out because cache mounts aren't part of the image layer.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release -p warpllm-server && \
    cp /app/target/release/warpllm-server /usr/local/bin/warpllm-server

# rustls bundles webpki roots, so no OpenSSL or ca-certificates needed.
# No HEALTHCHECK: distroless has no shell; orchestrators poll GET /health.
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=builder /usr/local/bin/warpllm-server /usr/local/bin/warpllm-server
# Defaults bind 0.0.0.0:8080; append flags to override, e.g.
# `docker run ... warpllm-server --port 9090`.
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/warpllm-server"]
