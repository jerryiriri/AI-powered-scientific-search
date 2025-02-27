# Production Readiness Checklist

## Implemented in this repo

- Configuration via `.env` and strongly-typed settings (`app/config.py`)
- Structured JSON logging with request metadata (`app/logging_utils.py`)
- Request ID propagation (`x-request-id`) and per-request latency logs (`app/api.py` middleware)
- Standardized error envelopes for validation, HTTP, and unhandled exceptions (`app/api.py`)
- Liveness (`/health`) and readiness (`/ready`) endpoints
- Configurable embedding model and vector-store path
- Optional Langfuse tracing hooks for `/build-index`, `/query`, `/answer` (`app/tracing.py`)
- Containerized app runtime (`Dockerfile`)
- Docker Compose orchestration including optional Langfuse stack (`docker-compose.yml`)

## Recommended next hardening items

- Authentication and authorization for all write/query endpoints
- Rate limiting and request quotas
- Input sanitization and prompt-injection guardrails for answer synthesis
- Secrets management via cloud secret manager (not plain `.env` in production)
- Background worker for index building (avoid long-running HTTP requests)
- Retry/backoff/circuit breaker for external dependencies (ArXiv/PubMed)
- Metrics (Prometheus/OpenTelemetry): request latency, error rates, retrieval hit quality
- Data retention policy for stored PDFs and vector artifacts
- Backup/restore strategy for vector store and trace data
- CI pipeline gates: tests, lint, type checks, dependency scanning, image scan
