# Security Notes

Secure-design decisions and known limitations, for transparency. This is a
proof-of-concept service; do not describe it as production-secure.

## Implemented

- **Input validation:** Pydantic rejects empty/whitespace-only input, non-strings and comments over 5,000 characters (HTTP 422). Comments that contain no analysable text after cleaning (e.g. only punctuation/emoji) are also rejected with 422 rather than labelled.
- **CORS:** allowed origins come from the `CORS_ORIGINS` environment variable (default: `http://localhost:3000`, `http://127.0.0.1:3000`). `*` is ignored. Credentials are disabled; only GET/POST and the `Content-Type` header are allowed. Set `CORS_ORIGINS` to your deployed frontend URL in production.
- **Error handling:** unexpected exceptions return a generic HTTP 500 message. Details go to the server log; comment text is never logged (only its length).
- **Rate limiting:** per-IP sliding window on `POST /moderate-comment` (`RATE_LIMIT` requests per `RATE_LIMIT_WINDOW` seconds, default 30/60). It is **process-local**: multiple instances each keep their own counters, and behind a reverse proxy the client IP seen by the app may be the proxy's. For multi-instance deployments use a shared store (e.g. Redis) or enforce limits at the gateway.
- **Readiness:** `/health` = process alive; `/ready` = models loaded (503 otherwise).
- **Output checks:** predicted labels must belong to the task's allowed set and probabilities must lie in [0, 1], otherwise the request fails safely.
- **Metrics scope:** `/metrics` counters are process-local and reset on restart; `/logs/recent` (database) persists.
- **Privacy:** raw comment text is not persisted unless `STORE_RAW_COMMENT=true`.

## Secrets & configuration

- No credentials or API keys are committed. `docker-compose.yml` contains throwaway local-development database credentials only; use real secrets via environment variables / a secret manager for any deployment.
- `.gitignore` excludes `.env*`, local databases, virtual environments, caches and raw datasets.
- If a secret is ever committed, rotate it - deleting it from the latest commit does not remove it from Git history.

## Model artifact

`backend/models/moderation_model.joblib` is a pickle-based file; loading an untrusted one can execute arbitrary code. Only load files you trained yourself.

## Known limitations

- **No authentication or authorisation** on any endpoint (including `/logs/recent` and `/metrics`). Add API-key or JWT auth before exposing the service beyond a demo.
- Rate limiting is not distributed (see above).
- No dependency pinning, vulnerability scanning, load testing or penetration testing has been done. Enable GitHub secret scanning, push protection and Dependabot on the repository.
- Model quality limitations (domain mismatch, label noise, unmeasured bias) are documented in `docs/limitations.md`.
