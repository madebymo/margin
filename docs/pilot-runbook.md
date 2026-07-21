# Trustworthy Adaptive Lesson v2 pilot runbook

This runbook covers the anonymous, single-region curated pilot. PostgreSQL 16,
Redis 7, an immutable reviewed release bundle, and fleet telemetry are required.
SQLite and process-local control providers are development-only.

## Build and immutable inputs

Build one image from a clean commit and record its image digest. The container
runs as an unprivileged `tutor` user and rebuilds the committed Svelte bundle in
the build stage.

```bash
docker build --pull --tag adaptive-math-tutor:${GIT_SHA} .
docker image inspect adaptive-math-tutor:${GIT_SHA} --format '{{json .RepoDigests}}'
```

Mount releases read-only at `/srv/tutor/releases`. Never replace a file in that
directory under an existing release ID. Set both the active path and its exact
SHA-256 digest; retain every pinned release and policy implementation until no
unexpired resume token refers to it.

```bash
sha256sum /srv/tutor/releases/release.json
export TUTOR_V2_ACTIVE_RELEASE_BUNDLE=/srv/tutor/releases/release.json
export TUTOR_V2_ACTIVE_RELEASE_SHA256='<sha256>'
export TUTOR_V2_RELEASE_REGISTRY_DIR=/srv/tutor/releases
```

## Required configuration

Store secrets in the deployment secret manager, not an image, manifest, or log.
Use separate database, Redis, token-signing, network-HMAC, and telemetry
credentials. Retain the resume-token key for the full recovery window.

```bash
export TUTOR_PILOT_PRODUCTION=1
export DATABASE_URL='postgresql+psycopg://...'
export TUTOR_REDIS_URL='rediss://...'
export TUTOR_RESUME_TOKEN_SECRET='<at-least-32-random-bytes>'
export TUTOR_NETWORK_HMAC_SECRET='<independent-random-secret>'
export TUTOR_TRUSTED_PROXY_CIDRS='<explicit-proxy-cidrs>'
export OTEL_EXPORTER_OTLP_ENDPOINT='https://...'

export TUTOR_ENABLE_API_SESSION_V2=1
export TUTOR_ENABLE_CONTENT_ALLOCATION_V2=1
export TUTOR_ENABLE_DIAGNOSIS_V2=1
export TUTOR_ENABLE_LESSON_FLOW_V2=1
export TUTOR_ENABLE_RICH_WIDGETS_V2=1
export TUTOR_PAUSE_V2_MUTATIONS=0
export TUTOR_V2_STUDENT_ROLLOUT_PERCENT=0
```

Before starting workers, initialize both strict Redis safety documents. Keep
the mutation switch paused until the deployment checks pass:

```bash
redis-cli -u "$TUTOR_REDIS_URL" SET tutor:v2:controls:mutations \
  '{"schema_version":1,"paused":true,"revision":"deploy-1-paused"}'
redis-cli -u "$TUTOR_REDIS_URL" SET tutor:v2:controls:quarantine \
  '{"schema_version":1,"revision":"deploy-1","quarantined_digests":[]}'
```

The built-in adapters refresh those documents every five seconds and reject
state older than 20 seconds. Both values are configurable, but maximum staleness
must stay at or below 60 seconds and exceed the refresh interval. A missing,
malformed, or unavailable control pauses mutations and blocks content-bearing
session access. Optional `TUTOR_V2_*_FACTORY` settings may replace metrics,
mutation, quarantine, or request-admission adapters with deployment-owned
implementations of the same contracts.

Request admission uses Redis token buckets shared by all workers. The default
buckets are 10 create/recover/reset requests per 10 minutes, 60 actions per
minute, and 120 reads per minute. Redis keys contain only keyed HMAC identities,
never network addresses. Forwarding headers are ignored unless the immediate
peer is inside `TUTOR_TRUSTED_PROXY_CIDRS`; verify the ingress overwrites them.

Terminate TLS at the trusted ingress. Forward only from the configured proxy
CIDRs, overwrite rather than append forwarding headers, enforce the application
body limit at ingress too, and do not cache API or session-error responses.

## Deployment sequence

1. Back up PostgreSQL and verify the latest backup is restorable.
2. Publish the reviewed bundle and verify its manifest, attestation, and SHA.
3. Run `alembic -c backend/alembic.ini upgrade head` as a one-shot pre-deploy
   job. Verify `alembic -c backend/alembic.ini current` reports
   `20260721_0004 (head)`. Do not use application startup to create, stamp, or
   alter production tables. The revisions are additive and preserve the
   historical `schema_migrations` table and legacy rows when adopting a
   pre-Alembic database.
4. Start the new image with rollout `0` and mutations paused.
5. Require `/livez` to return 200 and `/readyz` to return 200 from every worker.
   Readiness includes fresh Redis controls, shared admission configuration, and
   a healthy local telemetry queue.
6. Exercise create, action, reload/resume, duplicate retry, reset, and quarantine
   recovery against a test cohort.
7. Confirm telemetry dimensions contain the exact release digest and pinned
   policy versions, without raw answers, expected answers, context, or IDs.
8. Unpause mutations, dogfood, then progress through the approved 5/25/100
   canary gates. A percentage changes admission for new sessions only.

Graceful termination must stop new requests, finish or roll back in-flight
transactions, drain the bounded telemetry queue, close verifier workers, and
dispose database and Redis pools before the platform kill timeout.

## Readiness and alerting

`/livez` is process liveness only. `/readyz` must fail closed when the migration
head, database transaction, Redis safety state, active release digest, required
production providers, or local telemetry queue is not healthy. Do not route
student traffic to an unready worker.

Alert on:

- any duplicate advance, missing evidence, corrupted replay, answer leakage,
  verifier escape, or quarantined-content response;
- eligible resume success below 99.5% or action 5xx at or above 1%;
- database pool/lock/statement timeouts, verifier saturation, stale safety
  snapshots, dropped telemetry, or retention backlog;
- a runtime release digest that differs from the deployment manifest.

Operational metrics may contain stable release, policy, capability, and item
IDs. They must never contain student/context text, submitted or expected
answers, cookies, session/learner IDs, network addresses, or provider exception
bodies.

## Retention, backup, and restore

Run retention from the platform scheduler, never from worker startup or a
request path:

```bash
DATABASE_URL="$DATABASE_URL" python -m tutor.db.retention_job \
  --batch-size 100 \
  --max-batches 10
```

Each page selects only sessions whose every bound token has expired, advances
an internal ordered cursor, then rechecks expiry under the normal checkpoint-
then-token lock order. The job retains learner identity and append-only
evidence while deleting only expired anonymous resume and episode ledgers. It
exits `0` when the current scan is complete, `3` when the hard batch limit
leaves more work for the next scheduled run, and nonzero on configuration or
database failure. Logs contain aggregate counts only, never the cursor or a
session/learner identifier. Alert on repeated `batch_limit_reached` results or
job failure.

Use encrypted PostgreSQL backups with point-in-time recovery. At least once per
release wave, restore into an isolated environment, run migrations, register
all still-referenced release and policy versions, and prove exact session replay
before declaring the backup usable. Redis safety controls must be reconstructed
from the operator audit record; Redis is not the source of session truth.

## Trust-incident rollback

For leakage, verifier escape, duplicate advancement, corrupt replay, missing
evidence, or unsupported widget generation:

1. Pause new mutations fleet-wide.
2. Set new-session rollout to `0`.
3. Quarantine the affected release digest. Verify reads, actions, recovery, and
   receipt replay return a content-free quarantine error.
4. Activate the last reviewed safe SHA without modifying the old bundle.
5. Verify `/readyz`, release dimensions, and a clean test session.
6. Keep affected pinned sessions blocked. Offer only the opaque atomic reset to
   a safe release; never replay compromised transcript content.
7. Unpause unaffected releases only after the incident owner signs off.

Do not unquarantine an artifact in place. Publish a new release ID and digest
after correction and independent review.

## Canary evidence

- Dogfood: at least 50 completed sessions with no trust-invariant breach.
- 5%: at least seven days, 200 eligible resumes, and 1,000 actions.
- 25%: at least seven additional days, 500 eligible resumes, and 5,000 actions.
- 100%: only after all reliability and zero-tolerance trust gates pass.

Save the exact image digest, release digest, migration head, policy/capability
versions, reviewer attestation, automated test report, manual accessibility
sign-off, backup-restore evidence, and operator approval for every promotion.
