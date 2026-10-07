# AI Media Engine

Phase 1 is a single Python process that proves a local media job lifecycle.
No GPU is started, no model is downloaded, and no external provider is called.

`LIVE_EXTERNAL_PROVIDERS` defaults to false. A credential on the machine does
not turn a provider on. The global kill switch fails closed when it is set.

## Phase 1 path

`POST /v1/jobs` stores a job as `QUEUED` and returns immediately.
`MediaController.start()` recovers persisted jobs, reconciles the resource
ledger, then one background thread drains the queue. The thread waits on a
condition and wakes when a job is queued. `stop()` joins that thread.
`LocalAPIServer` boots that controller and joins it on HTTP shutdown.
`IMAGE_GENERATE` still uses the fake GPU path. `TEXT_GENERATE` uses
`FakeTextEngine` and does not provision a GPU. Clients seeded today are
`film-studio` and `nahaber`. Idle and lifetime shutdown still use an injected clock.

SQLite stores job metadata, attempts, and the resource ledger. Artifact bytes
stay on the filesystem.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The suite is local and does not open a network connection.

## Read-only Vast discovery

Offer search is separate from job execution. `LIVE_EXTERNAL_PROVIDERS` stays
false, and this command does not create, start, stop, or destroy a GPU.
It runs only when both values are present. The key below is a placeholder:

```bash
VAST_API_KEY=your-key-here VAST_READ_ONLY_DISCOVERY=1 \
  PYTHONPATH=src python3 -m media_engine.providers.vast_discover
```

Optional filters, all unset by default: `VAST_MIN_VRAM_GB`,
`VAST_MAX_HOURLY_PRICE_USD`, `VAST_MIN_RELIABILITY`, `VAST_GPU_COUNT`,
`VAST_GPU_MODEL`, `VAST_LIMIT` (default 64).
The command prints normalized offers and does not print the API key.
`dlperf` is only a marketplace heuristic, not measured inference speed.

## Read-only GPU preflight

This command searches offers, ranks them, and stops at human approval.
It does not create or destroy a GPU. There is no default approval.

```bash
VAST_API_KEY=your-key-here VAST_READ_ONLY_DISCOVERY=1 \
  VAST_MIN_VRAM_GB=24 VAST_GPU_COUNT=1 \
  VAST_MAX_HOURLY_PRICE_USD=0.60 VAST_MIN_RELIABILITY=0.90 \
  VAST_MAX_JOB_SECONDS=600 \
  PYTHONPATH=src python3 -m media_engine.providers.vast_preflight
```

`VAST_GPU_MODEL` is an optional investigation filter, for example a marketplace
name with spaces or underscores. It is not a built-in model list.

## Phase 2C policy approval

Print the one-run policy and its fingerprint. This command does not contact
Vast and does not create a GPU:

```bash
PYTHONPATH=src python3 -m media_engine.providers.phase2c_policy
```

The live command rents one RTX 5090 only when the approval token is exactly
`approve-policy:` plus that fingerprint. It discovers one eligible offer at
the last moment and creates it once. There is no default approval. The
instance image is `ubuntu:22.04` with 8 GB disk, which is the rental shell,
not a model.

```bash
VAST_API_KEY=your-key-here \
LIVE_EXTERNAL_PROVIDERS=true \
VAST_PROVISIONING=1 \
VAST_HUMAN_POLICY_APPROVAL='approve-policy:<fingerprint>' \
PYTHONPATH=src python3 -m media_engine.providers.phase2c
```

## Phase 2E image API

NaHaber talks only to this process. The bearer is `AI_MEDIA_ENGINE_API_KEY`.
If that variable is missing, every route except `GET /health` fails closed.
The server does not print the key.

```bash
curl -s http://127.0.0.1:8080/health

curl -s -X POST http://127.0.0.1:8080/v1/images/generations \
  -H "Authorization: Bearer $AI_MEDIA_ENGINE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A cinematic luxury Mediterranean hotel terrace at sunset","width":1024,"height":1024}'

curl -s http://127.0.0.1:8080/v1/jobs/JOB_ID \
  -H "Authorization: Bearer $AI_MEDIA_ENGINE_API_KEY"

curl -s -o image.png http://127.0.0.1:8080/v1/artifacts/ARTIFACT_ID \
  -H "Authorization: Bearer $AI_MEDIA_ENGINE_API_KEY"
```

`POST /v1/images/generations` returns `job_id` and `status: queued` immediately.
Poll `GET /v1/jobs/{job_id}` until `completed` or `failed`. A completed job
includes `artifact_id`, dimensions, `seed`, and `sha256`. Fetch the PNG from
`GET /v1/artifacts/{artifact_id}`.

The image worker keeps one Vast GPU. When machine 47281 is rentable it
attaches the warm cache volume 54653022 at `/models`. Otherwise it rents
another compatible GPU and copies the completed cache from R2 onto local
disk before loading the model. An idle worker is destroyed after 300
seconds. The hard lifetime defaults to 1800 seconds.
`MEDIA_ENGINE_MAX_GPU_HOURLY_USD` defaults to 0.60. A higher hourly rate is
used only when the projected copy, load, and one image stay within the
operation budget.
`negative_prompt` is rejected because the Qwen path does not apply it.
