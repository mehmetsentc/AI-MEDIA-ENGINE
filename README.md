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
`VAST_MAX_HOURLY_PRICE_USD`, `VAST_MIN_RELIABILITY`, `VAST_GPU_COUNT`.
The command prints normalized offers and does not print the API key.
