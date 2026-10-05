# AI Media Engine

Phase 1 is a single Python process that proves a local media job lifecycle.
No GPU is started, no model is downloaded, and no external provider is called.

`LIVE_EXTERNAL_PROVIDERS` defaults to false. A credential on the machine does
not turn a provider on. The global kill switch fails closed when it is set.

## Phase 1 path

`POST /v1/jobs` stores an `IMAGE_GENERATE` job as `QUEUED` and returns immediately.
`MediaController.start()` recovers persisted jobs, reconciles the resource
ledger, then one background thread drains the queue. The thread waits on a
condition and wakes when a job is queued. `stop()` joins that thread.
Idle and lifetime shutdown still use an injected clock.

SQLite stores job metadata, attempts, and the resource ledger. Artifact bytes
stay on the filesystem.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The suite is local and does not open a network connection.
