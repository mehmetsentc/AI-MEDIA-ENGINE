# DAIDOI system audit

Scope: the `overnight/readiness` tree at baseline `f78bc90`, plus the offline modules added on that branch. A file existing is not treated as proof that a live provider path works. PostgreSQL runtime code and the detached watchdog remain uncommitted on `main` and are not part of this branch.

## Architecture

DAIDOI is the public name of the AI media engine. The committed process is a standard-library HTTP server, a `MediaController`, a provider, and a SQLite job database. Development uses `FakeGPUProvider` and a local preview image. Real mode constructs `VastImageProvider` and refuses to paint a preview PNG.

The intended image path is:

Studio request, API client identity, entitlement and credit reservation, durable job, worker readiness, cache check, Qwen-Image, private media object, artifact metadata, credit settlement, studio history.

That full path has been exercised with fakes and SQLite. It has not produced a real Qwen-Image PNG.

## Subsystem classification

| Subsystem | Class | Evidence |
| --- | --- | --- |
| Studio frontend | Partial | Static studio is implemented and covered by `tests/test_studio.py`. Voice, music, and video views are labeled "Coming next". There is no credit-balance control. |
| Authentication | Partial | The local API requires a bearer token. Tests reject a missing or wrong token. There is no end-user account in the static app. `nahaber` is a seeded API client. |
| User profiles | Not implemented | No profile editor, avatar, or per-person settings beyond the API client id. |
| Entitlements | Implemented and tested | SQLite entitlements, including `max_concurrent_jobs`, are covered by the platform tests. |
| Credit ledger | Implemented and tested | SQLite grants, reserves, consumes, and refunds are idempotent under test. Production balances were not modified. |
| Credit reservation | Implemented and tested | `authorize_paid` reserves before provider create. A second call with the same job does not reserve again. |
| Durable jobs | Implemented and tested | SQLite enqueue ignores a repeated idempotency key. The PostgreSQL job store is uncommitted on `main` and is absent here. |
| Worker orchestration | Implemented and tested | Controller stages are covered with `FakeGPUProvider`. Live orchestration has not completed an image. |
| Vast GPU lifecycle | Implemented but unverified | Create, status, and destroy are unit-tested with a fake transport. Earlier authorized rentals failed before an image. This mission created no instance. |
| SSH health checks | Implemented and tested | Distinct codes for unreachable SSH, unpublished port, HTTP failure, and not-ready worker are unit-tested. |
| Model cache | Partial | The manifest contract and the new offline validator are tested with fixtures. Volume `54653022` contents were not inspected. |
| Cloudflare R2 | Partial | The model and media clients are unit-tested with fakes. The reliable transfer path is behind `MEDIA_ENGINE_R2_RELIABLE_TRANSFER` and defaults off. No 57.7 GB download was run. |
| Neon PostgreSQL | Not in this branch | Persistence code is uncommitted on `main`. This branch does not write Neon. A production cutover is not claimed. |
| Image generation | Partial | Qwen command construction is tested. Rendering in this branch's new harness uses a fixture PNG. No real generation has succeeded. |
| Artifact history | Implemented and tested | Studio assets link to artifact ids in SQLite tests. |
| Studio preview | Implemented and tested | Development artifacts are served from local files. Private media reads are tested with in-memory storage. |
| Error handling | Implemented and tested | `present_job` maps failure to "Could not create the image" and keeps capacity wait out of the error tone. |
| Billing and refund | Implemented and tested | SQLite settlement and refund are idempotent. Estimated GPU cost is stored separately from a provider invoice. |
| Watchdog and cleanup | Partial | `instance_guard.py` encodes stop rules and is unit-tested. The detached process is the untracked `vast_watchdog.py` on `main` and is not in this branch. |

## Cache and transfer

The known Qwen revision is `75e0b4be04f60ec59a75f475837eced720f823b6`. The known manifest contract is 31 files and `57,704,594,653` bytes. The manifest used by the contract has no authoritative sha256 fields, so the offline validator checks hashes only when a file record carries one.

`validate_cache` returns `CACHE_COMPLETE` only when every listed file is present at the expected size, declared counts agree, and any supplied hash matches the symlink target. It does not download, and it does not write the cache.

The default R2 copy still downloads each object on a new request, counts a file only when it finishes, and deletes a partial file after an error. `r2_reliability.py` adds byte progress, range resume, bounded retry, and atomic promotion. Production sync calls it only when `MEDIA_ENGINE_R2_RELIABLE_TRANSFER=1`. That flag was not enabled for a live transfer.

The committed budget function judges the sustained rate from samples after the first positive interval. A fast first sample does not keep a slow later window alive. One slow later interval does not set the rate. The hard deadline and the termination reserve still stop the clock even during warm-up.

## Cost controls

`SpendGrant` requires an execution id, a monetary ceiling, a maximum instance count, and an expiry. A missing, stale, expired, or mismatched grant stops. The same function is what a planner, controller, and watchdog can share.

`VastImageProvider.create` on this branch does not call `SpendGrant` yet. Its committed ceiling remains `$1.00` unless `STUDIO_GPU_BUDGET_USD` tightens it. The exact `$0.50` gate exists only in the uncommitted `main` working tree. Until those are joined under a new authorization, a paid run should not be started from this branch alone.

`termination_report` leaves `confirmed_charge_usd` empty. An estimated usage row is not a Vast invoice.

## Security and abuse

High-risk surfaces:

- `PUT` instance creation on the Vast asks API. The new checker refuses `PUT`, `DELETE`, and `PATCH` before transport.
- Studio job creation. The static app has no end-user login. The API bearer is a shared client secret.
- Credit reservation against a production ledger. This branch's tests use a temporary SQLite file.
- Model-bucket downloads. A cold stage of 57.7 GB is the expensive storage path. Cold stage stays off unless `MEDIA_ENGINE_ALLOW_COLD_STAGE` is set.
- Media URLs. Signed tokens default to a 300-second TTL.

Safe change made here: `python -m media_engine.providers.machine_check` prints one JSON object, requires `VAST_READ_ONLY_DISCOVERY=1`, and returns a closed result when the credential is missing or rejected. The API key is not part of the JSON.

No firewall, database policy, DNS, or cloud permission was changed.

## Dependencies that remain

- A rentable offer for machine `47281` and volume `54653022` can be attached only after a new explicit authorization.
- The volume's files cannot be validated until an instance mounts it.
- The reliable transfer path still needs one supervised live run before it replaces the default copy.
- The PostgreSQL runtime and the detached watchdog stay on the dirty `main` tree until they are reviewed separately.
