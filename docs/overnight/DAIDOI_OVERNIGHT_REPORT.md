# DAIDOI overnight report

## A. Executive summary

The overnight work is on an isolated branch. It adds a read-only machine check, an offline cache validator, a default-off resumable transfer path, a one-execution spend grant, and a simulated image pipeline. The full unit suite on that branch passed.

Machine `47281` was rentable during the one allowed inventory read. No instance was created. The volume contents are still unverified, the reliable transfer path has not been used against the real bucket, and no real Qwen-Image generation has occurred. DAIDOI is closer to a supervised cache check, and it is not ready for an unattended paid run.

## B. Repository

| Item | Value |
| --- | --- |
| Baseline SHA | `f78bc907aa09eb73802822f3a828bb9853f3c178` |
| Branch | `overnight/readiness` |
| Worktree | `/Users/user/AI-MEDIA-ENGINE-overnight` |
| Original checkout | `/Users/user/AI-MEDIA-ENGINE` on `main`, still dirty and uncommitted |
| Final SHA | See the last commit on `overnight/readiness` after the documentation commit |

Commits on this branch:

1. `95102b88507c1dbcba9c7033211225fa6ac6cdc5` `feat(provider): classify the warm-cache machine without creating an instance`
2. `4df9e376e27642de367ab6cc81ef6fa7f1cb87df` `feat(cache): keep resumable transfers behind a default-off flag`
3. `92728cee3857a48f1d7c9804e2195f55d7f286b7` `feat(safety): require a one-execution spend grant and independent stop rules`
4. `2ab7058bfa3dd0aa5e72351ea0e9b8ae863c79a8` `test(studio): cover the image pipeline with an offline fixture`
5. Documentation commit that adds this report

The branch was not pushed, merged, or deployed.

## C. Vast

One read-only inventory check ran. It used GET for volumes and instances and POST for the offer search. The checker refuses PUT, DELETE, and PATCH.

| Item | Observed value |
| --- | --- |
| Machine | `47281` |
| Offer | `30565038` |
| GPU | RTX A6000, 49140 MiB |
| Rentable | true |
| Rented | false |
| Hourly at 40 GB disk | `$0.37481481481481485` |
| Volume | `54653022` |
| Volume state | `VOLUME_READY` on machine `47281`, no attached instances |
| Classification | `OFFER_RENTABLE`, `suitable: true` |
| GPU instances created | 0 |
| Paid GPU spending | `$0` |

This is an opportunity for a later authorized check. It was not rented. The previous `$0.50` authorization was not used. Offer id `30565038` replaces the earlier observation `30565040`. Volume contents were not listed.

## D. Model

The offline contract still expects revision `75e0b4be04f60ec59a75f475837eced720f823b6`, 31 files, and `57,704,594,653` bytes. `qwen_contract` checks that document shape. `validate_cache` checks a local mount and returns `CACHE_COMPLETE` only when sizes and any supplied hashes agree. Fixture tests cover a missing path, an invalid manifest, a revision mismatch, missing and truncated files, and a complete fixture. The remote volume was not marked complete.

Transfer work adds `ProgressBook`, range planning, bounded retry, and atomic promotion. Authentication failures are not retried. A 200 response replaces a partial file instead of being appended. The production copy uses this path only when `MEDIA_ENGINE_R2_RELIABLE_TRANSFER=1`. The default copy is unchanged. The 57.7 GB object was not downloaded.

Remaining uncertainty: whether volume `54653022` actually holds that revision, whether the host accepts range requests the way the tests assume, and whether a sustained rate inside a `$0.50` window is achievable if the cache is incomplete. A cold download remains out of scope for that budget.

## E. Credits and security

`SpendGrant` stops on a missing, stale, expired, or mismatched authorization, and on a second instance or a different machine. Tests cover those cases. `VastImageProvider.create` on this branch does not call the grant yet. The committed planner ceiling remains `$1.00`, which `STUDIO_GPU_BUDGET_USD` can only tighten. The exact `$0.50` create gate is still only in the uncommitted `main` tree.

`instance_guard` replaces a dead watchdog lock, stops two owned instances immediately, and stops one owned instance at the deadline or after a stale heartbeat. A delete timeout or a still-listed instance is reported as unknown or still listed, with no confirmed charge. The detached watchdog process remains the untracked file on `main`.

SQLite tests show a failed job refunds the reservation once, a repeated refund does not move the balance again, and a repeated settlement does not charge twice. No production credit balance was modified.

The machine-check command fails closed without `VAST_READ_ONLY_DISCOVERY=1` and does not print the API key. Signed media tokens default to 300 seconds. The static studio has no end-user login. No firewall, database policy, or cloud setting was changed.

## F. Studio

The current frontend is the static studio. Development mode was opened locally and did not rent a GPU. At 1280 pixels the layout is three columns and does not overflow. At a narrow width it becomes one column with a horizontal navigation. Export is disabled. Download stays hidden until an artifact exists. The page has no credit balance.

Voice, music, and video remain "Coming next" once a scene asset is selected. They are not operational. The image form is the only generation surface, and it still needs a balance, a single-flight Generate button, job phases, and a refund line before a paid image is shown to a user. Details are in `DAIDOI_STUDIO_FRONTEND_PLAN.md`.

## G. Tests

Command: `PYTHONPATH=src /opt/homebrew/bin/python3.14 -m unittest discover -s tests -q` in the overnight worktree.

| Result | Count |
| --- | --- |
| Passed | 284 |
| Failed | 0 |
| Skipped | 0 |

Focused modules for the new code, including the existing R2 cache tests, also passed. No test was edited to hide a failure.

No Ruff or mypy configuration is present, so those gates were not available.

`tests/test_postgres_runtime.py` is not in this worktree. It was not executed. On `main` it writes through the configured PostgreSQL DSN, so running it would violate the production-data stop condition.

No test contacted Vast. The inventory read in section C was a single manual check, not part of the suite.

## H. Next paid test readiness

**NOT READY**

Blockers:

- The volume's files have not been read.
- The reliable transfer flag is off and has no live validation.
- `SpendGrant` is not yet consulted by instance creation on this branch.
- The `$0.50` create gate lives only in uncommitted `main` changes.
- No real Qwen-Image generation has occurred.

When those gates are joined and you authorize one run:

- Recommended machine: `47281`
- Recommended volume: `54653022`
- Recommended ceiling: `$0.50`, cache verification only
- Model cache reuse: possible only if the mounted files match the manifest; 80 GB of disk does not prove that
- Real Qwen-Image generation so far: no

## I. Next action

When you are present, authorize exactly one `$0.50` attach-and-verify on machine `47281`, volume `54653022`, and the current offer `30565038`, using the `$0.50` gate from the `main` working tree together with this branch's cache validator, and stop without downloading if the cache is incomplete.
