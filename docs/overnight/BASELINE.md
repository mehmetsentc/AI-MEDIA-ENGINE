# Overnight baseline

Recorded before the overnight branch changed any files. The original checkout was not reset, stashed, or committed.

## Repository

| Item | Value |
| --- | --- |
| Original checkout | `/Users/user/AI-MEDIA-ENGINE` |
| Original branch | `main` |
| Baseline SHA | `f78bc907aa09eb73802822f3a828bb9853f3c178` |
| Baseline subject | `fix(worker): judge cache throughput across later samples` |
| Parent | `d27555e` `fix(worker): bound image pull by the startup maximum` |
| `origin/main` | `6d7a940ffd611148d121c66d3b44c55875942ab9` |
| Main vs origin | `main` is ahead of `origin/main` by 2 commits. Those commits are not pushed. |
| Isolated worktree | `/Users/user/AI-MEDIA-ENGINE-overnight` |
| Overnight branch | `overnight/readiness` |
| Worktrees before this mission | one, the original checkout |

`f78bc907aa09eb73802822f3a828bb9853f3c178` is the latest known cache-budget commit. The overnight branch was created at that commit.

## Uncommitted work left on `main`

These paths were dirty or untracked on the original checkout and were not copied onto `overnight/readiness`:

- Modified: `pyproject.toml`, `src/media_engine/orchestrator/controller.py`, `src/media_engine/platform/service.py`, `src/media_engine/providers/phase2d.py`, `src/media_engine/providers/phase_clock.py`, `src/media_engine/providers/vast_image.py`, `src/media_engine/studio/api.py`, `src/media_engine/studio/boot.py`, `tests/test_phase2d.py`, `tests/test_r2_cache.py`, `tests/test_worker_phases.py`
- Untracked: PostgreSQL runtime modules and migrations, `src/media_engine/providers/vast_watchdog.py`, `tests/test_postgres_preparation.py`, `tests/test_postgres_runtime.py`

The dirty `vast_image.py` on `main` contains an uncommitted exact `$0.50` rental gate. That gate is not in this branch. The detached watchdog file stays untracked on `main` so a later checkout of `overnight/readiness` in the original tree cannot collide with it.

## Flags and boundaries already present at the baseline

- `LIVE_EXTERNAL_PROVIDERS` defaults off. A credential alone cannot enable a live provider.
- `VAST_READ_ONLY_DISCOVERY=1` is required for read-only offer commands.
- `MEDIA_ENGINE_ALLOW_COLD_STAGE` defaults off.
- `MEDIA_ENGINE_MAX_GPU_HOURLY_USD` defaults to `0.60`.
- `SafetyLimits.max_gpu_workers` defaults to 1.
- `global_kill_switch` defaults off. The platform ledger also has service, provider, and model kill switches.
- `MEDIA_ENGINE_ALLOW_TEST_CREDITS` defaults off and refuses a production DSN.
- Studio `real` and `production` modes do not fall back to the development image engine.
- The committed planner ceiling on this branch is `$1.00`. `STUDIO_GPU_BUDGET_USD` can only tighten that ceiling.

## Mission boundary

No GPU was rented to establish this baseline. No production database or object store was written.
