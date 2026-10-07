"""SSH that remains after the portable worker is ready.

Startup does not use these calls. They are the migration list toward
authenticated worker HTTP routes.
"""

PORTABLE_BOOTSTRAP_USES_SSH = False

REMAINING_SSH_AFTER_READY = (
    "confirm_cache",
    "r2_cold_stage",
    "ensure_model",
    "generate",
    "fetch_png",
)

MIGRATION_PLAN = (
    "Add authenticated worker routes for cache status, cache sync, generate, and artifact bytes.",
    "Inject runtime secrets as instance environment, not into the image.",
    "Call those routes after GET /health reports worker_ready.",
    "Keep SSH as a diagnostic channel only.",
)

# HTTP /health still decides worker_ready. The first cache, model, generate,
# or PNG call attaches the existing SSH session. That is enough for one test.
POST_READINESS_SSH = "SAFE_FOR_ONE_FINAL_TEST"
