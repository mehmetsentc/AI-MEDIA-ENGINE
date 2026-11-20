"""Host side of the remote media run: ship the code, start one job, read the receipt.

The image engine ships two single files to ``/workspace`` and the worker script
re-imports them by path when the package is not installed. That worked for one
engine; for the media capabilities it would mean copying ``invoke``, the model
registry and the audio and video helpers one by one, and keeping a parallel set
of import fallbacks in step with them. The package is 230 KB of pure Python, so
this ships all of it and sets ``PYTHONPATH`` instead. The worker then runs the
same ``invoke`` the local path runs, which is the point: one dispatch, no
second implementation to drift.

No function here rents, starts or pays for anything. They build commands and
read their output.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import tarfile
from pathlib import Path

REMOTE_ROOT = "/workspace/pysrc"
#: Where staged weights live on the worker. Defined once, here, because both
#: the staging command and the job command name it and two copies would drift.
WEIGHTS_ROOT = "/models/weights"
REMOTE_ARCHIVE = "/workspace/media_engine.tgz"
JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DONE = "JOB_DONE "
FAILED = "JOB_FAILED "


class RemoteRunError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def package_archive(source: Path, dest: Path) -> dict:
    """Pack the package reproducibly, so two builds of one tree are one archive.

    Timestamps, ownership and directory order all leak into a tar otherwise,
    which would make the digest below useless for telling "the worker has this
    code" from "the worker has some code".
    """
    source = Path(source)
    if not (source / "__init__.py").is_file():
        raise RemoteRunError("PACKAGE_NOT_FOUND")
    names = sorted(
        path for path in source.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    if not names:
        raise RemoteRunError("PACKAGE_EMPTY")
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for path in names:
            info = tarfile.TarInfo(name=str(Path(source.name) / path.relative_to(source)))
            body = path.read_bytes()
            info.size = len(body)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(body))
    packed = raw.getvalue()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(packed)
    return {
        "path": str(dest),
        "bytes": len(packed),
        "sha256": hashlib.sha256(packed).hexdigest(),
        "files": len(names),
    }


def install_command(archive: str = REMOTE_ARCHIVE, root: str = REMOTE_ROOT) -> str:
    """Unpack the shipped code and prove it imports before anything is loaded."""
    quoted_archive = shlex.quote(archive)
    quoted_root = shlex.quote(root)
    return (
        f"rm -rf {quoted_root} && mkdir -p {quoted_root} && "
        f"tar xf {quoted_archive} -C {quoted_root} && "
        f"PYTHONPATH={quoted_root} python3 -c "
        "'import media_engine.models.media_remote as m; print(\"REMOTE_READY\", len(m.EXTENSIONS))'"
    )


def job_command(
    job: dict,
    *,
    weights_root: str,
    out_dir: str,
    in_dir: str | None = None,
    root: str = REMOTE_ROOT,
    python: str = "python3",
    timeout_seconds: int = 900,
) -> str:
    """One job, written to the worker and run there.

    The job travels as a quoted argument rather than a heredoc so a prompt
    containing a quote, a newline or a backtick cannot end the command and
    start another one.
    """
    job_id = str(job.get("id") or "")
    if JOB_ID.fullmatch(job_id) is None:
        raise RemoteRunError("JOB_ID_INVALID")
    if not python or python != python.strip() or any(c in python for c in "\n\r\x00"):
        raise RemoteRunError("PYTHON_INVALID")
    body = json.dumps(job, separators=(",", ":"), sort_keys=True)
    inbox = shlex.quote(in_dir) if in_dir else ""
    parts = [
        f"mkdir -p {shlex.quote(out_dir)} /workspace/media-jobs",
        f"printf %s {shlex.quote(body)} > /workspace/media-jobs/{job_id}.json",
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1",
        (
            f"PYTHONPATH={shlex.quote(root)} "
            f"MEDIA_ENGINE_MULTIMODAL_TIMEOUT_SECONDS={int(timeout_seconds)} "
            f"{shlex.quote(python)} -u -m media_engine.models.media_remote "
            f"/workspace/media-jobs/{job_id}.json {shlex.quote(weights_root)} "
            f"{shlex.quote(out_dir)}" + (f" {inbox}" if inbox else "")
        ),
    ]
    return " && ".join(parts)


STAGE_DONE = "STAGE_DONE "
STAGE_FAILED = "STAGE_FAILED "


def stage_command(
    model_id: str,
    revision: str,
    *,
    weights_root: str = WEIGHTS_ROOT,
    root: str = REMOTE_ROOT,
    python: str = "python3",
) -> str:
    """Tell the worker to pull one model from R2 onto its own disk.

    The R2 credentials are read from the worker's environment, which the
    provider sets when it creates the instance; they are never put on a
    command line, where ``ps`` would show them to anything else on the box.
    """
    if not model_id or any(c in model_id for c in "\n\r\x00"):
        raise RemoteRunError("MODEL_INVALID")
    return (
        f"mkdir -p {shlex.quote(weights_root)} && "
        f"PYTHONPATH={shlex.quote(root)} {shlex.quote(python)} -u "
        f"-m media_engine.models.media_remote stage "
        f"{shlex.quote(model_id)} {shlex.quote(revision)} {shlex.quote(weights_root)}"
    )


def worker_stage_weights(worker, host: str, port: str, model_id: str, *, revision: str = "",
                         timeout_seconds: int = 3600) -> dict:
    """The stage_weights callback run_media_job asks for.

    Without this the production chain stopped on a machine already being paid
    for, because nothing implemented the step.
    """
    code, out, _err = worker.run(host, port, stage_command(model_id, revision), timeout_seconds)
    for line in reversed([item.strip() for item in str(out).splitlines() if item.strip()]):
        if line.startswith(STAGE_DONE):
            try:
                return json.loads(line[len(STAGE_DONE):])
            except json.JSONDecodeError as exc:
                raise RemoteRunError("STAGE_REPORT_INVALID") from exc
        if line.startswith(STAGE_FAILED):
            raise RemoteRunError(line[len(STAGE_FAILED):].strip() or "STAGE_FAILED")
    raise RemoteRunError("STAGE_NO_REPORT" if code == 0 else "STAGE_FAILED")


def parse_result(stdout: str) -> dict:
    """Read the worker's last line. Anything else is a failure, not a result."""
    for line in reversed([item.strip() for item in str(stdout).splitlines() if item.strip()]):
        if line.startswith(DONE):
            try:
                receipt = json.loads(line[len(DONE):])
            except json.JSONDecodeError as exc:
                raise RemoteRunError("RECEIPT_INVALID") from exc
            if not isinstance(receipt, dict) or not receipt.get("sha256"):
                raise RemoteRunError("RECEIPT_INVALID")
            return receipt
        if line.startswith(FAILED):
            raise RemoteRunError(line[len(FAILED):].strip() or "GENERATION_FAILED")
    raise RemoteRunError("NO_RECEIPT")


def verify_artifact(data: bytes, receipt: dict) -> None:
    """Prove the fetched bytes are the ones the worker hashed.

    scp can truncate and a half-written file can be fetched. Without this the
    failure shows up as a corrupt artifact handed to the user and charged for.
    """
    if len(data) != int(receipt.get("bytes", -1)):
        raise RemoteRunError("ARTIFACT_TRUNCATED")
    if hashlib.sha256(data).hexdigest() != str(receipt.get("sha256") or ""):
        raise RemoteRunError("ARTIFACT_CORRUPT")


def remote_artifact_path(out_dir: str, receipt: dict) -> str:
    name = str(receipt.get("file") or "")
    if not name or "/" in name or ".." in name:
        raise RemoteRunError("RECEIPT_INVALID")
    return out_dir.rstrip("/") + "/" + name
