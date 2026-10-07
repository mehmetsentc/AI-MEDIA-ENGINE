"""PNG artifact index. Bytes stay on the filesystem."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from media_engine.db import connect, init_schema


@dataclass
class Artifact:
    artifact_id: str
    job_id: str
    engine: str
    model: str
    prompt: str
    seed: Optional[int]
    width: Optional[int]
    height: Optional[int]
    created_at: str
    byte_count: int
    sha256: str
    path: str


class ArtifactStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        init_schema(db_path)

    def insert(self, artifact: Artifact) -> Artifact:
        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, job_id, engine, model, prompt, seed, width, height,
                    created_at, byte_count, sha256, path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id, artifact.job_id, artifact.engine, artifact.model,
                    artifact.prompt, artifact.seed, artifact.width, artifact.height,
                    artifact.created_at, artifact.byte_count, artifact.sha256, artifact.path,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return artifact

    def get(self, artifact_id: str) -> Optional[Artifact]:
        conn = connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return Artifact(
            artifact_id=row["artifact_id"],
            job_id=row["job_id"],
            engine=row["engine"],
            model=row["model"],
            prompt=row["prompt"],
            seed=row["seed"],
            width=row["width"],
            height=row["height"],
            created_at=row["created_at"],
            byte_count=int(row["byte_count"]),
            sha256=row["sha256"],
            path=row["path"],
        )
