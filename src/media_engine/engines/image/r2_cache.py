"""Cloudflare R2 store for the Phase 2D cache. No Hugging Face fallback."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


def _contract():
    try:
        from media_engine.engines.image import model_cache as contract
        return contract
    except ImportError:
        import importlib.util
        sibling = Path(__file__).with_name("model_cache.py")
        spec = importlib.util.spec_from_file_location("phase2d_model_cache", sibling)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_cache = _contract()
COMPLETE = _cache.COMPLETE
EMPTY = _cache.EMPTY
MANIFEST_NAME = _cache.MANIFEST_NAME
QWEN_CACHE_MIN_BYTES = _cache.QWEN_CACHE_MIN_BYTES
ModelCacheNotReady = _cache.ModelCacheNotReady
assess_cache = _cache.assess_cache
prepare_generation = _cache.prepare_generation

_REGION = "auto"
_SERVICE = "s3"
# R2 single PutObject stops at 5 GiB. Parts must be at least 5 MiB except the last.
PART_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class R2Config:
    endpoint: str
    bucket: str
    access_key_id: str
    secret_access_key: str

    def __repr__(self) -> str:
        return f"R2Config(endpoint={self.endpoint!r}, bucket={self.bucket!r})"


class R2StorageError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"R2 request failed: {status}")
        self.status = status


def config_from_env(env: os._Environ[str] | dict[str, str] | None = None) -> R2Config:
    source = os.environ if env is None else env
    if source.get("MODEL_CACHE_PROVIDER") != "r2":
        raise ModelCacheNotReady(EMPTY)
    endpoint = (source.get("R2_ENDPOINT") or "").strip().rstrip("/")
    bucket = (source.get("R2_BUCKET") or "").strip()
    access = source.get("R2_ACCESS_KEY_ID") or ""
    secret = source.get("R2_SECRET_ACCESS_KEY") or ""
    if not endpoint or not bucket or not access or not secret:
        raise ModelCacheNotReady(EMPTY)
    return R2Config(endpoint, bucket, access, secret)


class MemoryObjectStore:
    """Test double. Keys are paths relative to the cache mount."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.gets: list[str] = []
        self.lists = 0

    def get(self, key: str) -> tuple[bytes, dict[str, str]] | None:
        self.gets.append(key)
        return self.objects.get(key)

    def put(self, key: str, body: bytes, metadata: dict[str, str] | None = None) -> None:
        self.objects[key] = (body, dict(metadata or {}))

    def list_keys(self, prefix: str) -> list[str]:
        self.lists += 1
        return sorted(key for key in self.objects if key.startswith(prefix))


def sync_complete_cache(store, mount: Path, *, min_bytes: int) -> str:
    """Copy an R2 cache onto the worker. Incomplete manifests are not fetched."""
    loaded = store.get(MANIFEST_NAME)
    if loaded is None:
        raise ModelCacheNotReady(EMPTY)
    body, _meta = loaded
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ModelCacheNotReady("INVALID") from exc
    state = str(payload.get("state") or "") if isinstance(payload, dict) else ""
    if state != COMPLETE:
        raise ModelCacheNotReady(state or "INVALID")
    mount.mkdir(parents=True, exist_ok=True)
    _write_object(mount, MANIFEST_NAME, body, {})
    for key in store.list_keys("hf-cache/"):
        item = store.get(key)
        if item is None:
            raise ModelCacheNotReady("INVALID")
        data, meta = item
        _write_object(mount, key, data, meta)
    if assess_cache(mount, min_bytes=min_bytes) != COMPLETE:
        raise ModelCacheNotReady("INVALID")
    return COMPLETE


def load_from_r2(store, mount: Path, download, *, min_bytes: int) -> str:
    """Generation path. Does not upload and does not call a remote model download."""
    sync_complete_cache(store, mount, min_bytes=min_bytes)
    return prepare_generation(mount, download, min_bytes=min_bytes)


def sync_from_env(mount: Path, env: dict[str, str] | None = None) -> str:
    return sync_complete_cache(
        R2Client(config_from_env(env)),
        mount,
        min_bytes=QWEN_CACHE_MIN_BYTES,
    )


def upload_cache(store, mount: Path) -> None:
    """Explicit cache fill. The manifest is uploaded only after every other object."""
    objects: list[tuple[str, Path]] = []
    manifest: list[tuple[str, Path]] = []
    for path in sorted(mount.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        key = path.relative_to(mount).as_posix()
        if key == MANIFEST_NAME:
            manifest.append((key, path))
        else:
            objects.append((key, path))
    for key, path in objects:
        _upload_path(store, key, path)
    for key, path in manifest:
        _upload_path(store, key, path)


def _upload_path(store, key: str, path: Path) -> None:
    if path.is_symlink():
        store.put(key, b"", {"hf-symlink": os.readlink(path)})
        return
    put_file = getattr(store, "put_file", None)
    if put_file is not None:
        put_file(key, path)
        return
    store.put(key, path.read_bytes(), {})


def _write_object(mount: Path, key: str, body: bytes, metadata: dict[str, str]) -> None:
    relative = Path(key)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelCacheNotReady("INVALID")
    destination = mount / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    link = metadata.get("hf-symlink")
    if link:
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(link)
        return
    destination.write_bytes(body)


class R2Client:
    """Small path-style S3 client for Cloudflare R2."""

    def __init__(self, config: R2Config, opener=urlopen, now=None, part_size: int = PART_SIZE) -> None:
        self.config = config
        self._opener = opener
        self._now = now or (lambda: datetime.now(timezone.utc))
        if part_size < 1:
            raise ValueError("part_size must be positive")
        self.part_size = part_size

    def __repr__(self) -> str:
        return f"R2Client({self.config!r})"

    def get(self, key: str) -> tuple[bytes, dict[str, str]] | None:
        status, headers, body = self._request("GET", key, b"")
        if status == 404:
            return None
        if status != 200:
            raise R2StorageError(status)
        return body, _symlink_meta(headers)

    def put(self, key: str, body: bytes, metadata: dict[str, str] | None = None) -> None:
        extra = _meta_headers(metadata)
        status, _headers, _body = self._request("PUT", key, body, extra)
        if status not in (200, 201):
            raise R2StorageError(status)

    def put_file(self, key: str, path: Path, metadata: dict[str, str] | None = None) -> None:
        """Upload a file without reading more than one part into memory."""
        path = Path(path)
        if path.stat().st_size <= self.part_size:
            self.put(key, path.read_bytes(), metadata)
            return
        self._multipart(key, path, _meta_headers(metadata))

    def _multipart(self, key: str, path: Path, extra: dict[str, str]) -> None:
        status, _headers, body = self._request("POST", key, b"", extra, {"uploads": ""})
        if status not in (200, 201):
            raise R2StorageError(status)
        upload_id = _xml_text(body, "UploadId")
        parts: list[tuple[int, str]] = []
        try:
            number = 1
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(self.part_size)
                    if not chunk:
                        break
                    parts.append((number, self._upload_part(key, upload_id, number, chunk)))
                    number += 1
            status, _headers, _body = self._request(
                "POST", key, _complete_xml(parts), query={"uploadId": upload_id},
            )
            if status not in (200, 201):
                raise R2StorageError(status)
        except Exception:
            self._abort(key, upload_id)
            raise

    def _upload_part(self, key: str, upload_id: str, number: int, chunk: bytes) -> str:
        status, headers, _body = self._request(
            "PUT", key, chunk, query={"partNumber": str(number), "uploadId": upload_id},
        )
        if status not in (200, 201):
            raise R2StorageError(status)
        etag = _header(headers, "etag")
        if not etag:
            raise R2StorageError(status)
        return etag

    def _abort(self, key: str, upload_id: str) -> None:
        try:
            self._request("DELETE", key, b"", query={"uploadId": upload_id})
        except Exception:
            return

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        token = ""
        for _ in range(1000):
            query = {"list-type": "2", "prefix": prefix}
            if token:
                query["continuation-token"] = token
            status, _headers, body = self._request("GET", "", b"", query=query)
            if status != 200:
                raise R2StorageError(status)
            found, token = _parse_list(body)
            keys.extend(found)
            if not token:
                return keys
        raise R2StorageError(0)

    def _request(self, method: str, key: str, body: bytes, extra: dict[str, str] | None = None, query: dict[str, str] | None = None):
        parts = urlsplit(self.config.endpoint)
        host = parts.netloc
        path = "/" + self.config.bucket + (("/" + _encode_key(key)) if key else "")
        query_text = _canonical_query(query or {})
        url = self.config.endpoint + path + (("?" + query_text) if query_text else "")
        stamped = self._now()
        amzdate = stamped.strftime("%Y%m%dT%H%M%SZ")
        payload_hash = hashlib.sha256(body).hexdigest()
        headers = {
            "host": host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amzdate,
        }
        if extra:
            headers.update(extra)
        headers["authorization"] = _authorization(
            method, path, query_text, headers, payload_hash, amzdate,
            self.config.access_key_id, self.config.secret_access_key,
        )
        request = Request(url, data=body if method in {"PUT", "POST"} else None, method=method)
        for name, value in headers.items():
            if name == "host":
                continue
            request.add_header(name, value)
        try:
            with self._opener(request, timeout=60) as response:
                return response.status, dict(response.headers.items()), response.read()
        except Exception as exc:
            status = getattr(exc, "code", None)
            if status == 404:
                return 404, {}, b""
            if isinstance(status, int):
                raise R2StorageError(status) from None
            raise R2StorageError(0) from None


def _authorization(method, path, query, headers, payload_hash, amzdate, access, secret, region: str = _REGION) -> str:
    signed = sorted(headers)
    canonical_headers = "".join(f"{name}:{headers[name].strip()}\n" for name in signed)
    canonical = "\n".join([
        method,
        path,
        query,
        canonical_headers,
        ";".join(signed),
        payload_hash,
    ])
    datestamp = amzdate[:8]
    scope = f"{datestamp}/{region}/{_SERVICE}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amzdate,
        scope,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    ])
    key = ("AWS4" + secret).encode("utf-8")
    for part in (datestamp, region, _SERVICE, "aws4_request"):
        key = hmac.new(key, part.encode("utf-8"), hashlib.sha256).digest()
    signature = hmac.new(key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    return (
        "AWS4-HMAC-SHA256 "
        f"Credential={access}/{scope}, "
        f"SignedHeaders={';'.join(signed)}, "
        f"Signature={signature}"
    )


def _encode_key(key: str) -> str:
    return "/".join(quote(part, safe="") for part in key.split("/"))


def _canonical_query(query: dict[str, str]) -> str:
    pairs = sorted((quote(key, safe=""), quote(value, safe="")) for key, value in query.items())
    return "&".join(f"{key}={value}" for key, value in pairs)


def _meta_headers(metadata: dict[str, str] | None) -> dict[str, str]:
    if metadata and metadata.get("hf-symlink"):
        return {"x-amz-meta-hf-symlink": metadata["hf-symlink"]}
    return {}


def _complete_xml(parts: list[tuple[int, str]]) -> bytes:
    chunks = ["<CompleteMultipartUpload>"]
    for number, etag in parts:
        quoted = etag if etag.startswith('"') else '"' + etag + '"'
        chunks.append(f"<Part><PartNumber>{number}</PartNumber><ETag>{quoted}</ETag></Part>")
    chunks.append("</CompleteMultipartUpload>")
    return "".join(chunks).encode("utf-8")


def _header(headers: dict, name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return ""


def _xml_text(body: bytes, tag: str) -> str:
    root = ET.fromstring(body)
    for node in root.iter():
        if _local(node.tag) == tag and node.text:
            return node.text
    raise R2StorageError(0)


def _symlink_meta(headers: dict) -> dict[str, str]:
    for name, value in headers.items():
        if name.lower() == "x-amz-meta-hf-symlink":
            return {"hf-symlink": value}
    return {}


def _parse_list(body: bytes) -> tuple[list[str], str]:
    root = ET.fromstring(body)
    keys = [node.text for node in root.iter() if _local(node.tag) == "Key" and node.text]
    token = ""
    truncated = False
    for node in root.iter():
        if _local(node.tag) == "IsTruncated":
            truncated = (node.text or "").lower() == "true"
        if _local(node.tag) == "NextContinuationToken" and node.text:
            token = node.text
    return keys, token if truncated else ""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
