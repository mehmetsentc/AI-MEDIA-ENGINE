"""Phase 2D image proof. Fake transport and fake pixels. No live GPU."""
from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
import urllib.request
import zlib
from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.base import ImageEngineError
from media_engine.engines.image.qwen import FIRST_PROMPT, FIRST_SEED, QwenImageEngine, validate_png
from media_engine.providers.offers import GpuOffer
from media_engine.providers.phase2d import (
    DISK_GB,
    HOURS_PER_STORAGE_MONTH,
    MAX_ESTIMATED_COST_USD,
    MAX_HOURLY_USD,
    Phase2DStop,
    accept_offer,
    attach_ssh_key,
    budget_status,
    fingerprint_for,
    format_policy,
    policy_fields,
    policy_fingerprint,
    projected_max_cost,
    quote_all_in,
    run_phase2d,
    run_ssh_runtime,
    select_one,
    ssh_attach_allowed,
    ssh_endpoint,
    storage_hourly_usd,
)
from media_engine.resources.ledger import ResourceRecord, ResourceState
from media_engine.usage import UsageStore


def _png(width: int = 1024, height: int = 1024) -> bytes:
    row = b"\x00" + bytes((12, 34, 56)) * width
    raw = zlib.compress(row * height)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


def _env(**overrides) -> dict:
    env = {
        "VAST_API_KEY": "unit-test-vast-secret",
        "LIVE_EXTERNAL_PROVIDERS": "true",
        "VAST_PROVISIONING": "1",
        "VAST_HUMAN_POLICY_APPROVAL": "approve-policy:" + policy_fingerprint(),
    }
    env.update(overrides)
    return env


def _row(**overrides) -> dict:
    row = {
        "id": 200,
        "gpu_name": "RTX 5090",
        "num_gpus": 1,
        "gpu_ram": 32607,
        "dph_base": "0.20",
        "dph_total": "0.20222222222222222",
        "storage_cost": "0.10",
        "disk_space": "500",
        "reliability2": "0.99",
        "dlperf": "80",
        "rentable": True,
    }
    row.update(overrides)
    return row


def _gpu(**kwargs) -> GpuOffer:
    values = dict(
        provider="vast", offer_id="200", gpu_model="RTX 5090", gpu_count=1,
        vram_gb=Decimal("32"), hourly_price_usd=Decimal("0.20222222222222222"),
        gpu_hourly_price_usd=Decimal("0.20"),
        storage_price_per_gb_month=Decimal("0.10"),
        disk_gb=Decimal("500"),
        reliability=Decimal("0.99"), performance=Decimal("80"), availability="rentable",
    )
    values.update(kwargs)
    return GpuOffer(**values)


class _Clock:
    def __init__(self) -> None:
        self.t = 1_700_000_000.0

    def __call__(self) -> float:
        return self.t


def _ready_row() -> dict:
    return {
        "id": 555,
        "actual_status": "running",
        "intended_status": "running",
        "ssh_host": "ssh.example",
        "ssh_port": 12345,
        "dph_total": "0.39",
    }


class _StepClock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class Phase2DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "meta.sqlite3")
        self.artifacts = str(Path(self.tmp.name) / "artifacts")
        self.clock = _Clock()
        self.calls: list[tuple[str, str]] = []
        self.generated = 0
        self._urlopen = urllib.request.urlopen
        urllib.request.urlopen = self._block

    def tearDown(self) -> None:
        urllib.request.urlopen = self._urlopen

    def _block(self, *args, **kwargs):
        raise AssertionError("network")

    def _generate(self, instance_id, offer, deadline):
        self.generated += 1
        self.assertEqual(instance_id, "555")
        self.assertEqual(offer.offer_id, "200")
        self.assertGreater(deadline, self.clock.t)
        return _png(), {"strategy": "bf16-cpu-offload", "seed": FIRST_SEED}

    def _run(self, transport, env=None, generate=None) -> tuple[int, str]:
        stdout = io.StringIO()
        code = run_phase2d(
            _env() if env is None else env,
            db_path=self.db_path,
            artifact_dir=self.artifacts,
            search_transport=transport,
            lifecycle_transport=transport,
            generate=self._generate if generate is None else generate,
            now=self.clock,
            stdout=stdout,
        )
        text = stdout.getvalue()
        self.assertNotIn("unit-test-vast-secret", text)
        return code, text

    def test_policy_fingerprint_is_deterministic_and_changes(self) -> None:
        self.assertEqual(policy_fingerprint(), policy_fingerprint())
        changed = dict(policy_fields())
        changed["max_estimated_gpu_cost_usd"] = "0.21"
        self.assertNotEqual(fingerprint_for(changed), policy_fingerprint())
        text = format_policy()
        self.assertIn("PHASE 2D POLICY", text)
        self.assertIn("max_gpu_lifetime_seconds: 1800", text)
        self.assertIn("max_estimated_gpu_cost_usd: 0.20", text)
        self.assertIn("disk_gb: 80", text)
        self.assertIn("POLICY FINGERPRINT: " + policy_fingerprint(), text)

    def test_missing_approval_does_not_touch_transport(self) -> None:
        def transport(*args):
            raise AssertionError("transport")

        code, text = self._run(transport, _env(VAST_HUMAN_POLICY_APPROVAL=""))
        self.assertEqual(code, 2)
        self.assertIn("HUMAN_POLICY_APPROVAL_REQUIRED", text)

    def test_price_and_reliability_stop_before_create(self) -> None:
        with self.assertRaises(Phase2DStop) as caught:
            accept_offer(_gpu(hourly_price_usd=Decimal("0.41"), gpu_hourly_price_usd=Decimal("0.41")))
        self.assertEqual(caught.exception.code, "PRICE_ABOVE_CEILING")
        with self.assertRaises(Phase2DStop) as caught:
            accept_offer(_gpu(reliability=Decimal("0.97")))
        self.assertEqual(caught.exception.code, "OFFER_MISMATCH")
        code, text = self._run(self._machine(offers=[_row(dph_total="0.41")]))
        self.assertEqual(code, 3)
        self.assertIn("OFFER_GONE", text)
        self.assertNotIn("PUT", [method for method, _url in self.calls])

    def test_gpu_plus_disk_above_ceiling_is_rejected_before_create(self) -> None:
        monthly = Decimal("0.20")
        gpu = Decimal("0.39")
        bundled = gpu + storage_hourly_usd(monthly, 8)
        self.assertLess(gpu, MAX_HOURLY_USD)
        self.assertLess(bundled, MAX_HOURLY_USD)
        quote = quote_all_in(_gpu(
            hourly_price_usd=bundled,
            gpu_hourly_price_usd=gpu,
            storage_price_per_gb_month=monthly,
        ))
        self.assertIsNotNone(quote)
        _gpu_price, storage, total = quote
        self.assertEqual(storage, storage_hourly_usd(monthly, DISK_GB))
        self.assertGreater(total, MAX_HOURLY_USD)
        self.assertGreater(projected_max_cost(total), MAX_ESTIMATED_COST_USD)
        row = _row(
            dph_base=str(gpu),
            dph_total=str(bundled),
            storage_cost=str(monthly),
        )
        code, text = self._run(self._machine(offers=[row]))
        self.assertEqual(code, 3)
        self.assertIn("PRICE_ABOVE_CEILING", text)
        self.assertNotIn("PUT", [method for method, _url in self.calls])

    def test_gpu_plus_disk_within_ceiling_is_eligible(self) -> None:
        accepted = accept_offer(_gpu())
        quote = quote_all_in(_gpu())
        self.assertEqual(accepted.hourly_price_usd, quote[2])
        self.assertIsNone(budget_status(accepted.hourly_price_usd))
        self.assertLess(accepted.hourly_price_usd, MAX_HOURLY_USD)

    def test_lifetime_cost_gate_is_independent_of_the_hourly_ceiling(self) -> None:
        self.assertEqual(
            budget_status(Decimal("0.30"), seconds=3600),
            "COST_ABOVE_CEILING",
        )
        self.assertIsNone(budget_status(Decimal("0.30"), seconds=1800))
        self.assertGreater(projected_max_cost(Decimal("0.41")), MAX_ESTIMATED_COST_USD)

    def test_missing_or_zero_storage_price_is_unproven(self) -> None:
        self.assertIsNone(quote_all_in(_gpu(storage_price_per_gb_month=None)))
        self.assertIsNone(quote_all_in(_gpu(storage_price_per_gb_month=Decimal("0"))))
        with self.assertRaises(Phase2DStop) as caught:
            accept_offer(_gpu(storage_price_per_gb_month=None))
        self.assertEqual(caught.exception.code, "PRICE_UNPROVEN")
        code, text = self._run(self._machine(offers=[_row(storage_cost="0")]))
        self.assertEqual(code, 3)
        self.assertIn("PRICE_UNPROVEN", text)
        self.assertNotIn("PUT", [method for method, _url in self.calls])

    def test_ranking_uses_total_hourly_not_gpu_price_alone(self) -> None:
        cheap_gpu = _gpu(
            offer_id="1",
            hourly_price_usd=Decimal("0.11"),
            gpu_hourly_price_usd=Decimal("0.10"),
            storage_price_per_gb_month=Decimal("0.50"),
            performance=Decimal("10"),
        )
        costly_gpu = _gpu(
            offer_id="2",
            hourly_price_usd=Decimal("0.21"),
            gpu_hourly_price_usd=Decimal("0.20"),
            storage_price_per_gb_month=Decimal("0.10"),
            performance=Decimal("20"),
        )
        selected = select_one([cheap_gpu, costly_gpu])
        self.assertEqual(selected.offer_id, "2")
        self.assertEqual(selected.hourly_price_usd, quote_all_in(costly_gpu)[2])

    def test_one_create_one_png_one_destroy(self) -> None:
        code, text = self._run(self._machine())
        self.assertEqual(code, 0)
        self.assertEqual(self.generated, 1)
        methods = [method for method, _url in self.calls]
        self.assertEqual(methods.count("POST"), 1)
        self.assertEqual(methods.count("PUT"), 1)
        self.assertEqual(methods.count("DELETE"), 1)
        put = [url for method, url in self.calls if method == "PUT"][0]
        self.assertTrue(put.endswith("/asks/200/"))
        path = Path(self.artifacts) / "phase2d-555.png"
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        width, height = validate_png(path.read_bytes())
        self.assertEqual((width, height), (1024, 1024))
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(meta["prompt"], FIRST_PROMPT)
        self.assertEqual(meta["seed"], FIRST_SEED)
        self.assertEqual(meta["instance_id"], "555")
        self.assertEqual(meta["owner_label"], "media-engine:phase2d")
        self.assertIn("SHA256 " + meta["sha256"], text)
        self.assertIsNone(UsageStore(self.db_path).for_job("phase2d-555").actual_cost_usd)

    def test_existing_png_is_not_overwritten(self) -> None:
        root = Path(self.artifacts)
        root.mkdir(parents=True)
        original = root / "phase2d-555.png"
        original.write_bytes(b"keep")
        code, _text = self._run(self._machine())
        self.assertEqual(code, 0)
        self.assertEqual(original.read_bytes(), b"keep")
        copies = list(root.glob("phase2d-555-*.png"))
        self.assertEqual(len(copies), 1)

    def test_generation_failure_still_destroys(self) -> None:
        def fail(*_args):
            raise Phase2DStop("GENERATION_FAILED")

        code, text = self._run(self._machine(), generate=fail)
        self.assertEqual(code, 7)
        self.assertIn("GENERATION_FAILED", text)
        self.assertIn("TERMINATED", text)
        self.assertEqual([method for method, _url in self.calls].count("DELETE"), 1)
        self.assertEqual(list(Path(self.artifacts).glob("*.png")), [])

    def test_foreign_label_is_not_destroyed(self) -> None:
        code, text = self._run(self._machine(label="media-engine:other-app"))
        self.assertEqual(code, 5)
        self.assertIn("FOREIGN", text)
        self.assertEqual(self.generated, 0)
        self.assertNotIn("DELETE", [method for method, _url in self.calls])

    def test_deadline_destroys_without_a_second_create(self) -> None:
        def transport(method, url, body, headers):
            self.calls.append((method, url))
            if method == "POST":
                return json.dumps({"offers": [_row()]}).encode("utf-8")
            if method == "PUT":
                self.clock.t += 1801
                return b'{"new_contract": 555}'
            if method == "DELETE":
                self.destroyed = True
                return b"{}"
            if getattr(self, "destroyed", False):
                return b'{"instances": []}'
            return b'{"instances": [{"id": 555, "label": "media-engine:phase2d"}]}'

        code, text = self._run(transport)
        self.assertEqual(code, 7)
        self.assertIn("COST_ABOVE_CEILING", text)
        self.assertEqual(self.generated, 0)
        self.assertEqual([method for method, _url in self.calls].count("PUT"), 1)
        self.assertIn("DELETE", [method for method, _url in self.calls])

    def test_engine_requires_prompt_seed_and_png(self) -> None:
        seen = {}

        def pipeline(**kwargs):
            seen.update(kwargs)
            return _png(kwargs["width"], kwargs["height"])

        engine = QwenImageEngine(pipeline)
        png = engine.render(FIRST_PROMPT, width=1024, height=1024, seed=FIRST_SEED)
        self.assertEqual(seen["seed"], FIRST_SEED)
        self.assertEqual(validate_png(png), (1024, 1024))
        with self.assertRaises(ImageEngineError):
            engine.render("  ")
        with self.assertRaises(ImageEngineError):
            engine.render(FIRST_PROMPT, seed=-1)
        with self.assertRaises(ImageEngineError):
            QwenImageEngine(lambda **_kwargs: b"not-a-png").render(FIRST_PROMPT)

    def test_ssh_attach_is_one_owned_instance_and_rejects_secrets(self) -> None:
        seen = {}

        def transport(method, url, body, headers):
            seen["method"] = method
            seen["url"] = url
            seen["body"] = body
            self.assertNotIn(b"unit-test-vast-secret", body)

        attach_ssh_key(transport, "555", "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIexample phase2d", "unit-test-vast-secret")
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["url"], "https://console.vast.ai/api/v0/instances/555/ssh/")
        self.assertIn(b"ssh-ed25519", seen["body"])
        self.assertFalse(ssh_attach_allowed("DELETE", seen["url"]))
        self.assertFalse(ssh_attach_allowed("POST", "https://console.vast.ai/api/v0/asks/555/"))
        with self.assertRaises(Phase2DStop):
            attach_ssh_key(transport, "offer:555", "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIexample", "k")
        with self.assertRaises(Phase2DStop):
            attach_ssh_key(transport, "555", "-----BEGIN OPENSSH PRIVATE KEY-----\n", "k")

    def test_ssh_boot_refusal_retries_then_starts_runtime_once(self) -> None:
        clock = _StepClock()
        polls = {"n": 0}
        probes = {"n": 0}
        installs = {"n": 0}

        def fetch():
            polls["n"] += 1
            if polls["n"] == 1:
                return {
                    "actual_status": "loading",
                    "intended_status": "running",
                    "ssh_host": "ssh.example",
                    "ssh_port": 22,
                }
            return _ready_row()

        def ssh(host, port, command, data, timeout):
            self.assertEqual((host, port), ("ssh.example", "12345"))
            if command == "true":
                probes["n"] += 1
                if probes["n"] == 1:
                    return 255, b"", b"connect to host ssh.example port 12345: Connection refused"
            if "phase2d_gen.py" in command and data is None:
                installs["n"] += 1
            return 0, b"", b""

        def scp(host, port, remote, local, timeout):
            if str(remote).endswith("phase2d_report.json"):
                local.write_text('{"strategy": "bf16-cpu-offload"}\n', encoding="utf-8")
            else:
                local.write_bytes(_png())

        png, report = run_ssh_runtime(
            instance_id="555",
            fetch=fetch,
            attach=lambda: None,
            ssh=ssh,
            scp=scp,
            deadline=clock.t + 60,
            now=clock,
            sleep=clock.sleep,
            diagnostic_dir=Path(self.artifacts),
            script=b"print('ok')\n",
            job=b"{}",
        )
        self.assertEqual(installs["n"], 1)
        self.assertEqual(probes["n"], 2)
        self.assertGreaterEqual(polls["n"], 2)
        self.assertGreater(clock.t, 1_000.0)
        self.assertEqual(report["strategy"], "bf16-cpu-offload")
        self.assertEqual(validate_png(png), (1024, 1024))

    def test_ssh_never_ready_skips_model_command(self) -> None:
        clock = _StepClock()
        installs = {"n": 0}

        def ssh(host, port, command, data, timeout):
            if "phase2d_gen.py" in command and data is None:
                installs["n"] += 1
            return 0, b"", b""

        with self.assertRaises(Phase2DStop) as caught:
            run_ssh_runtime(
                instance_id="555",
                fetch=lambda: {
                    "actual_status": "loading",
                    "intended_status": "running",
                    "ssh_host": "ssh.example",
                    "ssh_port": 22,
                },
                attach=lambda: None,
                ssh=ssh,
                scp=lambda *args: None,
                deadline=clock.t + 12,
                now=clock,
                sleep=clock.sleep,
                diagnostic_dir=Path(self.artifacts),
                script=b"",
                job=b"{}",
            )
        self.assertEqual(caught.exception.code, "SSH_NOT_READY")
        self.assertEqual(installs["n"], 0)
        saved = json.loads((Path(self.artifacts) / "phase2d-555.diagnostic.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["stage"], "SSH_NOT_READY")

    def test_ssh_auth_failure_keeps_sanitized_evidence(self) -> None:
        clock = _StepClock()
        secret = "unit-test-vast-secret"
        token = "a" * 64

        def ssh(host, port, command, data, timeout):
            return 255, b"stdout " + secret.encode(), b"Permission denied (publickey) Bearer " + secret.encode() + b" " + token.encode()

        with self.assertRaises(Phase2DStop) as caught:
            run_ssh_runtime(
                instance_id="555",
                fetch=lambda: _ready_row(),
                attach=lambda: None,
                ssh=ssh,
                scp=lambda *args: None,
                deadline=clock.t + 30,
                now=clock,
                sleep=clock.sleep,
                diagnostic_dir=Path(self.artifacts),
                script=b"",
                job=b"{}",
                secrets=(secret,),
            )
        self.assertEqual(caught.exception.code, "SSH_AUTH_FAILED")
        text = (Path(self.artifacts) / "phase2d-555.diagnostic.json").read_text(encoding="utf-8")
        saved = json.loads(text)
        self.assertEqual(saved["ssh_exit_code"], 255)
        self.assertIn("Permission denied", saved["stderr_tail"])
        self.assertNotIn(secret, text)
        self.assertNotIn(token, text)
        self.assertNotIn("PRIVATE KEY", text)

    def test_remote_command_failure_retains_exit_and_stderr(self) -> None:
        clock = _StepClock()

        def ssh(host, port, command, data, timeout):
            if "phase2d_gen.py" in command and data is None:
                return 1, b"", b"pip exploded"
            return 0, b"", b""

        with self.assertRaises(Phase2DStop) as caught:
            run_ssh_runtime(
                instance_id="555",
                fetch=lambda: _ready_row(),
                attach=lambda: None,
                ssh=ssh,
                scp=lambda *args: None,
                deadline=clock.t + 30,
                now=clock,
                sleep=clock.sleep,
                diagnostic_dir=Path(self.artifacts),
                script=b"x",
                job=b"{}",
            )
        self.assertEqual(caught.exception.code, "REMOTE_COMMAND_FAILED")
        saved = json.loads((Path(self.artifacts) / "phase2d-555.diagnostic.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["remote_exit_code"], 1)
        self.assertIn("pip exploded", saved["stderr_tail"])

    def test_missing_remote_report_is_explicit(self) -> None:
        clock = _StepClock()

        def ssh(host, port, command, data, timeout):
            return 0, b"", b""

        with self.assertRaises(Phase2DStop) as caught:
            run_ssh_runtime(
                instance_id="555",
                fetch=lambda: _ready_row(),
                attach=lambda: None,
                ssh=ssh,
                scp=lambda *args: None,
                deadline=clock.t + 30,
                now=clock,
                sleep=clock.sleep,
                diagnostic_dir=Path(self.artifacts),
                script=b"x",
                job=b"{}",
            )
        self.assertEqual(caught.exception.code, "REMOTE_REPORT_MISSING")
        saved = json.loads((Path(self.artifacts) / "phase2d-555.diagnostic.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["stage"], "REMOTE_REPORT_MISSING")
        self.assertEqual(saved["remote_exit_code"], 0)

    def test_runtime_failure_does_not_create_again(self) -> None:
        def fail(*_args):
            raise Phase2DStop("SSH_NOT_READY")

        code, text = self._run(self._machine(), generate=fail)
        self.assertEqual(code, 7)
        self.assertIn("SSH_NOT_READY", text)
        self.assertIn("TERMINATED", text)
        self.assertEqual([method for method, _url in self.calls].count("PUT"), 1)
        self.assertEqual([method for method, _url in self.calls].count("DELETE"), 1)

    def test_ssh_endpoint_uses_published_port_only_when_running(self) -> None:
        self.assertIsNone(ssh_endpoint({
            "actual_status": "loading",
            "intended_status": "running",
            "ssh_host": "ssh.example",
            "ssh_port": 22,
        }))
        self.assertEqual(ssh_endpoint({
            "actual_status": "running",
            "intended_status": "running",
            "public_ipaddr": "203.0.113.10",
            "ssh_host": "ssh.example",
            "ssh_port": 22,
            "ports": {"22/tcp": [{"HostPort": "34567"}]},
        }), ("203.0.113.10", "34567"))

    def test_unresolved_ledger_blocks_create(self) -> None:
        from media_engine.resources.ledger import ResourceLedger
        ResourceLedger(self.db_path).upsert(ResourceRecord(
            resource_id="9", provider="vast", owner_id="phase2d",
            state=ResourceState.READY, create_attempted=True,
            hourly_price_usd="0.20", created_at=1.0, updated_at=1.0, last_error=None,
        ))
        code, text = self._run(self._machine())
        self.assertEqual(code, 2)
        self.assertIn("UNRESOLVED_RESOURCE", text)
        self.assertEqual(self.calls, [])

    def _machine(self, *, label: str = "media-engine:phase2d", offers=None):
        state = {"destroyed": False}

        def transport(method, url, body, headers):
            self.calls.append((method, url))
            self.assertNotIn(b"unit-test-vast-secret", body)
            if method == "POST":
                self.assertIn(b'"allocated_storage":80', body)
                self.assertIn(b'"order":[["dph_total","asc"]]', body)
                return json.dumps({"offers": offers if offers is not None else [_row()]}).encode("utf-8")
            if method == "PUT":
                self.assertIn(b'"runtype":"ssh"', body)
                self.assertIn(b'"disk":80', body)
                self.assertIn(b"pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime", body)
                self.assertIn(b"media-engine:phase2d", body)
                return b'{"new_contract": 555}'
            if method == "DELETE":
                state["destroyed"] = True
                return b"{}"
            if state["destroyed"]:
                return b'{"instances": []}'
            return json.dumps({"instances": [{"id": 555, "label": label}]}).encode("utf-8")

        return transport
