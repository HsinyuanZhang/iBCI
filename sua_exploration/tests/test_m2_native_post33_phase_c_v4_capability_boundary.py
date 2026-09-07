"""CPU-only adversarial tests for the Phase-C production capability boundary."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sua_exploration.mc_maze import m2_native_post33_authorization_v4 as authorization
from sua_exploration.mc_maze import m2_native_post33_phase_c_v4 as contract
from sua_exploration.mc_maze import m2_native_post33_openers_v4 as openers
from sua_exploration.tests.m2_native_post33_phase_c_v4_test_support import (
    consume_opening_authorization_nonce_with_test_anchor,
    enable_synthetic_fixture_root,
)


ROOT = Path(__file__).resolve().parents[2]
SPINT_ROOT = ROOT / "SPINT-main"
STREAMING_ROOT = ROOT / "streaming_calibration_exp"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


class CapabilityBoundaryTest(unittest.TestCase):
    def test_production_apis_do_not_accept_anchor_or_validated_mapping(self) -> None:
        verifier = inspect.signature(authorization.verify_signed_authorization).parameters
        self.assertNotIn("public_key_path", verifier)
        self.assertNotIn("expected_public_key_sha256", verifier)
        stage_a = inspect.signature(openers.open_stage_a).parameters
        self.assertNotIn("validated_authorization", stage_a)
        for required in (
            "authorization_path",
            "signature_path",
            "phase_c_program_receipt_path",
            "portable_manifest_path",
            "shard_manifest_path",
            "cost_supplement_path",
        ):
            self.assertIn(required, stage_a)

    def test_synthetic_relaxations_are_quarantined_and_cannot_write_workspace_root(self) -> None:
        """Private fixture branches reject a production-like root before any write.

        The direct private calls are intentional adversarial coverage.  A
        leading underscore is not an authority boundary, so the relaxed paths
        must prove their disposable test-root capability themselves.
        """
        production_apis = (
            contract.finalize_cell_score_sealed,
            contract.verify_cell_exact,
            contract.finalize_stage_a_score_sealed,
            contract.verify_stage_a_exact,
            contract.finalize_matrix_score_sealed,
            contract.verify_matrix_exact,
            openers.open_stage_a,
            openers.open_full_matrix,
            openers.validate_stage_a_decision,
        )
        for api in production_apis:
            with self.subTest(api=api.__name__):
                parameters = inspect.signature(api).parameters
                self.assertNotIn("allow_synthetic", parameters)
                self.assertNotIn("allow_synthetic_decoder_evidence", parameters)

        scripts = (
            ROOT / "sua_exploration/scripts/finalize_m2_native_post33_phase_c_v4_cell.py",
            ROOT / "sua_exploration/scripts/finalize_m2_native_post33_phase_c_v4_matrix.py",
            ROOT / "sua_exploration/scripts/verify_m2_native_post33_phase_c_v4_artifacts.py",
            ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_matrix.py",
            ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_cell_pipeline.py",
        )
        for script in scripts:
            with self.subTest(script=script.name):
                source = script.read_text(encoding="utf-8")
                self.assertNotIn("--allow-synthetic", source)
                self.assertNotIn("allow_synthetic=", source)

        lifecycle_path = STREAMING_ROOT / "src/utils/decoder_lifecycle_phase_c_v4.py"
        spec = importlib.util.spec_from_file_location("phase_c_lifecycle_boundary", lifecycle_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        lifecycle = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(lifecycle)
        for api in (lifecycle.write_stage_exclusive, lifecycle.finalize_lifecycle_evidence):
            with self.subTest(lifecycle_api=api.__name__):
                parameters = inspect.signature(api).parameters
                self.assertNotIn("synthetic_proof", parameters)
                self.assertNotIn("allow_synthetic", parameters)

        production_like = ROOT / ".m2-phase-c-v4-private-synthetic-denied"
        self.assertFalse(production_like.exists())
        key = contract.CellKey(contract.PROTOCOL_ID, "t4", 0, 42)

        # The tests-only helper itself validates before attempting the O_EXCL
        # sentinel write; it cannot create a marker in the workspace.
        with self.assertRaisesRegex(PermissionError, "inside the workspace"):
            enable_synthetic_fixture_root(production_like)
        self.assertFalse(production_like.exists())

        with self.assertRaisesRegex(PermissionError, "inside the workspace"):
            contract._finalize_stage_a_score_sealed(
                production_like, allow_synthetic=True
            )
        self.assertFalse(production_like.exists())

        with self.assertRaisesRegex(PermissionError, "inside the workspace"):
            openers._open_stage_a_after_consumed_capability(
                production_like,
                authorization_path=production_like / "authorization.json",
                signature_path=production_like / "authorization.sig",
                opening_claim=production_like / "claim.json",
                allow_synthetic=True,
            )
        self.assertFalse(production_like.exists())

        with self.assertRaisesRegex(PermissionError, "inside the workspace"):
            lifecycle._write_stage_exclusive(
                production_like / "lifecycle-stages",
                stage="pretrain",
                decoder=object(),
                optimizer=None,
                cell_identity=key.identity(),
                synthetic_proof=True,
                synthetic_fixture_root=production_like,
            )
        self.assertFalse(production_like.exists())

        with self.assertRaisesRegex(PermissionError, "inside the workspace"):
            lifecycle._finalize_lifecycle_evidence(
                production_like / "lifecycle-stages",
                production_like / "decoder_lifecycle_evidence.json",
                cell_identity=key.identity(),
                allow_synthetic=True,
                synthetic_fixture_root=production_like,
            )
        self.assertFalse(production_like.exists())

    def test_stage_a_rejects_preexisting_stage_b_and_foreign_cell_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase = contract.phase_root(root)
            for arm in contract.ARMS:
                for fold in contract.FOLDS:
                    (phase / "cells" / f"arm-{arm}" / f"fold-{fold}" / "seed-42").mkdir(
                        parents=True
                    )
            contract.validate_stage_a_cell_directory_set(root)

            extra = phase / "cells" / "arm-spint" / "fold-0" / "seed-43"
            extra.mkdir()
            with self.assertRaisesRegex(ValueError, "seed directory exact set"):
                contract.validate_stage_a_cell_directory_set(root)
            shutil.rmtree(extra)

            foreign = phase / "cells" / "foreign-arm"
            foreign.mkdir()
            with self.assertRaisesRegex(ValueError, "arm directory exact set"):
                contract.validate_stage_a_cell_directory_set(root)

    def test_opening_capability_is_fixed_anchor_scoped_and_single_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cells"
            root.mkdir()
            private = Ed25519PrivateKey.generate()
            public = (Path(temporary) / "root-public.pem").resolve()
            public.write_bytes(
                private.public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            fixture = self._opening_fixture(
                Path(temporary), root=root, private=private, public=public, suffix="good",
                stage="stage_a_opening", scope="opening", seeds=[42],
            )
            _, claim = consume_opening_authorization_nonce_with_test_anchor(
                root=root,
                opening_stage="stage_a_opening",
                authorization_path=fixture["authorization"],
                signature_path=fixture["signature"],
                phase_c_program_receipt_path=fixture["program"],
                portable_manifest_path=fixture["portable"],
                shard_manifest_path=fixture["shard"],
                cost_supplement_path=fixture["supplement"],
                public_key_path=public,
                expected_public_key_sha256=contract.sha256_file(public),
            )
            self.assertTrue(claim.is_file())
            with self.assertRaises(FileExistsError):
                consume_opening_authorization_nonce_with_test_anchor(
                    root=root,
                    opening_stage="stage_a_opening",
                    authorization_path=fixture["authorization"],
                    signature_path=fixture["signature"],
                    phase_c_program_receipt_path=fixture["program"],
                    portable_manifest_path=fixture["portable"],
                    shard_manifest_path=fixture["shard"],
                    cost_supplement_path=fixture["supplement"],
                    public_key_path=public,
                    expected_public_key_sha256=contract.sha256_file(public),
                )
            wrong_scope = self._opening_fixture(
                Path(temporary), root=root, private=private, public=public, suffix="wrong-scope",
                stage="stage_a", scope="cell_execution", seeds=[42],
            )
            with self.assertRaisesRegex(PermissionError, "opening capability"):
                consume_opening_authorization_nonce_with_test_anchor(
                    root=root,
                    opening_stage="stage_a_opening",
                    authorization_path=wrong_scope["authorization"],
                    signature_path=wrong_scope["signature"],
                    phase_c_program_receipt_path=wrong_scope["program"],
                    portable_manifest_path=wrong_scope["portable"],
                    shard_manifest_path=wrong_scope["shard"],
                    cost_supplement_path=wrong_scope["supplement"],
                    public_key_path=public,
                    expected_public_key_sha256=contract.sha256_file(public),
                )

    def test_gpu_mapping_is_single_physical_device_and_is_recorded(self) -> None:
        shard = {"gpu_id": "17"}
        observed = authorization.validate_observed_gpu(
            shard, observed_visible_devices="17"
        )
        self.assertEqual(
            observed,
            {
                "cuda_visible_devices": "17",
                "logical_cuda_device_index": 0,
                "physical_gpu_id": "17",
            },
        )
        for invalid in ("", "16", "17,16", "17,", ",17", " 17", "17 "):
            with self.subTest(cuda_visible_devices=invalid):
                with self.assertRaisesRegex(PermissionError, "signed GPU"):
                    authorization.validate_observed_gpu(
                        shard, observed_visible_devices=invalid
                    )

        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = temporary_path / "cells"
            key = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42)
            files = {
                name: (temporary_path / name).resolve()
                for name in ("authorization.json", "authorization.sig", "shard.json", "claim.json")
            }
            for path in files.values():
                path.write_text(path.name, encoding="utf-8")
            evidence = authorization._execution_capability_evidence(
                root=root,
                key=key,
                authorization={
                    "authorization_id": "fixture-gpu-evidence",
                    "single_use_nonce": "a" * 64,
                    "stage": "stage_a",
                    "capability_scope": "cell_execution",
                    "host_id": "fixture-host",
                },
                authorization_path=files["authorization.json"],
                signature_path=files["authorization.sig"],
                shard_manifest_path=files["shard.json"],
                claim_path=files["claim.json"],
                gpu_observation=observed,
            )
            self.assertEqual(evidence["observed_gpu"], observed)
            self.assertEqual(evidence["capability_scope"], "cell_execution")
            self.assertEqual(evidence["observed_gpu"]["physical_gpu_id"], "17")

    def test_direct_entrypoint_guards_precede_cuda_data_and_trainer_work(self) -> None:
        """AST ordering plus direct subprocess negatives for all protected paths."""
        protected = (
            (
                ROOT / "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v4.py",
                "require_cell_execution_capability(\n        root=",
                "subprocess.run(worker",
            ),
            (
                SPINT_ROOT / "src/evaluate_post33_phase_c_v4.py",
                "require_cell_execution_capability(\n        root=",
                "datamodule = hydra.utils.instantiate",
            ),
            (
                STREAMING_ROOT / "src/evaluate_post33_phase_c_v4.py",
                "require_cell_execution_capability(\n        root=",
                "datamodule = hydra.utils.instantiate",
            ),
            (
                SPINT_ROOT / "src/train_post33_phase_c_v4.py",
                "require_cell_execution_capability_from_environment(root=root, key=key)",
                "legacy.train(cfg)",
            ),
            (
                STREAMING_ROOT / "src/train_post33_phase_c_v4.py",
                "require_cell_execution_capability_from_environment(root=root, key=key)",
                "legacy.train(cfg)",
            ),
        )
        for path, guard, protected_action in protected:
            source = path.read_text(encoding="utf-8")
            with self.subTest(entrypoint=path.name):
                self.assertLess(source.index(guard), source.index(protected_action))

        # T4 accepts a completion receipt that the model constructor will
        # later dereference.  The receipt must be tied to the paired SPINT
        # cell under this exact root before any data/model/Trainer work (or
        # before the pipeline can launch the trainer subprocess).  Match the
        # indented call rather than the import so this remains an ordering
        # assertion about executable code.
        paired_teacher_guards = (
            (
                STREAMING_ROOT / "src/train_post33_phase_c_v4.py",
                "    require_same_root_paired_spint_teacher(\n",
                "    metric_dict, _ = legacy.train(cfg)",
            ),
            (
                STREAMING_ROOT / "src/evaluate_post33_phase_c_v4.py",
                "    require_same_root_paired_spint_teacher(\n",
                "    datamodule = hydra.utils.instantiate(cfg.data)",
            ),
            (
                ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_cell_pipeline.py",
                "        require_same_root_paired_spint_teacher(args.cell_root, key, paired)",
                "    command, cwd = _training_command(args, key, paths)",
            ),
        )
        for path, paired_guard, protected_action in paired_teacher_guards:
            source = path.read_text(encoding="utf-8")
            with self.subTest(paired_teacher_guard=path.name):
                self.assertLess(source.index(paired_guard), source.index(protected_action))

        matrix_source = (
            ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_matrix.py"
        ).read_text(encoding="utf-8")
        self.assertLess(
            matrix_source.index("validate_observed_gpu({\"gpu_id\": gpu_id}"),
            matrix_source.index("subprocess.Popen("),
        )

        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            missing = temporary_path / "missing-authorization.json"
            root = temporary_path / "self-created-root"
            common = [
                "--cell-root", str(root), "--fold", "0", "--seed", "42",
                "--owner-token", "self-created-owner",
                "--authorization", str(missing),
                "--authorization-signature", str(temporary_path / "missing.sig"),
                "--program-receipt", str(temporary_path / "missing-program.json"),
                "--portable-manifest", str(temporary_path / "missing-portable.json"),
                "--shard-manifest", str(temporary_path / "missing-shard.json"),
                "--cost-supplement", str(temporary_path / "missing-supplement.json"),
            ]
            invocations = (
                (
                    "top-evaluator",
                    ROOT,
                    [
                        sys.executable,
                        str(ROOT / "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v4.py"),
                        "--protocol", contract.PROTOCOL_ID, "--phase", contract.PHASE_ID,
                        "--arm", "spint", *common,
                        "--opaque-payload-out", str(temporary_path / "top-payload.json"),
                        "--score-commitment-out", str(temporary_path / "top-commitment.json"),
                        "--deployment-cost-evidence-out", str(temporary_path / "top-cost.json"),
                    ],
                    (temporary_path / "top-payload.json", temporary_path / "top-commitment.json"),
                ),
                (
                    "spint-worker",
                    SPINT_ROOT,
                    [
                        sys.executable, str(SPINT_ROOT / "src/evaluate_post33_phase_c_v4.py"),
                        *common,
                        "--opaque-payload-out", str(temporary_path / "spint-payload.json"),
                        "--deployment-cost-evidence-out", str(temporary_path / "spint-cost.json"),
                    ],
                    (temporary_path / "spint-payload.json", temporary_path / "spint-cost.json"),
                ),
                (
                    "t4-worker",
                    STREAMING_ROOT,
                    [
                        sys.executable, str(STREAMING_ROOT / "src/evaluate_post33_phase_c_v4.py"),
                        *common,
                        "--opaque-payload-out", str(temporary_path / "t4-payload.json"),
                        "--decoder-evidence-out", str(temporary_path / "t4-decoder.json"),
                        "--outer-runtime-evidence-out", str(temporary_path / "t4-runtime.json"),
                        "--deployment-cost-evidence-out", str(temporary_path / "t4-cost.json"),
                    ],
                    (
                        temporary_path / "t4-payload.json", temporary_path / "t4-decoder.json",
                        temporary_path / "t4-runtime.json", temporary_path / "t4-cost.json",
                    ),
                ),
            )
            for label, cwd, command, outputs in invocations:
                with self.subTest(entrypoint=label):
                    environment = self._capability_free_environment()
                    # Direct script execution puts the script's ``src/``
                    # directory first, not its project root.  Add the
                    # entrypoint workspace explicitly so the process reaches
                    # its real argument parser and capability guard rather
                    # than failing on an incidental import-path detail.
                    environment["PYTHONPATH"] = (
                        f"{cwd}{os.pathsep}{environment['PYTHONPATH']}"
                    )
                    completed = subprocess.run(
                        command,
                        cwd=cwd,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=45,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("missing-authorization.json", completed.stderr)
                    self.assertTrue(all(not output.exists() for output in outputs))

    def test_direct_training_wrapper_guard_rejects_self_created_owner(self) -> None:
        """Invoke each wrapper's first entry gate in a clean child process."""
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            for arm, workspace, wrapper in (
                ("spint", SPINT_ROOT, SPINT_ROOT / "src/train_post33_phase_c_v4.py"),
                ("t4", STREAMING_ROOT, STREAMING_ROOT / "src/train_post33_phase_c_v4.py"),
            ):
                with self.subTest(arm=arm):
                    root = temporary_path / arm / "cell-root"
                    key = contract.CellKey(contract.PROTOCOL_ID, arm, 0, 42)
                    owner_token = f"self-created-{arm}-owner"
                    paths = contract.claim_cell(root, key, owner_token=owner_token)
                    resolved = paths["resolved_config"]
                    config: dict[str, object] = {
                        "cell_paths": {
                            "owner": str(paths["owner"]),
                            "resolved_config": str(resolved),
                        },
                        "data": {"loso_fold": 0},
                        "seed": 42,
                        "cell_owner_token": owner_token,
                    }
                    if arm == "t4":
                        config["paths"] = {
                            "artifact_dir": str(paths["secondary_artifact_root"]),
                        }
                    child = "\n".join(
                        (
                            "import importlib.util, json",
                            "from omegaconf import OmegaConf",
                            f"spec = importlib.util.spec_from_file_location('phase_c_direct_{arm}', {str(wrapper)!r})",
                            "module = importlib.util.module_from_spec(spec)",
                            "assert spec.loader is not None",
                            "spec.loader.exec_module(module)",
                            f"cfg = OmegaConf.create(json.loads({json.dumps(config)!r}))",
                            "module._write_resolved_exclusive(cfg)",
                        )
                    )
                    completed = subprocess.run(
                        [sys.executable, "-c", child],
                        cwd=workspace,
                        env=self._capability_free_environment(),
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=45,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(
                        "missing required Phase-C capability environment",
                        completed.stderr,
                    )
                    self.assertFalse(resolved.exists())

    @staticmethod
    def _capability_free_environment() -> dict[str, str]:
        env = dict(os.environ)
        for variable in authorization.CAPABILITY_ENVIRONMENT.values():
            env.pop(variable, None)
        root = str(ROOT)
        inherited = env.get("PYTHONPATH")
        env["PYTHONPATH"] = root if not inherited else f"{root}{os.pathsep}{inherited}"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def _opening_fixture(
        self,
        temporary: Path,
        *,
        root: Path,
        private: Ed25519PrivateKey,
        public: Path,
        suffix: str,
        stage: str,
        scope: str,
        seeds: list[int],
    ) -> dict[str, Path]:
        program = _write_json(temporary / f"program-{suffix}.json", {"fixture": "program"})
        portable = _write_json(temporary / f"portable-{suffix}.json", {"fixture": "portable"})
        cost = _write_json(temporary / f"cost-{suffix}.json", {"fixture": "cost"})
        supplement = _write_json(temporary / f"supplement-{suffix}.json", {"fixture": "supplement"})
        shard = _write_json(
            temporary / f"shard-{suffix}.json",
            {
                "schema": "m2_post33_phase_c_shard_manifest_v4",
                "protocol_id": contract.PROTOCOL_ID,
                "phase_id": contract.PHASE_ID,
                "host_id": "unit-test-host",
                "gpu_id": "0",
                "arms_in_order": list(contract.ARMS),
                "paired_same_host_required": True,
                "absolute_cell_root": str(root.resolve()),
                "fold_allowlist": list(contract.FOLDS),
                "seed_allowlist": seeds,
                "portable_transfer_manifest_sha256": contract.sha256_file(portable),
            },
        )
        now = datetime.now(timezone.utc)
        body: dict[str, object] = {
            "schema": "m2_post33_phase_c_gpu_authorization_v4",
            "protocol_id": contract.PROTOCOL_ID,
            "phase_id": contract.PHASE_ID,
            "status": "GO",
            "authorization_id": f"fixture-{suffix}",
            "single_use_nonce": hashlib.sha256(suffix.encode("utf-8")).hexdigest(),
            "issued_at": (now - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
            "expires_at": (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "stage": stage,
            "capability_scope": scope,
            "host_id": "unit-test-host",
            "gpu_id": "0",
            "absolute_cell_root": str(root.resolve()),
            "fold_allowlist": list(contract.FOLDS),
            "seed_allowlist": seeds,
            "arms_in_order": list(contract.ARMS),
            "phase_c_program_receipt": contract.file_metadata(program),
            "portable_manifest": contract.file_metadata(portable),
            "shard_manifest": contract.file_metadata(shard),
            "evaluator": contract.file_metadata(
                ROOT / "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v4.py"
            ),
            "cost_receipt": contract.file_metadata(cost),
            "cost_supplement": contract.file_metadata(supplement),
            "public_key": contract.file_metadata(public),
        }
        if stage == "full_opening":
            decision = _write_json(temporary / f"decision-{suffix}.json", {"fixture": "decision"})
            signature = _write_json(temporary / f"decision-{suffix}.sig.json", {"fixture": "sig"})
            body["stage_a_decision"] = contract.file_metadata(decision)
            body["stage_a_decision_signature"] = contract.file_metadata(signature)
        auth = _write_json(
            temporary / f"authorization-{suffix}.json",
            {"schema": "m2_post33_phase_c_signed_authorization_envelope_v4", "authorization": body},
        )
        signature = (temporary / f"authorization-{suffix}.sig").resolve()
        signature.write_bytes(base64.b64encode(private.sign(auth.read_bytes())))
        return {
            "authorization": auth,
            "signature": signature,
            "program": program,
            "portable": portable,
            "shard": shard,
            "supplement": supplement,
        }


if __name__ == "__main__":
    unittest.main()
