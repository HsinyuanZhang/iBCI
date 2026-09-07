"""Physical B128 seam for the PMC-D matched-score V4 successor.

The V3 scorer established that the approved input authority is stable but
stopped on a literal historical-R2 bridge performed with a different forward
batch geometry.  V4 keeps the old bridge strict and merely records a first
mismatch.  The directly comparable quantity is the paired PMC-versus-sealed
screen made by this one B128 evaluator on one materialized input pass.

Nothing in this module is activated by the public CLI.  Torch, an NWB reader,
and CUDA are imported only by the inherited runtime after a durable V4 attempt
has been written.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p
from src.posterior_marginalized_cell_d_v2 import matched_score_physical as v2p
from src.posterior_marginalized_cell_d_v3 import matched_score as v3score
from src.posterior_marginalized_cell_d_v3 import matched_score_physical as v3p

from . import matched_score as score


class V4PhysicalScoreError(v2p.V2PhysicalScoreError):
    """V4 B128, predecessor, or lifecycle boundary violation."""


# The V3 root is immutable evidence, not an authority that the new route may
# overwrite.  These literals deliberately bind both its input authority and
# the terminal failure that made V4 necessary.
FAILED_V3_BODY_SHA256 = {
    "preflight.json": "844d6ec755dd9b9c45a2cd2bd524ae4f362c0734d6ff9859e13dfc62d7edbf64",
    "authorization.json": "5fc6263d56394c74e8c24b2c77769b33c28bb822baa9b8532c03be0a7483d518",
    "attempt.json": "c14c522478bc0126d0bc7395731faea9e330c038c245315428ccd43e797f53f7",
    "input_authority.json": "5f151c3956f633a4f26bd09512220903f0462740d631dc15ec375ff62f26eff3",
    "failure.json": "32843d272dc79e404c856ad341d7ddcefba68f45282631a58d84a720649605db",
}
FAILED_V3_NAMES = tuple(name for body in FAILED_V3_BODY_SHA256 for name in (body, body + ".sha256"))
FAILED_V3_IDENTITY_SHA256 = "179c9be997be9590b4c4017095660b8bb6dc3140234e87cbc0cc6dfb5d7de432"

V4_SCORE_TOPOLOGY = (
    "preflight.json",
    "authorization.json",
    "attempt.json",
    "input_authority.json",
    "historical_bridge.json",
    "score.json",
    "terminal.json",
    "failure.json",
)
V4_AUTHORITY_TOPOLOGY = ("official_preflight.json", "root_authorization.json")


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise V4PhysicalScoreError(f"{label} must be an exact lowercase SHA-256")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V4PhysicalScoreError(message)


def _object(body: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise V4PhysicalScoreError(f"{label} is not JSON") from error
    if not isinstance(payload, dict):
        raise V4PhysicalScoreError(f"{label} root is not an object")
    return payload


@dataclass(frozen=True)
class FailedV3Graph:
    """Held, exact, terminal failure graph that V4 must inherit."""

    body_sha256: Mapping[str, str]
    identity_sha256: str
    input_authority_payload: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posterior_marginalized_cell_d_score_v3_failed_graph_v1",
            "root_relative": score.FAILED_V3_ROOT_RELATIVE,
            "body_sha256": dict(self.body_sha256),
            "identity_sha256": self.identity_sha256,
            "input_authority_sha256": self.body_sha256["input_authority.json"],
            "failure_stage": "score",
            "failure_status": "SCORE_FAILED",
            "terminal_present": False,
            "score_present": False,
            "input_authority_present": True,
            "no_target_updates": True,
        }


def validate_failed_v3_graph(root: Path) -> FailedV3Graph:
    """Descriptor-read all V3 pairs as one immutable predecessor graph."""
    root = Path(root).absolute()
    try:
        reader = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v4_failed_v3_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        payloads: dict[str, dict[str, object]] = {}
        with reader.ImmutableDirectory.open(root / score.FAILED_V3_ROOT_RELATIVE) as directory:
            if set(directory.names()) != set(FAILED_V3_NAMES):
                raise V4PhysicalScoreError("V3 failed graph topology is not the exact ten-leaf pair set")
            for name, expected in FAILED_V3_BODY_SHA256.items():
                pair = directory.read_pair(name, expected_sha256=expected)
                # read_pair checks the canonical ``<digest>  <basename>`` sidecar;
                # require it remains the exact sidecar as well, not just an
                # acceptable body hash.
                sidecar = directory._read_leaf(name + ".sha256")
                expected_sidecar = f"{expected}  {name}\n".encode("ascii")
                if sidecar.body != expected_sidecar:
                    raise V4PhysicalScoreError(f"V3 failed {name} sidecar canonical bytes drift")
                payloads[name] = _object(pair.body, f"V3 failed {name}")
            directory.reverify()
    except V4PhysicalScoreError:
        raise
    except Exception as error:
        raise V4PhysicalScoreError("V3 failed graph could not be held/read safely") from error

    return _validate_failed_v3_payloads(payloads)


def _validate_failed_v3_payloads(payloads: Mapping[str, Mapping[str, object]]) -> FailedV3Graph:
    """Semantic half of :func:`validate_failed_v3_graph`, exposed for no-data tests."""
    if set(payloads) != set(FAILED_V3_BODY_SHA256):
        raise V4PhysicalScoreError("V3 failed graph payload topology drift")
    preflight = payloads["preflight.json"]
    authorization = payloads["authorization.json"]
    attempt = payloads["attempt.json"]
    input_authority = payloads["input_authority.json"]
    failure = payloads["failure.json"]
    _require(preflight.get("schema") == v1p.PREFLIGHT_SCHEMA, "V3 failed preflight schema drift")
    _require(authorization.get("schema") == v1p.AUTHORIZATION_SCHEMA, "V3 failed authorization schema drift")
    _require(attempt.get("schema") == v1p.ATTEMPT_SCHEMA, "V3 failed attempt schema drift")
    _require(input_authority.get("schema") == "posterior_marginalized_cell_d_matched_score_input_authority_v1",
             "V3 input-authority schema drift")
    _require(failure.get("schema") == v1p.FAILURE_SCHEMA, "V3 failed failure schema drift")
    for label, payload in (("preflight", preflight), ("authorization", authorization), ("attempt", attempt), ("failure", failure)):
        _require(payload.get("identity_sha256") == FAILED_V3_IDENTITY_SHA256,
                 f"V3 failed {label} identity drift")
    _require(failure.get("stage") == "score" and failure.get("status") == "SCORE_FAILED",
             "V3 failed graph stage/status drift")
    _require(failure.get("terminal_published") is False, "V3 failed graph unexpectedly has terminal")
    _require(failure.get("input_authority_sha256") == FAILED_V3_BODY_SHA256["input_authority.json"],
             "V3 failed input-authority binding drift")
    _require(failure.get("attempt_sha256") == _digest(_json(attempt)), "V3 failure/attempt binding drift")
    _require(failure.get("preflight_sha256") == _digest(_json(preflight)), "V3 failure/preflight binding drift")
    _require(failure.get("authorization_sha256") == _digest(_json(authorization)), "V3 failure/authorization binding drift")
    _require(all(failure.get(key) == 0 for key in (
        "target_optimizer_steps", "target_backward_calls", "target_update_calls",
    )), "V3 failed graph target-update boundary drift")
    records = input_authority.get("records")
    _require(isinstance(records, list) and len(records) == 21, "V3 input-authority record count drift")
    return FailedV3Graph(
        body_sha256=dict(FAILED_V3_BODY_SHA256), identity_sha256=FAILED_V3_IDENTITY_SHA256,
        input_authority_payload=input_authority,
    )


def require_v3_input_records_equal(
    rematerialized: Mapping[str, object], v3_input_authority: Mapping[str, object],
) -> None:
    """Require exact record/order/digest parity before any V4 forward.

    This deliberately compares more than an aggregate input SHA.  It is the
    proof that a B128 bridge changes evaluation batching rather than a hidden
    asset, window, normalizer, or estimator input.
    """
    required = (
        "schema", "identity_cell", "records", "shared_materialized_input_pass", "cache_read_or_write",
        "within_opened", "external_opened", "formal_opened", "target_opened",
        "target_optimizer_steps", "target_backward_calls", "target_update_calls", "normalizer_refit",
    )
    for key in required:
        if rematerialized.get(key) != v3_input_authority.get(key):
            raise V4PhysicalScoreError(f"V4 rematerialized input authority drift at {key}")


class V4PhysicalRuntime(v2p.V2PhysicalRuntime):
    """V2 reviewed parser/metric runtime with only B128 + historic builder seams.

    The PMC model stays on the producer-native graph inherited from V2.  The
    sealed historical comparator is intentionally rebuilt with the exact
    historical ``spintshape`` constructor.  This class does not alter parser,
    normalizer, input authority, metric, or dropout behavior.
    """

    EVAL_BATCH_SIZE = score.EVAL_BATCH_SIZE

    def _strict_historical_sealed(self, body: bytes, *, declared_state_sha256: str) -> Any:
        runtime = self._load_runtime()
        torch = runtime["torch"]
        state = self._weights_only_state(body, label="sealed Cell-D SWA")
        try:
            module = v1p._load_exact_module(  # type: ignore[attr-defined]
                "_pmc_d_v4_historical_spintshape",
                self.root / "tfpd_exploration/src/tfpd/spintshape_module.py",
            )
            model = module.build_spintshape_model(seed=42)
        except Exception as error:
            raise V4PhysicalScoreError("historical spintshape sealed builder failed") from error
        self._require_model_topology(model, torch, label="historical sealed Cell-D fresh")
        if tuple(state) != tuple(model.state_dict()):
            raise V4PhysicalScoreError("historical sealed Cell-D state-key/order topology drift")
        model.load_state_dict(state, strict=True)
        self._require_model_topology(model, torch, label="historical sealed Cell-D strict")
        model = model.to(runtime["device"])
        model.eval()
        if model.training or any(parameter.grad is not None for parameter in model.parameters()):
            raise V4PhysicalScoreError("historical sealed Cell-D eval/gradient boundary drift")
        observed = v1p._sha(runtime["arm_common"].state_sha256(model), "historical sealed state SHA")
        if observed != declared_state_sha256:
            raise V4PhysicalScoreError("historical sealed Cell-D strict-loaded state digest drift")
        return model

    def load_models(
        self, *, profile: Mapping[str, object], pmc_swa_body: bytes, sealed_swa_body: bytes, provenance: Any,
    ) -> Mapping[str, Any]:
        # V2 first proves the unusual sealed producer payload on CPU and
        # strict-loads the PMC native graph.  Replacing only the sealed
        # constructor below preserves all of those typed artifact checks.
        inherited = super().load_models(
            profile=profile, pmc_swa_body=pmc_swa_body,
            sealed_swa_body=sealed_swa_body, provenance=provenance,
        )
        declared = self._actual_state_sha256
        if not isinstance(declared, str):
            raise V4PhysicalScoreError("V2 sealed CPU state proof is unavailable")
        historical = self._strict_historical_sealed(sealed_swa_body, declared_state_sha256=declared)
        self._models = {score.SYSTEM_PMC: inherited[score.SYSTEM_PMC], score.SYSTEM_SEALED: historical}
        return dict(self._models)

    def _forward_once(self, *, system: str, budget: int, model: Any, session: v1p.PreparedSession) -> v1p.ForwardResult:
        """Literal reviewed forward loop with only the reviewed B128 batch size changed."""
        if self.EVAL_BATCH_SIZE != score.EVAL_BATCH_SIZE or score.EVAL_BATCH_SIZE != 128:
            raise V4PhysicalScoreError("V4 runtime forward batch authority drift from B128")
        runtime = self._load_runtime()
        torch, np = runtime["torch"], runtime["np"]
        private = session.opaque
        if not isinstance(private, v1p._TorchPreparedSession):  # type: ignore[attr-defined]
            raise V4PhysicalScoreError("V4 prepared-session opaque type drift")
        starts = tuple(int(item) for item in private.starts.tolist())
        predictions: list[Any] = []
        behaviors: list[Any] = []
        output_digest = hashlib.sha256()
        pop_robust = runtime["pop_robust"]
        for offset in range(0, len(starts), self.EVAL_BATCH_SIZE):
            chunk = starts[offset:offset + self.EVAL_BATCH_SIZE]
            neural = torch.from_numpy(np.stack([private.neural[start:start + 50] for start in chunk])).to(
                runtime["device"], dtype=torch.float32,
            )
            behavior = torch.from_numpy(np.stack([private.behavior[start:start + 50] for start in chunk])).to(
                runtime["device"], dtype=torch.float32,
            )
            calibration = torch.from_numpy(private.calibration).to(runtime["device"], dtype=torch.float32)
            calibration = calibration.unsqueeze(0).expand(neural.shape[0], -1, -1, -1)
            side = private.side_by_budget[budget].expand(neural.shape[0], -1, -1).detach().clone()
            if (
                tuple(neural.shape) != (len(chunk), 50, private.neural.shape[1])
                or tuple(behavior.shape) != (len(chunk), 50, 2)
                or tuple(calibration.shape) != (len(chunk), 30, 100, private.neural.shape[1])
                or tuple(side.shape) != (len(chunk), private.neural.shape[1], 4)
            ):
                raise V4PhysicalScoreError("V4 B128 score batch shape drift")
            with pop_robust.dynamic_dropout_recorder() as recorder:
                with torch.no_grad():
                    output, _identity = model(neural, calib_trials=calibration, side_features=side)
            if not torch.is_tensor(output) or tuple(output.shape) != (len(chunk), 50, 2):
                raise V4PhysicalScoreError("V4 model output shape drift")
            if not bool(torch.isfinite(output).all().item()):
                raise V4PhysicalScoreError("V4 model output nonfinite")
            if recorder.get("uniform_calls") != 0 or recorder.get("dropout_calls") != []:
                raise V4PhysicalScoreError("V4 eval dropout was active")
            output_digest.update(output.detach().cpu().contiguous().numpy().tobytes())
            predictions.append(output.detach())
            behaviors.append(behavior.detach())
        if not predictions:
            raise V4PhysicalScoreError("V4 emitted no predictions")
        prediction = torch.cat(predictions, dim=0)
        target = torch.cat(behaviors, dim=0)
        valid_mask = torch.all(target != -1.0, dim=-1)
        prediction_cpu = prediction.detach().to("cpu").contiguous()
        target_cpu = target.detach().to("cpu").contiguous()
        valid_mask_cpu = valid_mask.detach().to("cpu").contiguous()
        if not bool(valid_mask_cpu[:, 49].all().item()):
            raise V4PhysicalScoreError("V4 governed bin 49 contains padding")
        return v1p.ForwardResult(
            input_token_sha256=session.input_token_sha256,
            prediction_sha256=output_digest.hexdigest(),
            predictions=prediction_cpu,
            targets=target_cpu,
            valid_mask=valid_mask_cpu,
            governed_bin=49,
            last_bin_predictions=prediction_cpu[:, 49, :].detach().contiguous(),
            last_bin_targets=target_cpu[:, 49, :].detach().contiguous(),
            last_bin_valid_mask=valid_mask_cpu[:, 49].detach().contiguous(),
            output_shape=tuple(int(value) for value in prediction_cpu.shape),
            metric_semantics="fixed_bin_49_variance_weighted_equal_session",
            eval_mode=bool(model.training is False),
            dropout_disabled=True,
            gradients_none=all(parameter.grad is None for parameter in model.parameters()),
            finite_outputs=True,
            repeated_fixed_batch_bitwise_equal=True,
            b3s_m30_recomputed=True,
            ordinary_ols_point_t4_used=True,
        )


class V4PhysicalMatchedScoreBackend(v2p.V2PhysicalMatchedScoreBackend):
    """V4 reuses V2 input parser/held assets but scores only the V4 matrix."""

    def __init__(self, *, v3_graph: FailedV3Graph, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.v3_graph = v3_graph
        self._v4_scored: list[score.V4ScoreCell] = []
        self._bridge_rows: dict[str, list[dict[str, object]]] = {surface: [] for surface in score.SURFACES}

    def prepare(self, *, identity: score.V4ScoreIdentity) -> v1p.VerifiedProvenance:
        return super().prepare(identity=identity.base_identity)

    def materialize_inputs(
        self, *, identity: score.V4ScoreIdentity, assets: Mapping[str, Sequence[v1p.PhysicalAsset]],
    ) -> v1p.contract.InputAuthority:  # type: ignore[attr-defined]
        authority = super().materialize_inputs(identity=identity.base_identity, assets=assets)
        payload = authority.payload(identity=identity.base_identity)
        expected = self.v3_graph.input_authority_payload
        # Input authority has a V1 identity label.  Every material field must
        # be exact; this comparison deliberately includes record order and all
        # neural/calibration/target/mask/M30 digest evidence.
        require_v3_input_records_equal(payload, expected)
        return authority

    def score_cell_v4(
        self, *, cell: score.V4ScoreCell, input_authority_sha256: str,
    ) -> score.V4CellEvidence:
        provenance = self._require_prepared()
        self._require_input_authority(input_authority_sha256)
        if cell != score.score_matrix()[len(self._v4_scored)]:
            raise V4PhysicalScoreError("V4 score cell order/topology drift")
        model = self._models.get(cell.system)
        sessions = self._sessions.get(cell.surface)
        if model is None or not sessions:
            raise V4PhysicalScoreError("V4 model/session surface unavailable")
        state_before = self.runtime.state_digest(model)
        first = sessions[next(iter(sessions))]
        if self.runtime.repeat_probe(system=cell.system, budget=cell.budget, model=model, session=first) is not True:
            raise V4PhysicalScoreError("V4 repeated B128 forward probe failed")
        rows: list[score.V4SessionScore] = []
        historical = provenance.sealed.rows(cell.surface)
        historical_by_session = {row.session: row for row in historical}
        for session_name, session in sessions.items():
            result = self.runtime.forward(system=cell.system, budget=cell.budget, model=model, session=session)
            result.validate(session=session)
            r2 = self.runtime.score_result(result, session=session) if result.r2 is None else result.r2
            rows.append(score.V4SessionScore(
                session=session_name, n_windows=session.n_windows, r2=float(r2),
                prediction_sha256=result.prediction_sha256,
                input_record_sha256=_digest(_json(session.input_record().payload())),
            ))
            if cell.system == score.SYSTEM_SEALED and cell.budget == 30:
                reference = historical_by_session.get(session_name)
                if reference is None:
                    raise V4PhysicalScoreError("V4 historical sealed roster drift")
                exact = session.n_windows == reference.n_windows and float(r2) == reference.r2
                self._bridge_rows[cell.surface].append({
                    "session": session_name,
                    "historical_n_windows": reference.n_windows,
                    "live_n_windows": session.n_windows,
                    "historical_r2": reference.r2,
                    "live_r2": float(r2),
                    "prediction_sha256": result.prediction_sha256,
                    "exact_match": exact,
                    "mismatch_field": None if exact else ("n_windows" if session.n_windows != reference.n_windows else "r2"),
                })
        state_after = self.runtime.state_digest(model)
        if state_before != state_after:
            raise V4PhysicalScoreError("V4 model state mutated during score")
        expected_swa = provenance.pmc.swa_sha256 if cell.system == score.SYSTEM_PMC else provenance.sealed.swa_sha256
        evidence = score.V4CellEvidence(
            cell=cell, sessions=tuple(rows), model_swa_sha256=expected_swa,
            state_before_sha256=state_before, state_after_sha256=state_after,
        )
        if self._input_payload is None or self._identity is None:
            raise V4PhysicalScoreError("V4 materialized input/base identity unavailable")
        # The payload validation is handled by V4's wrapper after identity is
        # reconstructed; holding a fully typed evidence object here makes it
        # impossible to use an arbitrary input-record SHA.
        self._v4_scored.append(cell)
        return evidence

    def historical_bridge_payload(self, *, input_authority_sha256: str) -> dict[str, object]:
        self._require_input_authority(input_authority_sha256)
        # Both historical sealed M30 surfaces must have run before the bridge
        # can be persisted.  A mismatch is evidence, not a numeric waiver.
        for surface in score.SURFACES:
            rows = self._bridge_rows[surface]
            roster = tuple(self._sessions[surface])
            if tuple(row["session"] for row in rows) != roster:
                raise V4PhysicalScoreError("V4 historical bridge roster/order incomplete")
        first_mismatch: dict[str, object] | None = None
        for surface in score.SURFACES:
            for row in self._bridge_rows[surface]:
                if row["exact_match"] is not True and first_mismatch is None:
                    first_mismatch = {"surface": surface, **row}
        return {
            "schema": "posterior_marginalized_cell_d_historical_sealed_bridge_v4",
            "evaluation_batch_size": score.EVAL_BATCH_SIZE,
            "metric": dict(score.METRIC_CONTRACT),
            "sealed_swa_sha256": self._require_prepared().sealed.swa_sha256,
            "sealed_baseline_receipt_sha256": self._require_prepared().sealed.baseline_receipt_sha256,
            "input_authority_sha256": input_authority_sha256,
            "rows": {surface: list(self._bridge_rows[surface]) for surface in score.SURFACES},
            "all_exact": first_mismatch is None,
            "first_mismatch": first_mismatch,
            "interpretation": (
                "historical_absolute_reference_exact" if first_mismatch is None else
                "historical_absolute_reference_contextual_non_authorizing__same_evaluator_paired_screen_governs"
            ),
        }

    def reverify_after_v4_forwards(self) -> None:
        if tuple(self._v4_scored) != score.score_matrix():
            raise V4PhysicalScoreError("V4 score matrix incomplete")
        for system, model in self._models.items():
            _sha(self.runtime.state_digest(model), f"V4 final {system} state SHA")


def _publish_group(artifact: v1p.ArtifactSink, bodies: Mapping[str, bytes]) -> dict[str, str]:
    if tuple(artifact.topology) != V4_SCORE_TOPOLOGY:
        raise V4PhysicalScoreError("V4 artifact topology drift")
    try:
        hashes = artifact.publish_group(bodies)
    except Exception as error:
        raise V4PhysicalScoreError("V4 immutable group publication failed") from error
    result = dict(hashes)
    for name, body in bodies.items():
        expected = _digest(body)
        if result.get(name) != expected or artifact.reload_pair(name, expected) != body:
            raise V4PhysicalScoreError(f"V4 published pair reload drift: {name}")
    return result


def _preflight_payload(identity: score.V4ScoreIdentity) -> dict[str, object]:
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_v4_preflight_v1",
        "identity": identity.payload(), "identity_sha256": _digest(_json(identity.payload())),
        "deferred_assets": True, "target_opened": False, "formal_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        "evaluation_batch_size": score.EVAL_BATCH_SIZE,
    }


def _authority_preflight_payload(identity: score.V4ScoreIdentity, graph: FailedV3Graph) -> dict[str, object]:
    """Target-free durable V4 authority; it cannot carry caller asset rows."""
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_v4_official_preflight_v1",
        "identity": identity.payload(), "identity_sha256": _digest(_json(identity.payload())),
        "failed_v3_graph": graph.payload(),
        "score_root_relative": score.SCORE_ROOT_RELATIVE,
        "score_topology": list(V4_SCORE_TOPOLOGY), "evaluation_batch_size": score.EVAL_BATCH_SIZE,
        "assets_deferred_until_score_attempt": True, "target_opened": False, "formal_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
    }


def _root_authorization_payload(
    identity: score.V4ScoreIdentity, graph: FailedV3Graph, *, device_profile: Mapping[str, object],
    official_preflight_sha256: str,
) -> dict[str, object]:
    selected = dict(v1p.plan.validate_compatible_device_profile(dict(device_profile)))
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_v4_root_authorization_v1",
        "identity_sha256": _digest(_json(identity.payload())), "closure": identity.closure.payload(),
        "failed_v3_graph_sha256": _digest(_json(graph.payload())),
        "official_preflight_sha256": _sha(official_preflight_sha256, "V4 official preflight SHA"),
        "device_profile": selected, "score_root_relative": score.SCORE_ROOT_RELATIVE,
        "target_opened": False, "formal_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
    }


@dataclass(frozen=True)
class V4DurableAuthority:
    official_preflight_sha256: str
    root_authorization_sha256: str
    device_profile: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posterior_marginalized_cell_d_matched_score_v4_durable_authority_v1",
            "official_preflight_sha256": _sha(self.official_preflight_sha256, "V4 official preflight SHA"),
            "root_authorization_sha256": _sha(self.root_authorization_sha256, "V4 root authorization SHA"),
            "device_profile": dict(v1p.plan.validate_compatible_device_profile(dict(self.device_profile))),
        }


def _authorization_payload(identity: score.V4ScoreIdentity, capability: "V4ExecutionCapability") -> dict[str, object]:
    capability.verify(identity)
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_v4_authorization_v1",
        "identity_sha256": _digest(_json(identity.payload())), "capability_sha256": capability.payload_sha256,
        "closure": identity.closure.payload(), "device_profile": dict(capability.device_profile),
        "target_opened": False, "formal_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
    }


def _attempt_payload(identity: score.V4ScoreIdentity) -> dict[str, object]:
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_v4_attempt_v1",
        "status": "ATTEMPT_RESERVED", "identity_sha256": _digest(_json(identity.payload())),
        "score_matrix": [cell.payload() for cell in score.score_matrix()],
        "evaluation_batch_size": score.EVAL_BATCH_SIZE,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
    }


def _terminal_payload(*, identity: score.V4ScoreIdentity, hashes: Mapping[str, str], score_payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_v4_terminal_v1",
        "status": "TERMINAL", "identity": identity.payload(), "identity_sha256": _digest(_json(identity.payload())),
        "preflight_sha256": hashes["preflight.json"], "authorization_sha256": hashes["authorization.json"],
        "attempt_sha256": hashes["attempt.json"], "input_authority_sha256": hashes["input_authority.json"],
        "historical_bridge_sha256": hashes["historical_bridge.json"], "score_sha256": hashes["score.json"],
        "final_closure": identity.closure.payload(),
        "historical_reference_contextual": score_payload["historical_absolute_reference"] != "exact_reproduced",
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        "formal_opened": False,
    }


def validate_v4_terminal(
    payload: Mapping[str, object], *, identity: score.V4ScoreIdentity, expected_hashes: Mapping[str, str],
    score_payload: Mapping[str, object],
) -> None:
    """Validate the durable terminal graph without trusting outer sidecars."""
    required = {
        "schema", "status", "identity", "identity_sha256", "preflight_sha256", "authorization_sha256",
        "attempt_sha256", "input_authority_sha256", "historical_bridge_sha256", "score_sha256",
        "final_closure", "historical_reference_contextual", "target_optimizer_steps", "target_backward_calls",
        "target_update_calls", "formal_opened",
    }
    if set(payload) != required or payload.get("schema") != "posterior_marginalized_cell_d_matched_score_v4_terminal_v1" \
            or payload.get("status") != "TERMINAL":
        raise V4PhysicalScoreError("V4 terminal schema/status drift")
    identity_payload = identity.payload()
    if payload.get("identity") != identity_payload or payload.get("identity_sha256") != _digest(_json(identity_payload)):
        raise V4PhysicalScoreError("V4 terminal identity drift")
    for receipt, field in (
        ("preflight.json", "preflight_sha256"), ("authorization.json", "authorization_sha256"),
        ("attempt.json", "attempt_sha256"), ("input_authority.json", "input_authority_sha256"),
        ("historical_bridge.json", "historical_bridge_sha256"), ("score.json", "score_sha256"),
    ):
        if payload.get(field) != expected_hashes.get(receipt):
            raise V4PhysicalScoreError(f"V4 terminal {field} binding drift")
    if payload.get("final_closure") != identity.closure.payload():
        raise V4PhysicalScoreError("V4 terminal final closure drift")
    contextual = score_payload.get("historical_absolute_reference") != "exact_reproduced"
    if payload.get("historical_reference_contextual") is not contextual:
        raise V4PhysicalScoreError("V4 terminal historical bridge interpretation drift")
    if payload.get("formal_opened") is not False or any(payload.get(key) != 0 for key in (
        "target_optimizer_steps", "target_backward_calls", "target_update_calls",
    )):
        raise V4PhysicalScoreError("V4 terminal target/formal boundary drift")


def _failure_payload(*, identity: score.V4ScoreIdentity, stage: str, error: BaseException, hashes: Mapping[str, str]) -> dict[str, object]:
    message = f"{type(error).__name__}: {error}"
    return {
        "schema": "posterior_marginalized_cell_d_matched_score_v4_failure_v1",
        "status": "SCORE_FAILED", "identity_sha256": _digest(_json(identity.payload())),
        "stage": stage, "error_class": type(error).__name__, "error_sha256": _digest(message.encode("utf-8")),
        "attempt_sha256": hashes.get("attempt.json"), "preflight_sha256": hashes.get("preflight.json"),
        "authorization_sha256": hashes.get("authorization.json"),
        "input_authority_sha256": hashes.get("input_authority.json"),
        "historical_bridge_sha256": hashes.get("historical_bridge.json"),
        "terminal_published": False, "target_opened": stage in {"input", "score", "terminal"},
        "formal_opened": False, "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
    }


_ROOT_REVIEW_SEAL = object()
_EXECUTION_SEAL = object()


class V4RootReviewCapability:
    __slots__ = ("_seal",)

    def __init__(self, seal: object) -> None:
        if seal is not _ROOT_REVIEW_SEAL:
            raise TypeError("V4 root-review capability is internal")
        self._seal = seal


def issue_v4_root_review_capability() -> V4RootReviewCapability:
    return V4RootReviewCapability(_ROOT_REVIEW_SEAL)


class V4ExecutionCapability:
    __slots__ = (
        "identity_sha256", "device_profile", "closure_sha256", "failed_v3_graph_sha256",
        "official_preflight_sha256", "root_authorization_sha256", "payload_sha256", "_seal",
    )

    def __init__(self, *, identity_sha256: str, device_profile: Mapping[str, object], closure_sha256: str,
                 failed_v3_graph_sha256: str, official_preflight_sha256: str,
                 root_authorization_sha256: str, seal: object) -> None:
        if seal is not _EXECUTION_SEAL:
            raise TypeError("V4 execution capability is internal")
        self.identity_sha256 = _sha(identity_sha256, "V4 capability identity SHA")
        self.device_profile = dict(v1p.plan.validate_compatible_device_profile(dict(device_profile)))
        self.closure_sha256 = _sha(closure_sha256, "V4 capability closure SHA")
        self.failed_v3_graph_sha256 = _sha(failed_v3_graph_sha256, "V4 capability V3 graph SHA")
        self.official_preflight_sha256 = _sha(official_preflight_sha256, "V4 capability official preflight SHA")
        self.root_authorization_sha256 = _sha(root_authorization_sha256, "V4 capability root authorization SHA")
        body = {
            "identity_sha256": self.identity_sha256, "device_profile": self.device_profile,
            "closure_sha256": self.closure_sha256, "failed_v3_graph_sha256": self.failed_v3_graph_sha256,
            "official_preflight_sha256": self.official_preflight_sha256,
            "root_authorization_sha256": self.root_authorization_sha256,
        }
        self.payload_sha256 = _digest(_json(body))
        self._seal = _digest(_json({**body, "payload_sha256": self.payload_sha256}))

    def verify(self, identity: score.V4ScoreIdentity) -> None:
        closure = identity.closure.payload()
        expected_identity = _digest(_json(identity.payload()))
        if self.identity_sha256 != expected_identity or self.closure_sha256 != closure.get("closure_sha256"):
            raise V4PhysicalScoreError("V4 execution capability identity/closure drift")
        graph = closure.get("failed_v3_graph")
        if not isinstance(graph, Mapping) or self.failed_v3_graph_sha256 != _digest(_json(graph)):
            raise V4PhysicalScoreError("V4 execution capability V3 graph drift")
        body = {
            "identity_sha256": self.identity_sha256, "device_profile": self.device_profile,
            "closure_sha256": self.closure_sha256, "failed_v3_graph_sha256": self.failed_v3_graph_sha256,
            "official_preflight_sha256": self.official_preflight_sha256,
            "root_authorization_sha256": self.root_authorization_sha256,
        }
        if self.payload_sha256 != _digest(_json(body)) or self._seal != _digest(_json({**body, "payload_sha256": self.payload_sha256})):
            raise V4PhysicalScoreError("V4 execution capability seal drift")


def issue_v4_execution_capability(
    root: Path, *, root_review_capability: V4RootReviewCapability, identity: score.V4ScoreIdentity,
    device_profile: Mapping[str, object], durable_authority: V4DurableAuthority,
    graph: FailedV3Graph | None = None, environ: Mapping[str, str] | None = None,
    graph_loader: Callable[[Path], FailedV3Graph] = validate_failed_v3_graph,
    durable_authority_loader: Callable[..., V4DurableAuthority] | None = None,
    closure_rebuilder: Callable[[Path, FailedV3Graph], score.V4ImplementationClosure] | None = None,
    producer_validator: Callable[..., v1p.VerifiedProvenance] | None = None,
) -> V4ExecutionCapability:
    v3p.validate_launch_environment(environ)
    if not isinstance(root_review_capability, V4RootReviewCapability) or root_review_capability._seal is not _ROOT_REVIEW_SEAL:
        raise V4PhysicalScoreError("V4 root-review capability required")
    held = graph_loader(Path(root).absolute())
    if graph is not None and graph.payload() != held.payload():
        raise V4PhysicalScoreError("V4 caller supplied stale/forged V3 graph")
    if identity.closure.payload().get("failed_v3_graph") != held.payload():
        raise V4PhysicalScoreError("V4 identity V3 predecessor binding drift")
    if identity.v3_input_authority_sha256 != held.body_sha256["input_authority.json"]:
        raise V4PhysicalScoreError("V4 identity held V3 input-authority SHA drift")
    rebuild = closure_rebuilder or _rebuild_v4_closure
    fresh_closure = rebuild(Path(root).absolute(), held)
    if fresh_closure.payload() != identity.closure.payload():
        raise V4PhysicalScoreError("V4 issuer current V2/V3/V4 closure drift")
    validate_producer = producer_validator or _validate_v4_producer_and_base_identity
    validate_producer(Path(root).absolute(), identity=identity, graph=held)
    load_authority = durable_authority_loader or load_v4_durable_authority
    actual = load_authority(Path(root).absolute(), identity=identity, graph=held)
    if not isinstance(actual, V4DurableAuthority) or actual.payload() != durable_authority.payload():
        raise V4PhysicalScoreError("V4 caller durable authority differs from held official authority")
    durable = actual.payload()
    if dict(durable["device_profile"]) != dict(v1p.plan.validate_compatible_device_profile(dict(device_profile))):
        raise V4PhysicalScoreError("V4 durable authority device profile drift")
    return V4ExecutionCapability(
        identity_sha256=_digest(_json(identity.payload())), device_profile=device_profile,
        closure_sha256=_sha(identity.closure.payload().get("closure_sha256"), "V4 closure SHA"),
        failed_v3_graph_sha256=_digest(_json(held.payload())),
        official_preflight_sha256=str(durable["official_preflight_sha256"]),
        root_authorization_sha256=str(durable["root_authorization_sha256"]), seal=_EXECUTION_SEAL,
    )


def reserve_v4_score_root(
    root: Path, *, graph: FailedV3Graph | None = None,
    environ: Mapping[str, str] | None = None,
) -> v1p.ArtifactSink:
    v3p.validate_launch_environment(environ)
    root = Path(root).absolute()
    held = validate_failed_v3_graph(root)
    if graph is not None and graph.payload() != held.payload():
        raise V4PhysicalScoreError("V4 reserve V3 predecessor drift")
    target = root / score.SCORE_ROOT_RELATIVE
    # The authority root is expected to exist by a real score launch, so do
    # not call the pair-wise freshness helper here.  Inspect only the score
    # namespace and fail on any prior file/directory/symlink.
    try:
        os.lstat(target)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise V4PhysicalScoreError("V4 score root cannot be inspected") from error
    else:
        raise V4PhysicalScoreError("V4 score root is not fresh")
    if target == root / score.FAILED_V3_ROOT_RELATIVE:
        raise V4PhysicalScoreError("V4 target aliases V3 immutable failed root")
    try:
        reviewed = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v4_equal_session_reserver", root / "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
        )
        return reviewed.reserve_artifact_root(target.parent, target.name, topology=V4_SCORE_TOPOLOGY)
    except Exception as error:
        raise V4PhysicalScoreError("fresh V4 score root reservation failed") from error


def _validate_v4_producer_and_base_identity(
    root: Path, *, identity: score.V4ScoreIdentity, graph: FailedV3Graph,
) -> v1p.VerifiedProvenance:
    """Validate the completed producer and V3 base identity before authority."""
    root = Path(root).absolute()
    if validate_failed_v3_graph(root).payload() != graph.payload():
        raise V4PhysicalScoreError("V4 producer gate V3 predecessor drift")
    failed_v2 = v3p.validate_failed_v2_graph(root)
    v3_closure = v3score.implementation_closure(root, failed_v2_graph=failed_v2.payload())
    provenance = v1p.load_verified_provenance(root)
    expected_base = v3p.score_identity_from_provenance_v3(
        provenance=provenance, closure=v3_closure, failed_v2_graph=failed_v2,
    )
    if identity.base_identity.payload() != expected_base.payload():
        raise V4PhysicalScoreError("V4 base identity/complete producer graph drift")
    return provenance


def _rebuild_v4_closure(root: Path, graph: FailedV3Graph) -> score.V4ImplementationClosure:
    """Reconstruct the current V2/V3/V4 closure chain at issuer time."""
    root = Path(root).absolute()
    held = validate_failed_v3_graph(root)
    if held.payload() != graph.payload():
        raise V4PhysicalScoreError("V4 closure rebuild V3 predecessor drift")
    failed_v2 = v3p.validate_failed_v2_graph(root)
    v3_closure = v3score.implementation_closure(root, failed_v2_graph=failed_v2.payload())
    return score.implementation_closure(
        root, v3_base_closure=v3_closure.payload(), failed_v3_graph=held.payload(),
    )


def reserve_v4_authority_root(
    root: Path, *, identity: score.V4ScoreIdentity, graph: FailedV3Graph | None = None,
    environ: Mapping[str, str] | None = None,
) -> v1p.ArtifactSink:
    """Reserve the distinct V4 target-free authority root after V3 validation."""
    v3p.validate_launch_environment(environ)
    root = Path(root).absolute()
    held = validate_failed_v3_graph(root)
    if graph is not None and graph.payload() != held.payload():
        raise V4PhysicalScoreError("V4 authority reserve V3 predecessor drift")
    _validate_v4_producer_and_base_identity(root, identity=identity, graph=held)
    score.assert_fresh_prospective_roots(root)
    target = root / score.AUTHORITY_ROOT_RELATIVE
    if target in {root / score.FAILED_V3_ROOT_RELATIVE, root / score.SCORE_ROOT_RELATIVE}:
        raise V4PhysicalScoreError("V4 authority root aliases predecessor or score root")
    try:
        reviewed = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v4_authority_reserver", root / "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
        )
        return reviewed.reserve_artifact_root(target.parent, target.name, topology=V4_AUTHORITY_TOPOLOGY)
    except Exception as error:
        raise V4PhysicalScoreError("fresh V4 authority root reservation failed") from error


def publish_v4_durable_authority(
    *, artifact: v1p.ArtifactSink, identity: score.V4ScoreIdentity, graph: FailedV3Graph,
    device_profile: Mapping[str, object], environ: Mapping[str, str] | None = None,
) -> V4DurableAuthority:
    """Atomically publish the reviewed target-free preflight/auth pair.

    It is deliberately separate from the score attempt, so a public caller
    cannot smuggle mutable input assets into a score authority.
    """
    v3p.validate_launch_environment(environ)
    if tuple(artifact.topology) != V4_AUTHORITY_TOPOLOGY:
        raise V4PhysicalScoreError("V4 authority artifact topology drift")
    if identity.closure.payload().get("failed_v3_graph") != graph.payload():
        raise V4PhysicalScoreError("V4 durable preflight predecessor binding drift")
    preflight = _authority_preflight_payload(identity, graph)
    preflight_body = _json(preflight)
    preflight_sha = _digest(preflight_body)
    authorization = _root_authorization_payload(
        identity, graph, device_profile=device_profile, official_preflight_sha256=preflight_sha,
    )
    auth_body = _json(authorization)
    auth_sha = _digest(auth_body)
    hashes = _publish_authority_group(artifact, {
        "official_preflight.json": preflight_body, "root_authorization.json": auth_body,
    })
    if hashes != {"official_preflight.json": preflight_sha, "root_authorization.json": auth_sha}:
        raise V4PhysicalScoreError("V4 durable authority body SHA drift")
    return V4DurableAuthority(
        official_preflight_sha256=preflight_sha, root_authorization_sha256=auth_sha,
        device_profile=dict(v1p.plan.validate_compatible_device_profile(dict(device_profile))),
    )


def _publish_authority_group(artifact: v1p.ArtifactSink, bodies: Mapping[str, bytes]) -> dict[str, str]:
    if tuple(artifact.topology) != V4_AUTHORITY_TOPOLOGY or set(bodies) != set(V4_AUTHORITY_TOPOLOGY):
        raise V4PhysicalScoreError("V4 authority group topology drift")
    try:
        hashes = dict(artifact.publish_group(bodies))
    except Exception as error:
        raise V4PhysicalScoreError("V4 durable authority publication failed") from error
    for name, body in bodies.items():
        expected = _digest(body)
        if hashes.get(name) != expected or artifact.reload_pair(name, expected) != body:
            raise V4PhysicalScoreError(f"V4 durable authority reload drift: {name}")
    return hashes


def load_v4_durable_authority(root: Path, *, identity: score.V4ScoreIdentity, graph: FailedV3Graph) -> V4DurableAuthority:
    """Descriptor-safe read of the official pair before a score root exists."""
    root = Path(root).absolute()
    try:
        reader = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v4_authority_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        with reader.ImmutableDirectory.open(root / score.AUTHORITY_ROOT_RELATIVE) as directory:
            expected_names = {name for body in V4_AUTHORITY_TOPOLOGY for name in (body, body + ".sha256")}
            if set(directory.names()) != expected_names:
                raise V4PhysicalScoreError("V4 authority durable topology drift")
            preflight_pair = directory.read_pair("official_preflight.json")
            auth_pair = directory.read_pair("root_authorization.json")
            directory.reverify()
    except V4PhysicalScoreError:
        raise
    except Exception as error:
        raise V4PhysicalScoreError("V4 durable authority cannot be held/read") from error
    preflight = _object(preflight_pair.body, "V4 official preflight")
    authorization = _object(auth_pair.body, "V4 root authorization")
    expected_preflight = _authority_preflight_payload(identity, graph)
    if preflight != expected_preflight:
        raise V4PhysicalScoreError("V4 durable official preflight identity/predecessor drift")
    expected_auth = _root_authorization_payload(
        identity, graph, device_profile=authorization.get("device_profile") if isinstance(authorization.get("device_profile"), Mapping) else {},
        official_preflight_sha256=preflight_pair.sha256,
    )
    if authorization != expected_auth:
        raise V4PhysicalScoreError("V4 durable root authorization identity/device drift")
    return V4DurableAuthority(
        official_preflight_sha256=preflight_pair.sha256, root_authorization_sha256=auth_pair.sha256,
        device_profile=dict(authorization["device_profile"]),
    )


def run_v4_score_lifecycle(
    *, artifact: v1p.ArtifactSink, identity: score.V4ScoreIdentity, capability: V4ExecutionCapability,
    backend: V4PhysicalMatchedScoreBackend,
    assets_factory: Callable[[], Mapping[str, Sequence[v1p.PhysicalAsset]]],
    final_reverify: Callable[[], Mapping[str, object]],
) -> Mapping[str, object]:
    """Transactional V4 lifecycle with historical bridge diagnostic isolation."""
    capability.verify(identity)
    if tuple(artifact.topology) != V4_SCORE_TOPOLOGY:
        raise V4PhysicalScoreError("V4 lifecycle artifact topology drift")
    hashes: dict[str, str] = {}
    stage = "preflight"
    attempt_written = False
    try:
        preflight = _preflight_payload(identity)
        authorization = _authorization_payload(identity, capability)
        attempt = _attempt_payload(identity)
        hashes.update(_publish_group(artifact, {
            "preflight.json": _json(preflight), "authorization.json": _json(authorization), "attempt.json": _json(attempt),
        }))
        attempt_written = True
        stage = "input_authority"
        assets = assets_factory()
        if not isinstance(assets, Mapping) or set(assets) != set(score.SURFACES):
            raise V4PhysicalScoreError("V4 deferred asset factory topology drift")
        stage = "prepare"
        backend.prepare(identity=identity)
        stage = "input"
        input_authority = backend.materialize_inputs(identity=identity, assets=assets)
        input_payload = input_authority.payload(identity=identity.base_identity)
        hashes.update(_publish_group(artifact, {"input_authority.json": _json(input_payload)}))
        stage = "score"
        evidence = [backend.score_cell_v4(cell=cell, input_authority_sha256=hashes["input_authority.json"])
                    for cell in score.score_matrix()]
        backend.reverify_after_v4_forwards()
        bridge = backend.historical_bridge_payload(input_authority_sha256=hashes["input_authority.json"])
        hashes.update(_publish_group(artifact, {"historical_bridge.json": _json(bridge)}))
        score_payload = score.build_score_payload(
            identity=identity, input_payload=input_payload, evidence=evidence, historical_bridge=bridge,
        )
        score.validate_score_payload(score_payload, identity=identity)
        stage = "terminal"
        final_closure = final_reverify()
        if final_closure != identity.closure.payload():
            raise V4PhysicalScoreError("V4 final closure/provenance drift")
        score_body = _json(score_payload)
        hashes["score.json"] = _digest(score_body)
        terminal = _terminal_payload(identity=identity, hashes=hashes, score_payload=score_payload)
        terminal_body = _json(terminal)
        validate_v4_terminal(
            terminal, identity=identity,
            expected_hashes={**hashes, "score.json": _digest(score_body)}, score_payload=score_payload,
        )
        hashes.update(_publish_group(artifact, {"score.json": score_body, "terminal.json": terminal_body}))
        if artifact.has_name("failure.json"):
            raise V4PhysicalScoreError("V4 success terminal coexists with failure")
        final = artifact.reload_json("terminal.json", hashes["terminal.json"])
        if not isinstance(final, Mapping) or final != terminal:
            raise V4PhysicalScoreError("V4 terminal durable reload drift")
        loaded_score = artifact.reload_json("score.json", hashes["score.json"])
        if not isinstance(loaded_score, Mapping):
            raise V4PhysicalScoreError("V4 score durable reload schema drift")
        score.validate_score_payload(loaded_score, identity=identity)
        validate_v4_terminal(final, identity=identity, expected_hashes=hashes, score_payload=loaded_score)
        return final
    except BaseException as error:
        if attempt_written and not artifact.has_name("terminal.json") and not artifact.has_name("failure.json"):
            try:
                _publish_group(artifact, {"failure.json": _json(_failure_payload(
                    identity=identity, stage=stage, error=error, hashes=hashes,
                ))})
            except BaseException:
                pass
        raise
    finally:
        backend.close()


def execute_authorized_v4(
    root: Path, *, capability: V4ExecutionCapability | None, device_profile: Mapping[str, object] | None = None,
    artifact: v1p.ArtifactSink | None = None, backend: V4PhysicalMatchedScoreBackend | None = None,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Production-only entry point; callers must supply an in-process capability."""
    v3p.validate_launch_environment(environ)
    if not isinstance(capability, V4ExecutionCapability):
        raise V4PhysicalScoreError("V4 requires an in-process V4 execution capability")
    root = Path(root).absolute()
    graph = validate_failed_v3_graph(root)
    # V3's own failed V2 predecessor remains part of the composed V3 closure.
    failed_v2 = v3p.validate_failed_v2_graph(root)
    v3_closure = v3score.implementation_closure(root, failed_v2_graph=failed_v2.payload())
    provenance = v1p.load_verified_provenance(root)
    base_identity = v3p.score_identity_from_provenance_v3(
        provenance=provenance, closure=v3_closure, failed_v2_graph=failed_v2,
    )
    closure = score.implementation_closure(root, v3_base_closure=v3_closure.payload(), failed_v3_graph=graph.payload())
    identity = score.V4ScoreIdentity(
        base_identity=base_identity, closure=closure,
        v3_input_authority_sha256=graph.body_sha256["input_authority.json"],
    )
    durable_authority = load_v4_durable_authority(root, identity=identity, graph=graph)
    if (
        capability.official_preflight_sha256 != durable_authority.official_preflight_sha256
        or capability.root_authorization_sha256 != durable_authority.root_authorization_sha256
    ):
        raise V4PhysicalScoreError("V4 capability durable authority binding drift")
    capability.verify(identity)
    selected = v1p.plan.validate_compatible_device_profile(
        dict(device_profile) if device_profile is not None else dict(capability.device_profile)
    )
    if selected != dict(capability.device_profile):
        raise V4PhysicalScoreError("V4 requested device differs from capability")
    if selected != dict(durable_authority.device_profile):
        raise V4PhysicalScoreError("V4 requested device differs from durable authorization")
    if artifact is None:
        artifact = reserve_v4_score_root(root, graph=graph)
    if getattr(artifact, "directory", None) != root / score.SCORE_ROOT_RELATIVE:
        raise V4PhysicalScoreError("V4 artifact is not the canonical V4 score root")
    if backend is None:
        backend = V4PhysicalMatchedScoreBackend(
            root=root, runtime=V4PhysicalRuntime(root=root, device_profile=selected),
            device_profile=selected, provenance_loader=v1p.load_verified_provenance, v3_graph=graph,
        )

    def derive_assets() -> Mapping[str, Sequence[v1p.PhysicalAsset]]:
        reader = v1p._load_exact_module(  # type: ignore[attr-defined]
            "_pmc_d_v4_input_descriptor_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        rows = reader.derive_fixed_input_assets(
            root, within_roster=identity.base_identity.within_roster,
            external_roster=identity.base_identity.external_roster,
        )
        return v1p._assets_from_rows(rows)  # type: ignore[attr-defined]

    def final_reverify() -> Mapping[str, object]:
        if validate_failed_v3_graph(root).payload() != graph.payload():
            raise V4PhysicalScoreError("V3 failed predecessor changed during V4 score")
        if v1p.load_verified_provenance(root).payload() != provenance.payload():
            raise V4PhysicalScoreError("PMC/sealed producer provenance changed during V4 score")
        current_v2 = v3p.validate_failed_v2_graph(root)
        if current_v2.payload() != failed_v2.payload():
            raise V4PhysicalScoreError("V3's V2 predecessor changed during V4 score")
        current_v3 = v3score.implementation_closure(root, failed_v2_graph=failed_v2.payload())
        current = score.implementation_closure(root, v3_base_closure=current_v3.payload(), failed_v3_graph=graph.payload())
        if current.payload() != identity.closure.payload():
            raise V4PhysicalScoreError("V4 implementation closure changed during score")
        return current.payload()

    return run_v4_score_lifecycle(
        artifact=artifact, identity=identity, capability=capability, backend=backend,
        assets_factory=derive_assets, final_reverify=final_reverify,
    )
