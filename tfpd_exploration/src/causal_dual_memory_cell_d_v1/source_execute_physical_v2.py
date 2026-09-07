"""Physical V2 theta-authority seam for CDM-D Source Execution.

The frozen V1 parser, strict Cell-D executor, B8 construction, and source
access ordering are inherited unchanged.  The sole override is the raw-T4
hash domain used while validating the sealed theta artifact.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from . import source_execute as v1
from . import source_execute_physical as v1_physical
from . import source_execute_v2 as v2


class SourceExecutionPhysicalV2Error(v2.SourceExecutionV2Error):
    """Fail closed for the one reviewed V2 physical seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceExecutionPhysicalV2Error(message)


def _sha256_array_bytes(value: Any, *, dtype: Any) -> str:
    """Digest raw contiguous bytes only; no dtype/shape prefix is permitted."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


class V2ThetaStrict27SessionProvider(v1_physical.ConcreteStrict27SessionProvider):
    """V1 direct-child parser with only the sealed raw-T4 SHA law repaired."""

    def __init__(self, *, root: Path, source_data: v1.StrictSourceDataRootCapability) -> None:
        super().__init__(root=Path(root), source_data=source_data)
        self._v2_theta_raw_proofs: dict[str, dict[str, object]] = {}

    def _theta_valid_mask(
        self, *, root: Path, session: str, raw_t4: Any, modules: Mapping[str, Any],
    ) -> tuple[Any, Mapping[str, object]]:
        """Validate all sealed theta facts using float32 contiguous raw bytes.

        V1's implementation converts this raw T4 to float64 and calls a
        tensor digest that prefixes shape/dtype.  That is intentionally never
        reached here.  The numerical raw tensor passed onward by the inherited
        parser remains untouched; the float32 conversion exists only inside
        this authority comparator.
        """
        import io
        import numpy as np

        torch = modules["torch"]
        cdm_source_adapter = modules["cdm_source_adapter"]
        from torch.nn.parameter import UninitializedParameter
        from torch.torch_version import TorchVersion
        from mc_maze.unit_side_features import MODULATION_EPS

        bound = v1.descriptor_read_fixed_asset(
            Path(root),
            next(asset for asset in v1.FIXED_ASSETS if asset.label == "theta_artifact"),
        )
        try:
            with torch.serialization.safe_globals([UninitializedParameter, TorchVersion]):
                payload = torch.load(io.BytesIO(bound.body), map_location="cpu", weights_only=True)
        except Exception as error:
            raise SourceExecutionPhysicalV2Error("V2 theta authority weights-only load failed") from error
        _require(
            isinstance(payload, Mapping) and payload.get("kind") == "tfpd_sparsification_theta_authority_v1",
            "V2 theta authority payload schema drift",
        )
        authority = payload.get("authority")
        _require(isinstance(authority, Mapping) and isinstance(authority.get(session), Mapping),
                 "V2 theta authority strict source session row missing")
        row = authority[session]
        _require(set(row) == {"theta", "valid", "n_units", "raw_t4_sha256"},
                 "V2 theta authority row topology drift")
        sealed_theta = row.get("theta")
        sealed_valid = row.get("valid")
        sealed_raw_sha = row.get("raw_t4_sha256")
        _require(
            torch.is_tensor(sealed_theta) and sealed_theta.ndim == 1 and sealed_theta.dtype == torch.float64
            and torch.is_tensor(sealed_valid) and sealed_valid.ndim == 1 and sealed_valid.dtype == torch.bool
            and isinstance(sealed_raw_sha, str) and len(sealed_raw_sha) == 64,
            "V2 theta authority tensor/digest type drift",
        )
        units = int(row.get("n_units", -1))
        _require(
            sealed_theta.numel() == units and sealed_valid.numel() == units,
            "V2 theta authority tensor/unit topology drift",
        )
        sealed_theta_array = np.ascontiguousarray(
            sealed_theta.detach().cpu().numpy(), dtype=np.float64,
        )
        sealed_valid_array = np.ascontiguousarray(
            sealed_valid.detach().cpu().numpy(), dtype=np.bool_,
        )
        # The one semantic V2 repair.  This helper uses only contiguous
        # float32 raw bytes for the raw SHA; it never calls V1's incompatible
        # tensor_digest or uses it as a fallback.  The inherited parser keeps
        # the original ordinary raw tensor after this validation.
        canonical_channels = np.arange(units, dtype=np.int64)
        recomputed_valid, proof = v2.validate_theta_raw_t4_authority(
            raw_t4,
            sealed_raw_t4_sha256=sealed_raw_sha,
            sealed_theta_float64=sealed_theta_array,
            sealed_valid_mask=sealed_valid_array,
            n_units=units,
            channel_ids=canonical_channels,
            session=session,
            modulation_eps=float(MODULATION_EPS),
        )
        # The V1 source-authority payload uses the accepted CDM core digest
        # domain for topology rows.  Preserve that existing relation while
        # keeping the raw-T4 SHA itself in the sealed byte domain above.
        proof["valid_mask_sha256"] = cdm_source_adapter.core.array_digest(recomputed_valid)
        proof["sealed_valid_mask_sha256"] = cdm_source_adapter.core.array_digest(sealed_valid_array)
        proof["canonical_unit_order_sha256"] = cdm_source_adapter.core.array_digest(canonical_channels)
        self._v2_theta_raw_proofs[session] = proof
        return recomputed_valid.copy(), {
            "theta_artifact_sha256": bound.asset.sha256,
            "raw_t4_sha256": sealed_raw_sha,
            "valid_mask_sha256": cdm_source_adapter.core.array_digest(recomputed_valid),
            "invalid_unit_count": v1.KNOWN_THETA_INVALID_UNIT_COUNTS.get(session, 0),
        }

    def _materialize_session(self, *, session: str, modules: Mapping[str, Any]) -> v1_physical.Strict27SessionMaterial:
        """Reuse V1's full parser, then bind its existing channel-order proof."""
        material = super()._materialize_session(session=session, modules=modules)
        import numpy as np

        core, _source_adapter, _source_audit = v1_physical._load_runtime_primitives()
        channels = np.ascontiguousarray(material.source_theta_topology.channel_ids, dtype=np.int64)
        canonical = np.arange(channels.size, dtype=np.int64)
        canonical_sha = core.array_digest(canonical)
        _require(
            np.array_equal(channels, canonical)
            and material.raw_t4_channel_order_sha256 == canonical_sha,
            "V2 raw-T4 canonical unit-order proof drift",
        )
        proof = dict(self._v2_theta_raw_proofs.get(session, {}))
        _require(proof.get("raw_t4_shape") == [int(channels.size), 4],
                 "V2 raw-T4 proof/session unit count drift")
        proof["canonical_unit_order_exact"] = True
        proof["canonical_unit_order_sha256"] = canonical_sha
        self._v2_theta_raw_proofs[session] = proof
        return material

    def theta_raw_proofs(self, *, roster: tuple[str, ...]) -> list[dict[str, object]]:
        _require(tuple(self._v2_theta_raw_proofs) == tuple(roster),
                 "V2 theta raw proof roster/order drift")
        return [dict(self._v2_theta_raw_proofs[session]) for session in roster]


class PhysicalSourceExecutionBackendV2(v1_physical.PhysicalSourceExecutionBackend):
    """V1 physical backend with a V2 provider and V2 theta proof disclosure."""

    def source_authority(
        self, runtime: Any, *, identity: v1.SourceExecutionIdentity, flags: v1.RuntimeFlags,
    ) -> Mapping[str, object]:
        value = dict(super().source_authority(runtime, identity=identity, flags=flags))
        _require(isinstance(self.provider, V2ThetaStrict27SessionProvider),
                 "V2 physical backend provider seam drift")
        value["v2_theta_raw_proofs"] = self.provider.theta_raw_proofs(
            roster=tuple(runtime.physical_roster),
        )
        return value


def build_reviewed_physical_backend(
    *, root: Path, source_data: v1.StrictSourceDataRootCapability,
    selected_device: Mapping[str, object],
) -> PhysicalSourceExecutionBackendV2:
    """Construct V2 by subclassing only V1's theta-provider seam."""
    v1_physical._validate_pmc_device_profile(selected_device)
    _require(isinstance(source_data, v1.StrictSourceDataRootCapability),
             "reviewed V2 physical factory needs inherited strict source capability")
    return PhysicalSourceExecutionBackendV2(
        provider=V2ThetaStrict27SessionProvider(root=Path(root), source_data=source_data),
        executor=v1_physical.ConcreteCellDFourGroupExecutor(root=Path(root)),
    )
