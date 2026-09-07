"""V4 closure-loader seam for CDM-D independent-activity source execution.

V4 deliberately inherits V2's descriptor-safe theta/parser provider and V3's
independent-activity executor.  The only new behaviour is that the two
directly descriptor-executed runtime helpers are validated against the V4
closure, rather than attempting to reconstruct V1's frozen closure after the
accepted V3 core addition.  No parser, SWA loader, model-forward, or state
transition machinery is copied here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import source_execute as v1
from . import source_execute_physical as v1_physical
from . import source_execute_physical_v2 as v2_physical
from . import source_execute_physical_v3 as v3_physical
from . import source_execute_v4 as v4


class SourceExecutionPhysicalV4Error(v4.SourceExecutionV4Error):
    """Fail closed for the one reviewed V4 loader-injection seam."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceExecutionPhysicalV4Error(message)


class V4ThetaStrict27SessionProvider(v2_physical.V2ThetaStrict27SessionProvider):
    """V2's exact theta parser with only V4 runtime closure injection."""

    def _runtime_modules(self) -> Mapping[str, Any]:
        return v1_physical._runtime_imports(
            self.root,
            closure_builder=v4.execution_closure_payload,
        )


class V4IndependentActivityCellDFourGroupExecutor(v3_physical.IndependentActivityCellDFourGroupExecutor):
    """V3's independent-activity executor with only V4 SWA-loader closure."""

    def _runtime_modules(self, root: Path) -> Mapping[str, Any]:
        return v1_physical._runtime_imports(
            Path(root),
            closure_builder=v4.execution_closure_payload,
        )


class PhysicalSourceExecutionBackendV4(v3_physical.PhysicalSourceExecutionBackendV3):
    """Unchanged V3 physical route whose two runtime loaders use V4 bytes."""

    def __init__(
        self,
        *,
        provider: V4ThetaStrict27SessionProvider,
        executor: V4IndependentActivityCellDFourGroupExecutor,
    ) -> None:
        super().__init__(provider=provider, executor=executor)
        _require(
            type(provider) is V4ThetaStrict27SessionProvider
            and type(executor) is V4IndependentActivityCellDFourGroupExecutor,
            "V4 physical route requires exactly its two closure-injection seams",
        )


def build_reviewed_physical_backend(
    *,
    root: Path,
    source_data: v1.StrictSourceDataRootCapability,
    selected_device: Mapping[str, object],
) -> PhysicalSourceExecutionBackendV4:
    """Construct the sole V4 physical composition without performing I/O."""

    v1_physical._validate_pmc_device_profile(selected_device)
    _require(isinstance(source_data, v1.StrictSourceDataRootCapability),
             "reviewed V4 physical factory needs a typed strict source-data capability")
    return PhysicalSourceExecutionBackendV4(
        provider=V4ThetaStrict27SessionProvider(root=Path(root), source_data=source_data),
        executor=V4IndependentActivityCellDFourGroupExecutor(root=Path(root)),
    )
