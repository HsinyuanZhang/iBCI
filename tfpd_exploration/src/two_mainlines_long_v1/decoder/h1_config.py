"""H1 temporal Transformer widths. PE must cover W=700; do not copy M2 max_len 256."""

from __future__ import annotations

from dataclasses import dataclass


class H1DecoderError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise H1DecoderError(message)


@dataclass(frozen=True)
class H1TemporalConfig:
    conv_channels: int = 16
    conv_kernel: int = 5
    e0_dim: int = 700  # C2 fused identity [N, 700]
    hc_dim: int = 4
    set_dim: int = 256
    slots: int = 8
    heads: int = 8
    layers: int = 4
    temporal_width: int = 256
    ffn: int = 512
    readout_hidden: int = 128
    out_dim: int = 7
    window: int = 700
    n_units: int = 176
    unit_dropout: float = 0.10
    pe_max_len: int = 700
    route_key_dim: int = 32
    prediction_divisor: float = 20.0

    @property
    def token_in(self) -> int:
        return self.conv_channels + self.e0_dim + self.hc_dim

    @property
    def local_dim(self) -> int:
        return self.conv_channels


H1_TEMPORAL = H1TemporalConfig()

FRONTEND_RNG_DOMAIN = 0
TEMPORAL_RNG_DOMAIN = 1_000_003
ROUTING_RNG_DOMAIN = 7_700_042

C2_EPOCH15_PATH = (
    "/home/xinyuan/Work_host/ibci_c3_film/tfpd_exploration/"
    "h1_series_20260830/artifacts/c2_references/c2_epoch_015.ckpt"
)
C2_CKPT_SHA256 = "ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215"

GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
