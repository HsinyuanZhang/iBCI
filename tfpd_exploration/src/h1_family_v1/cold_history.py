"""Pure deterministic H1 cold-history treatment; no cache/model imports."""
from __future__ import annotations
import hashlib
import torch

W, UNITS = 700, 176
DOMAIN = b"h1_selected_ema_cold_history_v1"


def _seed(seed: int, epoch: int, batch_id: int) -> int:
    if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (seed, epoch, batch_id)):
        raise ValueError("seed/epoch/batch_id must be nonnegative integers")
    return int.from_bytes(hashlib.sha256(DOMAIN + f"|{seed}|{epoch}|{batch_id}".encode()).digest()[:8], "little")


def apply_cold_history(x: torch.Tensor, *, seed: int, epoch: int, batch_id: int,
                       probability: float = .5) -> tuple[torch.Tensor, torch.Tensor]:
    """Clone `[B,700,176]`; selected rows retain uniformly 1..699 right bins.

    p=0 is an exact clone control.  The treatment zeros only missing left
    history; it never changes the current bin or targets (which are external).
    """
    _seed(seed, epoch, batch_id)
    if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or x.ndim != 3
            or tuple(x.shape[1:]) != (W, UNITS) or x.shape[0] == 0 or not bool(torch.isfinite(x).all())):
        raise ValueError("expected finite FP32 nonempty [B,700,176]")
    if probability not in (0., .5): raise ValueError("probability must be 0 or .5")
    out = x.clone()
    lengths = torch.full((len(x),), W, dtype=torch.int64, device=x.device)
    if probability == 0.: return out, lengths
    generator = torch.Generator(device="cpu").manual_seed(_seed(seed, epoch, batch_id))
    chosen = torch.rand((len(x),), generator=generator) < .5
    if bool(chosen.any()): lengths[chosen.to(x.device)] = torch.randint(1, W, (int(chosen.sum()),), generator=generator, dtype=torch.int64).to(x.device)
    missing = torch.arange(W, device=x.device).view(1, W) < (W - lengths).view(-1, 1)
    out.masked_fill_(missing.unsqueeze(-1), 0.)
    if not torch.equal(out[:, -1], x[:, -1]): raise RuntimeError("cold history changed current bin")
    return out, lengths
