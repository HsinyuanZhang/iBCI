from __future__ import annotations

import torch

from mc_maze.misleading_identity_swap import (
    apply_activity_identity_swap,
    build_partial_matched_involution,
    session_epoch_seed,
)


def _features() -> torch.Tensor:
    # Four tight pairs in standardized [a,c,m,b] space.
    return torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0, 2.0],
        [2.1, 2.0, 2.0, 2.0],
        [4.0, 4.0, 4.0, 4.0],
        [4.1, 4.0, 4.0, 4.0],
        [6.0, 6.0, 6.0, 6.0],
        [6.1, 6.0, 6.0, 6.0],
    ])


def test_session_epoch_seed_is_stable_and_changes_on_either_axis() -> None:
    assert session_epoch_seed("s1", 3) == session_epoch_seed("s1", 3)
    assert session_epoch_seed("s1", 3) != session_epoch_seed("s1", 4)
    assert session_epoch_seed("s1", 3) != session_epoch_seed("s2", 3)


def test_partial_mapping_is_deterministic_involution_with_fixed_points() -> None:
    swap = build_partial_matched_involution(_features(), fraction=0.5, seed=7)
    same = build_partial_matched_involution(_features(), fraction=0.5, seed=7)
    assert torch.equal(swap.permutation, same.permutation)
    assert swap.selected_count == 4
    assert swap.selected.sum().item() == 4
    assert torch.equal(swap.permutation[swap.permutation], torch.arange(8))
    assert torch.equal(swap.permutation == torch.arange(8), ~swap.selected)


def test_swap_changes_only_selected_activity_rows() -> None:
    swap = build_partial_matched_involution(_features(), fraction=0.5, seed=9)
    mean = torch.arange(2 * 8 * 3).reshape(2, 8, 3)
    observed = apply_activity_identity_swap(mean, swap)
    assert torch.equal(observed, mean.index_select(1, swap.permutation))
    assert torch.equal(observed[:, ~swap.selected], mean[:, ~swap.selected])


def test_t4_and_z4_siblings_can_share_the_exact_hidden_mapping() -> None:
    match = _features()
    swap_t4 = build_partial_matched_involution(match, fraction=0.5, seed=123)
    # Model-visible Z4 can be all zero; donor selection must use a separately
    # carried, unmasked matching descriptor with the same bytes as T4.
    visible_z4 = torch.zeros_like(match)
    assert torch.count_nonzero(visible_z4).item() == 0
    swap_z4 = build_partial_matched_involution(match.clone(), fraction=0.5, seed=123)
    assert torch.equal(swap_t4.permutation, swap_z4.permutation)
