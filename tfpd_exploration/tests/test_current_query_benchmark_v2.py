"""A benchmark must not describe an untrained measured arm as trained."""
from pathlib import Path

import pytest

from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.benchmark_stream import (
    _validate_state_arguments,
)


def test_state_required_only_for_measured_arms():
    full = Path("full-trained.pt")
    query = Path("query-trained.pt")
    assert _validate_state_arguments(full, None, False, ["full", "full_exact"]) == {"full"}
    assert _validate_state_arguments(None, query, False, ["query_cached"]) == {"query"}
    assert _validate_state_arguments(full, query, False, ["full_exact", "query_cached"]) == {"full", "query"}
    with pytest.raises(ValueError, match="each measured arm"):
        _validate_state_arguments(full, None, False, ["full_exact", "query_cached"])
    with pytest.raises(ValueError, match="each measured arm"):
        _validate_state_arguments(None, None, False, ["full_exact"])


def test_initialization_is_explicit_and_cannot_mix_with_trained():
    assert _validate_state_arguments(None, None, True, ["full_exact", "query_cached"]) == {"full", "query"}
    with pytest.raises(ValueError, match="do not mix"):
        _validate_state_arguments(Path("full.pt"), None, True, ["full_exact"])
