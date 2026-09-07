"""Focused Stage-0 gate tests, mirroring the frozen contract one gate per test."""

from __future__ import annotations

import pytest
import torch

from src.tfpd.bilinear_readin import BilinearTaskFrameDecoder
from src.tfpd.gates import (
    build_models,
    gate1_permutation_invariance,
    gate2_variable_n,
    gate3_finite_live_gradients,
    gate4_query_support_separation,
    gate5_attainable_carrier_effect,
)
from src.tfpd.population_vector import LearnedPopulationVectorDecoder
from src.tfpd.synth import generate_session


@pytest.fixture(scope="module")
def models():
    return build_models(seed=42)


@pytest.mark.parametrize("model_name", ["bilinear", "population_vector"])
class TestAllGates:
    def test_g1_permutation_invariance(self, models, model_name):
        passed, detail = gate1_permutation_invariance(models[model_name], seed=1)
        assert passed, detail

    def test_g2_variable_n(self, models, model_name):
        passed, detail = gate2_variable_n(models[model_name], seed=2)
        assert passed, detail

    def test_g3_finite_live_gradients(self, model_name):
        passed, detail = gate3_finite_live_gradients(model_name, seed=3)
        assert passed, detail

    def test_g4_query_support_separation(self, models, model_name):
        passed, detail = gate4_query_support_separation(models[model_name], seed=4)
        assert passed, detail

    def test_g5_attainable_carrier_effect(self, model_name):
        passed, detail = gate5_attainable_carrier_effect(model_name, seed=42)
        assert passed, detail


def test_carrier_shape_validation(models):
    session = generate_session(seed=5, num_units=32)
    with pytest.raises(ValueError):
        models["bilinear"](session.counts.unsqueeze(0), session.carrier.unsqueeze(0)[:, :16])
