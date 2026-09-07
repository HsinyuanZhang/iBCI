from pathlib import Path

import numpy as np
import pytest
import torch
from falcon_challenge.config import FalconConfig, FalconTask

from tfpd_exploration.src.family_runtime_v1.m2_falcon import FamilyM2FalconDecoder
from tfpd_exploration.src.family_runtime_v1.m2_cold_start import ColdStartLinearConvM2Decoder
from tfpd_exploration.tests.test_m2_grouped_value_runtime import P
from tfpd_exploration.tests.test_m2_runtime_v3 import TAGS


def test_m2_falcon_real_tag_formats_padded_lifecycle_and_resize():
    torch.set_num_threads(1)
    config = FalconConfig(task=FalconTask.m2)
    adapter = FamilyM2FalconDecoder(config, P, batch_size=7)
    direct = ColdStartLinearConvM2Decoder(P, batch_size=7)
    canonical = TAGS[:2]
    handles = [Path("sub-MonkeyN-held-in-minival_ses-2020-10-19-Run1_behavior+ecephys.nwb"),
               Path("sub-MonkeyNRun2_20201019_held_in_eval.nwb")]
    assert [config.hash_dataset(x.stem) for x in handles] == canonical
    adapter.reset(handles); direct.reset(canonical)
    source = np.random.default_rng(702).poisson(.3, (53, 2, 96)).astype(np.float32)
    for index, value in enumerate(source):
        padded = np.zeros((7, 96), dtype=np.float32); padded[:2] = value
        if index % 7 == 0:
            adapter.observe(padded); direct.observe(value)
        else:
            public = padded if index % 2 else value
            if index % 3 == 0:
                public = public.astype(np.uint8)
            elif index % 3 == 1:
                public = public.astype(np.float64)
            got = adapter.predict(public)
            np.testing.assert_allclose(got, direct.predict(value), rtol=1e-5, atol=1e-5)
            assert got.shape == (7, 2) and got.flags.owndata and got.flags.c_contiguous
            assert np.all(got[2:] == 0)
        before = adapter.runtime._engine.raw.clone()
        adapter.on_done(np.ones(7, dtype=bool))
        assert torch.equal(before, adapter.runtime._engine.raw)
    with pytest.raises(Exception, match="absent|unknown"):
        adapter.reset(["Run1_20990101"])
    np.testing.assert_allclose(adapter.predict(source[0]), direct.predict(source[0]), rtol=1e-5, atol=1e-5)
    adapter.set_batch_size(1)
    with pytest.raises(Exception, match="reset"):
        adapter.predict(source[0, :1])
    adapter.reset(canonical[:1])
    assert adapter.predict(source[0, :1]).shape == (1, 2)


def test_m2_falcon_input_guards_do_not_advance_history():
    torch.set_num_threads(1)
    adapter = FamilyM2FalconDecoder(FalconConfig(task=FalconTask.m2), P, batch_size=7)
    adapter.reset(TAGS[:1])
    before = adapter.runtime._engine.raw.clone()
    bad_padding = np.zeros((7, 96), dtype=np.float32); bad_padding[-1, 0] = 1
    bad_nan = np.zeros((1, 96), dtype=np.float32); bad_nan[0, 0] = np.nan
    for value in (bad_padding, bad_nan, np.zeros((1, 96), dtype=np.complex64),
                  np.zeros((1, 192), dtype=np.float32)):
        with pytest.raises(Exception):
            adapter.predict(value)
        assert torch.equal(before, adapter.runtime._engine.raw)
    noncontiguous = np.zeros((1, 192), dtype=np.uint8)[:, ::2]
    assert adapter.predict(noncontiguous).shape == (7, 2)
    for size in (0, 8, 1.5):
        with pytest.raises(Exception): adapter.set_batch_size(size)
    with pytest.raises(Exception, match="m2"):
        FamilyM2FalconDecoder(FalconConfig(task=FalconTask.m1), P, batch_size=1)
