from pathlib import Path
import importlib.util
import numpy as np
import torch

P = Path(__file__).resolve().parents[1] / "scripts" / "activity_data.py"
S = importlib.util.spec_from_file_location("activity_data_test", P); activity = importlib.util.module_from_spec(S); S.loader.exec_module(activity)

def test_m2_real_geometry_and_support_query_separation():
    d = activity.load_m2_data()
    assert set(d) >= {"train","validation","evaluation","metadata"}
    assert len(d["train"]) == len(d["validation"]) == 7
    for s, item in d["train"].items():
        assert item["activity"].shape == (33,100,96)
        assert item["activity"].dtype == np.float32
        assert np.all(item["starts"] >= item["support_provenance"]["query_boundary_padded"])
        assert item["support_provenance"]["trial_ids"] == list(range(33))
        assert d["validation"][s]["activity"] is not item["Y"]

def test_m2_windows_label_plane_and_padding_mask():
    item = next(iter(activity.load_m2_data()["train"].values()))
    x,y,v = activity.windows(item, np.array([0,1]), 50, "cpu", 5.)
    assert x.shape == (2,50,96) and y.shape == (2,2) and v.dtype == torch.bool
    assert torch.equal(y, torch.as_tensor(np.asarray(item["Y"][:2])*5.))

def test_m2_ext6_raw_support_is_separate_from_query_files():
    d=activity.load_m2_data(include_eval=True)
    assert len(d["evaluation"]) == 6
    for x in d["evaluation"].values():
        assert x["activity"].shape == (33,100,96)
        assert x["support_provenance"]["historical_allqueries_overlap_possible"]
        assert np.all(x["starts"][x["post_support_indices"]] >= x["support_provenance"]["strict_post33_boundary_padded"])

def test_m1_real_m10_geometry_and_post_support_filter():
    d=activity.load_m1_data()
    assert len(d["train"]) == 4 and d["batches"] is None
    for item in d["train"].values():
        assert item["activity"].shape == (10,1024,64)
        assert np.all(item["starts"] >= item["support_provenance"]["query_boundary_padded"])

def test_h1_real_m3_resampled_geometry_and_minival_support_reuse():
    d=activity.load_h1_data()
    assert len(d["train"]) == len(d["validation"]) == 13
    for s,item in d["train"].items():
        assert item["activity"].shape == (3,1024,176)
        assert item["support_provenance"]["strict_query_disjoint"]
        assert d["validation"][s]["activity"] is item["activity"]

def test_all_public_eval_faces_load():
    assert len(activity.load_m1_data(include_eval=True)["evaluation"]) == 3
    assert len(activity.load_h1_data(include_eval=True)["evaluation"]) == 14
