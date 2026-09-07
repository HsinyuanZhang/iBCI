import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_new_carrier_is_persisted_and_source_only_in_design():
    text=(ROOT/'src/m1_optimized_v2/bank.py').read_text()
    assert 'SOURCE_SESSIONS' in text and 'target_path_resolved' in text and 'fit_once' in text
    assert 'OLD_STAGE0_D0_SHA256' in text and 'collides with sealed Stage-0' in text
def test_new_loader_declares_truthful_fold_zero_not_old_fold_one_epoch_narrative():
    text=(ROOT/'src/m1_optimized_v2/data.py').read_text()
    assert 'outer_fold":0' in text and 'fixed epoch, never target query' in text
    assert 'glob(' not in text and 'rglob(' not in text
def test_b3_loader_requires_setup_and_strict_encoder_keys():
    text=(ROOT/'src/m1_optimized_v2/calibration.py').read_text()
    assert 'lit.setup("fit")' in text and 'load_state_dict(incoming,strict=True)' in text
    assert 'sfix_decoder_weights_copied' not in text
def test_common_temporal_binding_is_required():
    text=(ROOT/'src/m1_optimized_v2/model.py').read_text()
    assert 'current_query_v2' in text and 'isinstance(model.temporal, QueryTemporalStack)' in text

def test_train_has_no_outer_validation_or_selection_path():
    text=(ROOT/'src/m1_optimized_v2/train.py').read_text()
    assert 'target-unread source loader' in text
    assert 'no target session or terminal evaluation' in text
