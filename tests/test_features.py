import shutil

import pytest

from src import llvm_pipeline as lp
from src.features import FEATURES, extract_features, func_properties, opcode_histogram
from tests.test_metrics import LL

needs_llvm = pytest.mark.skipif(not shutil.which("opt"), reason="LLVM not installed")


def test_hand_features_without_llvm(tmp_path):
    f = tmp_path / "x.ll"
    f.write_text(LL)
    feats = extract_features(f, None)
    assert set(FEATURES) <= set(feats)
    assert feats["loop_method"] == "cfg_back_edges" and feats["loop_count"] == 1
    assert feats["functions"] == 1 and feats["returns"] == 2
    assert feats["load_frac"] == pytest.approx(1 / 13)
    assert feats["cyclomatic"] == feats["cfg_edges"] - feats["basic_blocks"] + 2


def test_opcode_histogram_sums_to_one(tmp_path):
    f = tmp_path / "x.ll"
    f.write_text(LL)
    h = opcode_histogram(f)
    assert sum(h.values()) == pytest.approx(1.0)
    assert h["op_add"] == pytest.approx(2 / 13)


@needs_llvm
def test_llvm_features(tmp_path):
    f = tmp_path / "x.ll"
    f.write_text(LL)
    tc = lp.Toolchain()
    feats = extract_features(f, tc)
    assert feats["loop_method"] == "llvm_loopinfo" and feats["loop_count"] == 1 and feats["loop_depth"] == 1
    fp = func_properties(f, tc)
    assert fp["fp_BasicBlockCount"] == 5 and fp["fp_MaxLoopDepth"] == 1
