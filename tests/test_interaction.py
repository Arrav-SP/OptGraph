import itertools

import numpy as np
import pandas as pd
import pytest

from src.interaction import UNDEFINED, classify, compute_interactions, interaction_row

THR = 0.005


def test_formula_matches_definition():
    r = interaction_row(100, 80, 90, 60, THR)
    assert r["delta_b_alone"] == pytest.approx(0.10)
    assert r["delta_b_after_a"] == pytest.approx(0.20)
    assert r["interaction_score"] == pytest.approx(0.10)
    # closed form: (m(A) + m(B) - m(BA) - m(P)) / m(P)
    assert r["interaction_score"] == pytest.approx((80 + 90 - 60 - 100) / 100)
    assert r["interaction_type"] == "SYNERGY"


@pytest.mark.parametrize("m_p,m_a,m_b,m_ab,m_ba", [(100, 80, 90, 60, 65), (50, 50, 40, 40, 30),
                                                   (7, 9, 3, 2, 8)])
def test_asymmetry_identity(m_p, m_a, m_b, m_ab, m_ba):
    """I(A,B,P) - I(B,A,P) = (m(A(B(P))) - m(B(A(P)))) / m(P)."""
    i_ab = interaction_row(m_p, m_a, m_b, m_ab, THR)["interaction_score"]
    i_ba = interaction_row(m_p, m_b, m_a, m_ba, THR)["interaction_score"]
    assert i_ab - i_ba == pytest.approx((m_ba - m_ab) / m_p)


def test_zero_baseline_is_undefined_not_a_number():
    r = interaction_row(0, 0, 2, 1, THR)
    assert r["interaction_score"] is None and r["delta_b_alone"] is None
    assert r["interaction_type"] == UNDEFINED
    assert r["interaction_abs"] == (0 - 1) - (0 - 2)


def test_classify_threshold():
    assert classify(0.006, THR) == "SYNERGY" and classify(-0.006, THR) == "ANTAGONISM"
    assert classify(0.005, THR) == "WEAK" and classify(None, THR) == UNDEFINED


def _meas(context_values):
    rows = []
    for ctx, (p, a, b, ab, ba) in context_values.items():
        base = {"program_id": "p", "category": "c", "context": ctx, "status": "SUCCESS"}
        rows += [{**base, "variant": "baseline", "kind": "baseline", "pass_a": "", "pass_b": "", "m": p},
                 {**base, "variant": "A", "kind": "single", "pass_a": "A", "pass_b": "", "m": a},
                 {**base, "variant": "B", "kind": "single", "pass_a": "B", "pass_b": "", "m": b},
                 {**base, "variant": "A_then_B", "kind": "pair", "pass_a": "A", "pass_b": "B", "m": ab},
                 {**base, "variant": "B_then_A", "kind": "pair", "pass_a": "B", "pass_b": "A", "m": ba}]
    return pd.DataFrame(rows)


def test_compute_interactions_per_context_uses_its_own_baseline():
    meas = _meas({"C0": (100, 80, 90, 60, 65), "C1": (50, 50, 40, 40, 30)})
    out = compute_interactions(meas, THR, ["m"]).set_index(["context", "pass_a"])
    assert out.loc[("C0", "A"), "interaction_score"] == pytest.approx(0.10)
    assert out.loc[("C1", "A"), "interaction_score"] == pytest.approx((50 + 40 - 40 - 50) / 50)
    assert len(out) == 4


def test_compute_interactions_skips_unmeasured_metric():
    meas = _meas({"C0": (100, 80, 90, 60, 65)})
    meas.loc[meas["variant"] == "B", "m"] = np.nan
    assert compute_interactions(meas, THR, ["m"]).empty
