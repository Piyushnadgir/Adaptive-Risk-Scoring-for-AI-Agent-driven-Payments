"""
Unit tests for Adaptive Risk Scoring Proof-of-Concept modules.
Verifies data synthesis, domain scorers, weighted fusion, and decision thresholds.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure scripts and src are on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.fraud_poc import (
    make_clean,
    inject_attacks,
    kyc_score,
    api_score,
    txn_score,
    graph_score,
    fuse,
    decide,
    recall_at_fpr,
)


@pytest.fixture
def sample_dataset():
    rng = np.random.default_rng(42)
    clean = make_clean(rng, n_users=50, events_per_user=10)
    attack = inject_attacks(clean, rng, n_attacks=4)
    df = pd.concat([clean, attack], ignore_index=True)
    return df


def test_data_generation_schema(sample_dataset):
    df = sample_dataset
    required_cols = [
        "user_id", "identity_id", "identity_accounts", "kyc_age_days",
        "api_calls_per_min", "new_endpoint", "instruction_from_file",
        "amount", "user_avg_amount", "dest_account", "is_attack"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 0
    assert (df.is_attack == 1).sum() > 0
    assert (df.is_attack == 0).sum() > 0


def test_scorers_range(sample_dataset):
    df = sample_dataset
    for scorer in [kyc_score, api_score, txn_score, graph_score]:
        scores = scorer(df)
        assert len(scores) == len(df)
        assert np.all(scores >= 0.0), f"{scorer.__name__} has negative score"
        assert np.all(scores <= 100.0), f"{scorer.__name__} exceeds 100.0"


def test_fusion_calculation(sample_dataset):
    df = sample_dataset
    scores = {
        "kyc": kyc_score(df),
        "api": api_score(df),
        "txn": txn_score(df),
        "graph": graph_score(df),
    }
    weights = {"kyc": 0.25, "api": 0.25, "txn": 0.25, "graph": 0.25}
    fused = fuse(scores, weights)
    assert len(fused) == len(df)
    assert np.all(fused >= 0.0)
    assert np.all(fused <= 100.0)


def test_adaptive_decisions():
    scores = np.array([5.0, 25.0, 45.0, 75.0])
    thresholds = {"allow": 20.0, "restrict": 35.0, "review": 55.0}
    decisions = decide(scores, thresholds)
    assert list(decisions) == ["ALLOW", "RESTRICT", "REVIEW", "REVOKE"]


def test_recall_at_fpr():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    scores = np.array([10, 15, 20, 25, 30, 80, 85, 90])
    rec = recall_at_fpr(y, scores, fpr=0.20)
    assert rec == 1.0

