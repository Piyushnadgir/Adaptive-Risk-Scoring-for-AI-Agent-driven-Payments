"""
Adaptive Risk Scoring for AI-Agent Payments - Proof of Concept (PoC)

Pipeline:
  Dynamic User Generation -> Multi-vector Attack Injection
  -> 4 Domain Scorers (KYC, API Velocity/Intent, Transaction, Graph AML)
  -> Weighted-Sum Fusion
  -> Adaptive Authority Decision (ALLOW / RESTRICT / REVIEW / REVOKE)
  -> Real-time Action Walkthrough & Population-Scale Evaluation

Usage:
  python scripts/fraud_poc.py                 # Dynamic run with random seed
  python scripts/fraud_poc.py --seed 42       # Reproducible run for benchmarking
  python scripts/fraud_poc.py --users 1500    # Custom population size
  python scripts/fraud_poc.py --no-demo       # Skip transaction walkthrough
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# -----------------------------------------------------------------------------
# 0. CONFIGURATION LOADER
# -----------------------------------------------------------------------------
def load_project_configs() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Loads weights and thresholds from configs/ if available, else defaults."""
    base_dir = Path(__file__).resolve().parent.parent
    weights_file = base_dir / "configs" / "weights.yaml"
    thresholds_file = base_dir / "configs" / "thresholds.yaml"

    weights = {"kyc": 0.25, "api": 0.25, "txn": 0.25, "graph": 0.25}
    thresholds = {"allow": 20.0, "restrict": 35.0, "review": 55.0}

    if HAS_YAML and weights_file.exists():
        try:
            with open(weights_file, "r") as f:
                data = yaml.safe_load(f).get("weights", {})
                if data:
                    w_api = float(data.get("api", 1.0) + data.get("intent", 1.0))
                    w_txn = float(data.get("transaction", 1.0))
                    w_graph = float(data.get("network", 1.0) + data.get("aml", 1.0))
                    w_kyc = float(data.get("behaviour", 1.0))
                    total = w_api + w_txn + w_graph + w_kyc
                    if total > 0:
                        weights = {
                            "kyc": round(w_kyc / total, 3),
                            "api": round(w_api / total, 3),
                            "txn": round(w_txn / total, 3),
                            "graph": round(w_graph / total, 3),
                        }
        except Exception:
            pass

    return weights, thresholds


# -----------------------------------------------------------------------------
# 1. DYNAMIC SYNTHETIC DATA GENERATOR
# -----------------------------------------------------------------------------
def make_clean(rng: np.random.Generator, n_users: int = 2000, events_per_user: int = 18) -> pd.DataFrame:
    """
    Generates a realistic population of legitimate users and agent actions.
    Uses dynamic spending profiles (retail, business, student) and behavioral variance.
    """
    rows = []
    n_merchants = max(500, n_users // 2)

    for u in range(n_users):
        # Diverse user profiles
        user_type = rng.choice(["retail", "business", "student"], p=[0.70, 0.15, 0.15])
        if user_type == "retail":
            avg_amount = float(rng.lognormal(mean=5.8, sigma=0.5))    # ~$330 avg
            n_acc = int(rng.choice([1, 2, 3], p=[0.88, 0.10, 0.02]))
            kyc_age = int(rng.integers(60, 3000))
        elif user_type == "business":
            avg_amount = float(rng.lognormal(mean=7.2, sigma=0.6))    # ~$1300 avg
            n_acc = int(rng.choice([1, 2, 3, 4], p=[0.50, 0.35, 0.12, 0.03]))
            kyc_age = int(rng.integers(180, 4000))
        else:  # student
            avg_amount = float(rng.lognormal(mean=4.5, sigma=0.4))    # ~$90 avg
            n_acc = 1
            kyc_age = int(rng.integers(30, 800))

        # Payee preferences
        n_payees = int(rng.integers(6, 14))
        user_merchants = [f"MERCHANT_{m}" for m in rng.integers(0, n_merchants, size=n_payees)]

        # Dynamic events per user with natural variance
        n_events = max(5, int(rng.normal(events_per_user, 3)))
        for _ in range(n_events):
            api_calls = max(1.0, float(rng.normal(4.5, 2.0)))
            new_ep = rng.random() < 0.03            # rare legitimate new endpoint
            file_inst = rng.random() < 0.015        # rare automated file prompt
            amt = float(rng.lognormal(np.log(avg_amount), 0.45))

            # Mostly known merchants; rare 1-off peer transfer
            if rng.random() < 0.94:
                dest = str(rng.choice(user_merchants))
            else:
                dest = f"PEER_{rng.integers(0, 1000)}"

            rows.append(dict(
                user_id=u,
                identity_id=f"USR_{u:04d}",
                identity_accounts=n_acc,
                kyc_age_days=kyc_age,
                api_calls_per_min=api_calls,
                new_endpoint=new_ep,
                instruction_from_file=file_inst,
                amount=amt,
                user_avg_amount=avg_amount,
                dest_account=dest,
                attack_type="CLEAN",
                is_attack=0
            ))
    return pd.DataFrame(rows)


def inject_attacks(clean_df: pd.DataFrame, rng: np.random.Generator, n_attacks: int = 16) -> pd.DataFrame:
    """
    Simulates realistic attacks matching the Capstone project scenarios:
      1. Prompt Injection (Scenario 2): Agent tricked by external instructions
      2. Mule Network Fan-out (Scenario 3): Coordinated fund disbursement
      3. Sub-limit Smurfing (Scenario 2/3): Structuring payments just below 10k threshold
      4. Agent Hijack / Rapid Drain (Scenario 4): Stolen agent credentials & velocity burst
    """
    rows = []
    base_id = int(clean_df.user_id.max()) + 1
    attack_scenarios = ["PROMPT_INJECTION", "MULE_FANOUT", "SUBLIMIT_SMURF", "AGENT_HIJACK"]
    shared_hub = f"MULE_HUB_{rng.integers(10, 99)}"

    for k in range(n_attacks):
        atk_id = base_id + k
        identity_id = f"ATK_{atk_id:04d}"
        scenario = attack_scenarios[k % len(attack_scenarios)]
        events = int(rng.integers(8, 16))
        base_avg = float(rng.lognormal(5.8, 0.5))

        if scenario == "PROMPT_INJECTION":
            # Injected instruction redirects payment to an unauthorized endpoint
            dest_acc = f"UNTRUSTED_DEST_{k}"
            for _ in range(events):
                rows.append(dict(
                    user_id=atk_id, identity_id=identity_id,
                    identity_accounts=int(rng.choice([1, 2])),   # normal KYC profile
                    kyc_age_days=int(rng.integers(100, 600)),
                    api_calls_per_min=max(1.0, float(rng.normal(8, 2))),
                    new_endpoint=True,
                    instruction_from_file=True,
                    amount=float(rng.uniform(7500, 9800)),
                    user_avg_amount=base_avg,
                    dest_account=dest_acc,
                    attack_type=scenario,
                    is_attack=1
                ))

        elif scenario == "MULE_FANOUT":
            # Coordinated distribution across disposable mule accounts
            mules = [f"MULE_{k}_{j}" for j in range(rng.integers(4, 7))]
            for _ in range(events):
                rows.append(dict(
                    user_id=atk_id, identity_id=identity_id,
                    identity_accounts=int(rng.choice([3, 4, 5])),
                    kyc_age_days=int(rng.integers(20, 150)),
                    api_calls_per_min=max(1.0, float(rng.normal(11, 3))),
                    new_endpoint=rng.random() < 0.4,
                    instruction_from_file=rng.random() < 0.3,
                    amount=float(rng.uniform(7000, 9800)),
                    user_avg_amount=base_avg * 1.5,
                    dest_account=str(rng.choice(mules)),
                    attack_type=scenario,
                    is_attack=1
                ))

        elif scenario == "SUBLIMIT_SMURF":
            # Structuring: amounts intentionally kept between $8,800 and $9,950
            dest_acc = shared_hub if rng.random() < 0.6 else f"MULE_SMURF_{k}"
            for _ in range(events):
                rows.append(dict(
                    user_id=atk_id, identity_id=identity_id,
                    identity_accounts=int(rng.choice([2, 3])),
                    kyc_age_days=int(rng.integers(40, 250)),
                    api_calls_per_min=max(1.0, float(rng.normal(8, 2.5))),
                    new_endpoint=rng.random() < 0.3,
                    instruction_from_file=rng.random() < 0.2,
                    amount=float(rng.uniform(8800, 9950)),
                    user_avg_amount=base_avg,
                    dest_account=dest_acc,
                    attack_type=scenario,
                    is_attack=1
                ))

        else:  # AGENT_HIJACK
            # Stolen agent credential performing rapid unauthorized exfiltration
            mules = [f"EXFIL_{k}_{j}" for j in range(3)]
            for _ in range(events):
                rows.append(dict(
                    user_id=atk_id, identity_id=identity_id,
                    identity_accounts=int(rng.choice([2, 3, 4])),
                    kyc_age_days=int(rng.integers(15, 120)),
                    api_calls_per_min=max(1.0, float(rng.normal(16, 4))),
                    new_endpoint=rng.random() < 0.7,
                    instruction_from_file=rng.random() < 0.4,
                    amount=float(rng.uniform(6000, 9500)),
                    user_avg_amount=base_avg,
                    dest_account=str(rng.choice(mules)),
                    attack_type=scenario,
                    is_attack=1
                ))

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 2. DOMAIN SCORERS (0 - 100 RISK SCALE)
# -----------------------------------------------------------------------------
def kyc_score(df: pd.DataFrame) -> np.ndarray:
    """Stage 4: Evaluates account proliferation and account age."""
    s = np.zeros(len(df))
    s += 35 * (df.identity_accounts > 2)
    s += 25 * (df.identity_accounts > 4)
    s += 25 * (df.kyc_age_days < 180)
    return np.minimum(s, 100.0)


def api_score(df: pd.DataFrame) -> np.ndarray:
    """Stages 2 & 6: Evaluates API call velocity, unauthorized endpoints, and untrusted instructions."""
    s = np.zeros(len(df))
    s += 30 * (df.api_calls_per_min > 12)
    s += 35 * df.new_endpoint
    s += 35 * df.instruction_from_file
    return np.minimum(s, 100.0)


def txn_score(df: pd.DataFrame) -> np.ndarray:
    """Stage 5: Evaluates transaction structuring (smurfing) and historical deviation."""
    s = np.zeros(len(df))
    s += 45 * df.amount.between(8000, 9999)               # Smurfing band ($8k-$10k)
    s += 35 * (df.amount > 3.5 * df.user_avg_amount)       # Deviation from baseline
    return np.minimum(s, 100.0)


def graph_score(df: pd.DataFrame) -> np.ndarray:
    """Stage 7: Graph and AML analysis on mule topologies and hub accounts."""
    ids_per_dest = df.groupby("dest_account").identity_id.transform("nunique")
    exclusive = df.assign(excl=(ids_per_dest == 1)).drop_duplicates(["identity_id", "dest_account"])
    n_excl = exclusive.groupby("identity_id").excl.sum()
    n_excl = df.identity_id.map(n_excl).fillna(0)

    s = np.zeros(len(df))
    s += 40 * (n_excl >= 3)
    s += 30 * (n_excl >= 5)

    # Hub detection: Non-merchant accounts receiving funds from multiple distinct identities
    is_non_merchant = ~df.dest_account.str.startswith("MERCHANT_")
    s += 35 * (is_non_merchant & (ids_per_dest >= 3))
    return np.minimum(s, 100.0)


# -----------------------------------------------------------------------------
# 3. FUSION AND ADAPTIVE AUTHORITY DECISION
# -----------------------------------------------------------------------------
def fuse(scores: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Stage 8: Decomposed risk score fusion via weighted linear combination."""
    if weights is None:
        weights = {"kyc": 0.25, "api": 0.25, "txn": 0.25, "graph": 0.25}
    return sum(weights[k] * scores[k] for k in ["kyc", "api", "txn", "graph"])


def decide(score: np.ndarray, thresholds: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Stage 9: Adaptive Authority Matrix
      - ALLOW:    Score < allow_below       (Action approved silently)
      - RESTRICT: Score < restrict_below    (Limits lowered, step-up MFA required)
      - REVIEW:   Score < review_below      (Suspended for Human-in-the-Loop review)
      - REVOKE:   Score >= review_below     (Agent credentials revoked immediately)
    """
    allow_th = 20.0
    restrict_th = 40.0
    review_th = 60.0
    if thresholds:
        allow_th = thresholds.get("allow", 20.0)
        restrict_th = thresholds.get("restrict", 40.0)
        review_th = thresholds.get("review", 60.0)

    conditions = [
        score < allow_th,
        score < restrict_th,
        score < review_th,
    ]
    choices = ["ALLOW", "RESTRICT", "REVIEW"]
    return np.select(conditions, choices, default="REVOKE")


# -----------------------------------------------------------------------------
# 4. EVALUATION METRICS
# -----------------------------------------------------------------------------
def recall_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float = 0.01) -> float:
    """Recall on attack events when false positive rate on clean events is strictly bounded."""
    clean_scores = score[y == 0]
    thr = np.quantile(clean_scores, 1.0 - fpr)
    return float((score[y == 1] > thr).mean())


# -----------------------------------------------------------------------------
# 5. LIVE DEMO WALKTHROUGH
# -----------------------------------------------------------------------------
def print_live_demo(df: pd.DataFrame, scores: Dict[str, np.ndarray], weights: Dict[str, float], thresholds: Dict[str, float]):
    """Demonstrates how the adaptive risk pipeline evaluates individual live agent actions."""
    print("\n" + "=" * 78)
    print(" LIVE AGENT TRANSACTION EVALUATION (Adaptive Authority Pipeline)")
    print("=" * 78)

    sample_cases = []
    # 1. Clean retail transaction
    clean_idx = df[(df.is_attack == 0) & (df.amount < 500)].index
    if len(clean_idx) > 0:
        sample_cases.append(("1. Legitimate Utility Payment", clean_idx[0]))

    # 2. Suspicious / Elevated spend
    elev_idx = df[(df.is_attack == 0) & (df.amount > df.user_avg_amount * 3)].index
    if len(elev_idx) > 0:
        sample_cases.append(("2. Elevated Spend (Clean User, High Amount)", elev_idx[0]))

    # 3. Prompt injection attack (Scenario 2)
    pi_idx = df[df.attack_type == "PROMPT_INJECTION"].index
    if len(pi_idx) > 0:
        sample_cases.append(("3. Injected Prompt Execution (Scenario 2)", pi_idx[0]))

    # 4. Mule network / Hijack attack (Scenario 3/4)
    mule_idx = df[df.attack_type.isin(["MULE_FANOUT", "AGENT_HIJACK"])].index
    if len(mule_idx) > 0:
        sample_cases.append(("4. Mule Ring Sub-limit Drain (Scenario 3 & 4)", mule_idx[0]))

    for label, idx in sample_cases:
        row = df.iloc[idx]
        f_score = scores["FUSED"][idx]
        dec = df["decision"].iloc[idx]

        print(f"\n[Case] {label}")
        print(f"  Agent/User ID   : {row.identity_id:<12} | Payee Account : {row.dest_account}")
        print(f"  Payment Amount  : ${row.amount:,.2f}  | User Avg Amt  : ${row.user_avg_amount:,.2f}")
        print(f"  Context Signals : API Calls={row.api_calls_per_min:.1f}/min, New Endpoint={row.new_endpoint}, File Instruction={row.instruction_from_file}")
        print(f"  Decomposed Risk : KYC={scores['kyc'][idx]:.1f} | API={scores['api'][idx]:.1f} | Txn={scores['txn'][idx]:.1f} | Graph={scores['graph'][idx]:.1f}")
        print(f"  -> Fused Score  : {f_score:.2f} / 100.0  ===> DECISION: [{dec}]")

        if dec == "ALLOW":
            print("     Policy Action: Approved automatically. Trust score preserved.")
        elif dec == "RESTRICT":
            print("     Policy Action: Daily limit reduced. Triggered step-up OTP authentication.")
        elif dec == "REVIEW":
            print("     Policy Action: Payment paused. Queued for human compliance officer review.")
        else:
            print("     Policy Action: BLOCKED. Agent API credentials revoked; incident logged.")
    print("-" * 78)


# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION ROUTINE
# ------------------------------------------------- ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Risk Scoring for AI-Agent Payments - PoC Benchmark"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. Omit for dynamic real-time runs.")
    parser.add_argument("--users", type=int, default=None,
                        help="Number of clean users (default: dynamic ~1800-2400).")
    parser.add_argument("--attacks", type=int, default=None,
                        help="Number of attack identities (default: dynamic ~12-18).")
    parser.add_argument("--no-demo", action="store_true",
                        help="Skip the live agent transaction walkthrough.")
    args = parser.parse_args()

    # Initialize RNG: Dynamic by default (different output every run)
    rng = np.random.default_rng(args.seed)

    # Dynamic or custom population parameters
    n_users = args.users if args.users is not None else int(rng.integers(1800, 2400))
    n_attacks = args.attacks if args.attacks is not None else int(rng.integers(12, 18))

    weights, thresholds = load_project_configs()

    print("\n" + "=" * 78)
    print(" ADAPTIVE RISK SCORING FOR AI-AGENT PAYMENTS")
    print(" Capstone Project | PES University (PW_27_RB_06)")
    print(f" Mode: {'Dynamic Entropy (Different Each Run)' if args.seed is None else f'Fixed Seed ({args.seed})'}")
    print(f" Population: {n_users:,} Users | {n_attacks} Injected Attack Vectors")
    print("=" * 78)

    # 1. Synthesize Clean Population & Inject Attacks
    clean = make_clean(rng, n_users=n_users, events_per_user=18)
    attack = inject_attacks(clean, rng, n_attacks=n_attacks)
    df = pd.concat([clean, attack], ignore_index=True)
    y = df.is_attack.values

    # 2. Compute Decomposed Scores
    scores = {
        "kyc": kyc_score(df),
        "api": api_score(df),
        "txn": txn_score(df),
        "graph": graph_score(df),
    }
    scores["FUSED"] = fuse(scores, weights)
    df["decision"] = decide(scores["FUSED"], thresholds)

    # 3. Live Demo Walkthrough (unless suppressed)
    if not args.no_demo:
        print_live_demo(df, scores, weights, thresholds)

    # 4. Evaluation Benchmark
    print(f"\nPOPULATION BENCHMARK SUMMARY:")
    print(f"Total Transactions: {len(df):,}   Attack Events: {y.sum():,} ({y.mean():.2%})\n")
    print(f"{'Domain Scorer':<16}{'PR-AUC':>10}{'Recall @ 1% FPR':>20}")
    print("-" * 48)
    for name in ["kyc", "api", "txn", "graph"]:
        pr = average_precision_score(y, scores[name])
        rec = recall_at_fpr(y, scores[name], fpr=0.01)
        print(f"{name.upper():<16}{pr:>10.3f}{rec:>20.2%}")

    pr_fused = average_precision_score(y, scores["FUSED"])
    rec_fused = recall_at_fpr(y, scores["FUSED"], fpr=0.01)
    print("-" * 48)
    print(f"{'FUSED (ADAPTIVE)':<16}{pr_fused:>10.3f}{rec_fused:>20.2%}")
    print("-" * 48)

    # 5. Adaptive Decision Confusion Matrix
    print("\nADAPTIVE AUTHORITY DECISION BREAKDOWN:")
    crosstab = pd.crosstab(
        df.is_attack.map({0: "Clean Event (0)", 1: "Attack Event (1)"}),
        df.decision,
        margins=True
    )
    print(crosstab.to_string())
    print("\n[Done] Pipeline evaluation completed successfully.\n")


if __name__ == "__main__":
    main()
