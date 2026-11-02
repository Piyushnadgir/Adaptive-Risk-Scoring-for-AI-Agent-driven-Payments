"""
Starter PoC: fused risk scoring for AI-agent payments.

Pipeline: synthetic data (clean + injected attack) -> 4 rule scorers
-> weighted-sum fusion -> ALLOW/RESTRICT/REVIEW/REVOKE -> evaluation.

Run:  python fraud_poc.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

rng = np.random.default_rng(42)

# ---------------------------------------------------------------- 1. DATA
def make_clean(n_users=2000, events_per_user=20):
    rows = []
    for u in range(n_users):
        avg = rng.lognormal(mean=6, sigma=0.6)            # user's usual amount
        n_acc = 1 if rng.random() < 0.9 else 2            # a few legit multi-account users
        merchants = rng.integers(0, 3000, size=8)         # user's usual payees
        for _ in range(events_per_user):
            rows.append(dict(
                user_id=u, identity_id=f"ID{u}",
                identity_accounts=n_acc,
                kyc_age_days=int(rng.integers(30, 3000)),
                api_calls_per_min=max(1, rng.normal(5, 3)),
                new_endpoint=rng.random() < 0.05,
                instruction_from_file=rng.random() < 0.02,
                amount=float(rng.lognormal(np.log(avg), 0.5)),
                user_avg_amount=avg,
                dest_account=f"A{rng.choice(merchants)}",
                is_attack=0))
    return pd.DataFrame(rows)


def inject_attack(clean, n_identities=10, accounts_per_identity=5, events=12):
    """Hijacked agent reuses one identity across several mule accounts and
    sends just-under-limit amounts. Each signal is only mildly unusual."""
    rows = []
    base = clean.user_id.max() + 1
    for k in range(n_identities):
        mules = [f"MULE{k}_{j}" for j in range(accounts_per_identity)]
        for _ in range(events):
            rows.append(dict(
                user_id=base + k, identity_id=f"ID{base + k}",
                identity_accounts=accounts_per_identity,
                kyc_age_days=int(rng.integers(20, 200)),
                api_calls_per_min=max(1, rng.normal(14, 4)),
                new_endpoint=rng.random() < 0.3,
                instruction_from_file=rng.random() < 0.3,
                amount=float(rng.uniform(8500, 9900)),
                user_avg_amount=float(rng.lognormal(6, 0.6)) * 3,
                dest_account=str(rng.choice(mules)),
                is_attack=1))
    return pd.DataFrame(rows)


# -------------------------------------------------------------- 2. SCORERS
# Each returns 0-100. Add points per red flag, cap at 100.
def kyc_score(df):
    s = np.zeros(len(df))
    s += 30 * (df.identity_accounts > 2)
    s += 20 * (df.identity_accounts > 4)
    s += 25 * (df.kyc_age_days < 200)
    return np.minimum(s, 100)

def api_score(df):
    s = np.zeros(len(df))
    s += 30 * (df.api_calls_per_min > 12)
    s += 20 * df.new_endpoint
    s += 40 * df.instruction_from_file
    return np.minimum(s, 100)

def txn_score(df):
    s = np.zeros(len(df))
    s += 40 * df.amount.between(8000, 9999)               # just under a 10k limit
    s += 30 * (df.amount > 5 * df.user_avg_amount)
    return np.minimum(s, 100)

def graph_score(df):
    # Exclusive accounts = destinations that only ONE identity ever pays.
    # An identity fanning out to several exclusive accounts looks like mule setup.
    ids_per_dest = df.groupby("dest_account").identity_id.transform("nunique")
    exclusive = df.assign(excl=(ids_per_dest == 1)).drop_duplicates(["identity_id", "dest_account"])
    n_excl = exclusive.groupby("identity_id").excl.sum()
    n_excl = df.identity_id.map(n_excl)
    s = np.zeros(len(df))
    s += 40 * (n_excl >= 3)
    s += 30 * (n_excl >= 5)
    s += 30 * (ids_per_dest > 8)                          # hub account receiving from many identities
    return np.minimum(s, 100)


# --------------------------------------------------------- 3. FUSION + DECISION
WEIGHTS = dict(kyc=0.25, api=0.25, txn=0.25, graph=0.25)   # later: learn these (RL)

def fuse(scores):
    return sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)

def decide(score):
    # thresholds on the fused 0-100 score; keep these as plain, explainable rules
    return np.select([score < 20, score < 35, score < 50], ["ALLOW", "RESTRICT", "REVIEW"], "REVOKE")


# -------------------------------------------------------------- 4. EVALUATION
def recall_at_fpr(y, score, fpr=0.01):
    """Recall on attack events when only `fpr` of clean events may be flagged."""
    thr = np.quantile(score[y == 0], 1 - fpr)
    return float((score[y == 1] > thr).mean())

def main():
    clean = make_clean()
    attack = inject_attack(clean)
    df = pd.concat([clean, attack], ignore_index=True)
    y = df.is_attack.values

    scores = dict(kyc=kyc_score(df), api=api_score(df), txn=txn_score(df), graph=graph_score(df))
    scores["FUSED"] = fuse(scores)

    print(f"events: {len(df):,}   attack events: {y.sum()} ({y.mean():.2%})\n")
    print(f"{'scorer':<8}{'PR-AUC':>8}{'recall@1%FPR':>15}")
    for name, s in scores.items():
        print(f"{name:<8}{average_precision_score(y, s):>8.3f}{recall_at_fpr(y, s):>15.2f}")

    df["decision"] = decide(scores["FUSED"])
    print("\nDecisions on attack vs clean events:")
    print(pd.crosstab(df.is_attack, df.decision))

if __name__ == "__main__":
    main()
