# Adaptive Risk Scoring for AI-Agent Payments

PES University | UE24CS320A Capstone | Project ID PW_25_RB_06 | Guide: Prof. Rajesh Banginwar
Team: Ananya Uppili, Ananya V, Abhinav Bora, Piyush Nadgir

## What this project does
A tiered pipeline that scores every AI-agent payment action using intent, API/access context,
KYC context, transaction, agent-behaviour and account-network signals, then decides how much
authority the agent keeps right now: ALLOW / RESTRICT / REVIEW / REVOKE.
See `docs/architecture.md` for the 10-stage pipeline.

## Status
Research prototype on public datasets and a simulated mock bank. No real bank integration,
no live transactions, no regulatory-compliance claim.

## Repo map
| Path | Purpose |
|---|---|
| `src/adaptive_risk/pipeline/` | The 10 stages + orchestrator (tiers, early exit) |
| `src/adaptive_risk/mock_bank/` | Simulated bank API (balance, transactions, beneficiary, transfer) |
| `src/adaptive_risk/agent/` | Tool-calling agent + attack hooks |
| `src/adaptive_risk/gateway/` | Security layer the agent must call through |
| `src/adaptive_risk/data/` | Dataset loaders and the common feature schema |
| `src/adaptive_risk/evaluation/` | Metrics, the three configurations, experiment runner |
| `scenarios/` | The four demo scenarios and attack cases (YAML) |
| `configs/` | Thresholds, weights, tier settings |
| `data/` | Datasets (NOT committed; see `data/README.md`) |
| `experiments/` | One folder per experiment, with config and results |
| `notebooks/` | EDA per dataset |
| `docs/` | Architecture, dataset validation, decision log |

## Quick start
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pytest

## Team workflow
See `CONTRIBUTING.md`.
