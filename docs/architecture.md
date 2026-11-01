# Architecture

Pipeline (every agent action):
1. Authentication and policy check
2. Tool-call pre-filter (Tier 0)
3. Intent verification
4. Identity / KYC context
5. Transaction risk (Tier 1)
6. Agent behaviour and trust score (Tier 2)
7. Graph and AML analysis (Tier 3)
8. Fused, decomposed risk score
9. Adaptive authority decision (ALLOW / RESTRICT / REVIEW / REVOKE)
10. Execute, log, update trust

Add the pipeline diagram here: `docs/pipeline_diagram.png`.

Rules:
- Cheap stages run first; low-risk actions exit early; Tier 3 runs only on suspicious/uncertain cases.
- Every stage returns a `StageResult` (see `pipeline/base.py`): score, flags, reason.
- Fusion keeps component scores visible (explainability).
