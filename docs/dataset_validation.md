# Dataset validation checklist
Apply to every dataset before use; record the outcome in `data/datasets.yaml`.

1. Provenance: real, anonymised-real, simulated, or synthetic? How generated?
2. Label quality: source of labels, noise, delay, completeness.
3. Realism: class imbalance, time span, population, temporal drift.
4. Fit: which pipeline layer does it serve? Which features map to our schema?
5. Interpretability: are features meaningful (needed for explainable scores) or anonymised (e.g. PCA)?
6. Documentation: peer-reviewed paper or datasheet? Known criticisms?
7. Access and licence: can we use and cite it? Size manageable?
8. Evaluation suitability: can we do time-based splits, not just random splits?
