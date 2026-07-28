# Stine-Score pretrained model

`stine_score_v0.1.joblib` is a calibrated, interaction-capable histogram
gradient-boosted classifier. It returns the probability that Gabe Stine would
label a current Phy cluster `good`.

## Training data and target

- 3,556 labeled units from 31 probes across 10 Neuropixels sessions.
- Positive class: `good`.
- Negative class: `mua` and `noise`.
- Excluded class: `unsorted`.
- Merge and split history was intentionally ignored.

The model uses 48 numerical features derived from Kilosort/Phy outputs. Major
feature families are firing-rate level and stability, amplitude level and
distribution, refractory/ACG evidence, presence, and available Kilosort
summaries.

## Validation

Evaluation used nested leave-one-session-out validation with Platt probability
calibration fitted without the held-out session.

| Metric | Value |
| --- | ---: |
| Brier score | 0.07645 |
| Log loss | 0.24888 |
| ROC AUC | 0.96331 |
| Average precision | 0.95913 |

These are estimates of agreement with Gabe's past labels. They are not an
objective measure of single-unit quality and may not transfer to another
curator or data-processing pipeline.

## Compatibility

The artifact was serialized with Python 3.12, scikit-learn 1.9.0, NumPy 2.5.1,
Pandas 3.0.5, and joblib 1.5.3. Install the repository's pinned
`requirements.txt` before loading it.

SHA-256: `d71ac5a52d0b0a38788e0e1f41166634c0d919ba2826b278451cd2feef4ead29`

## Responsible use

Use the score to rank or triage clusters, then inspect the diagnostics in Phy.
Do not use it as an automatic final labeler without validating it on your own
recordings and curation policy.
