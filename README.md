# Stine-Score

**Stine-Score** predicts the probability that a Kilosort/Phy cluster would be
called `good` by me, Gabe Stine. It is a practical curation-assistance tool for
Neuropixels recordings: use the probability to prioritize manual review, not
as a substitute for scientific quality control.

The included pretrained model was trained on my manual Phy curation labels.
It learns nonlinear interactions among amplitude, firing-rate, refractory, and
stability measurements, then returns a `good_probability` for each
cluster.

I have found this metric to be useful for my spike sorting, so I am sharing it. 
Note, however, that this tool was inspired by and is similar to UnitRefine (Jain et al., 2025; https://www.biorxiv.org/content/10.1101/2025.03.30.645770v2), 
which is much more developed. So, you should probably use that. The main benefit 
of Stine-Score is that (1) it is simple and fast to run and (2) it automatically 
integrates with Phy. I have not compared Stine-Score to UnitRefine.

## What it measures

The extractor derives 48 features from Kilosort/Phy outputs, including:

- mean and median spike amplitude, amplitude-distribution shape, smoothness,
  and evidence for lower-tail truncation;
- mean firing rate, firing-rate stability, silent gaps, and presence;
- inter-spike-interval violations and autocorrelogram trough evidence;
- Kilosort contamination, template amplitude, and other available summaries.

Raw waveform sampling is available but disabled by default because it is slow
and was not needed for the first model.

## Pretrained model

[`models/stine_score_v0.1.joblib`](models/stine_score_v0.1.joblib) is included
for immediate scoring. It was trained on 3,556 manually labeled units from 31
probes.

Nested held-out-session validation produced a Brier score of 0.0765, ROC AUC
of 0.9633, and average precision of 0.9591. These numbers estimate agreement
with my historical curation decisions, not a universal biological ground truth. See
[`models/README.md`](models/README.md) for limitations and version details.

## Installation

Python 3.12 was used to build and test the included model.

```powershell
git clone https://github.com/gabrielstine/Stine-Score.git
cd Stine-Score
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Score one Phy folder

To apply the included model, you only need the path to one Phy output folder. The folder must contain `spike_clusters.npy`,
`spike_times.npy`, `amplitudes.npy`, `params.py`, and either `cluster_info.tsv`
or Kilosort's `cluster_Amplitude.tsv`, `cluster_ContamPct.tsv`, and
`cluster_KSLabel.tsv` summary files.

```powershell
.\.venv\Scripts\python score_probe.py `
  "Kilosort output directory" `
  models\stine_score_v0.1.joblib
```

This writes `cluster_good_probability.tsv` into that output folder. Existing Phy
labels and Kilosort arrays are not changed. The command stops if that file
already exists; add `--overwrite` only when intentional. Reopen Phy to see
`good_probability` as a cluster column.

If the cluster IDs change after further Phy curation, rerun the command with
`--overwrite` to regenerate probabilities for the current cluster table.

## Batch-score every Phy folder under a base directory

Give the batch command any base directory. It recursively finds directories
that contain the required Kilosort/Phy files; no naming convention or manifest
file is required. A typical layout might look like:

```text
<data-root>/
  <session>/
    <session>_<stream>/
      phy_*_<stream>_ap/
        spike_clusters.npy
        spike_times.npy
        amplitudes.npy
        cluster_info.tsv   # or cluster_Amplitude/ContamPct/KSLabel TSVs
        params.py
```

First do a dry run. It extracts and scores all clusters but writes only local
cache files.

```powershell
.\.venv\Scripts\python score_all_probes.py "D:\your-data-root" `
  models\stine_score_v0.1.joblib
```

When the output looks right, add a sortable Phy metadata column to every found
Phy folder:

```powershell
.\.venv\Scripts\python score_all_probes.py "D:\your-data-root" `
  models\stine_score_v0.1.joblib --apply
```

This creates `cluster_good_probability.tsv` inside each Phy folder, with
columns `cluster_id` and `good_probability`. Existing Kilosort arrays and Phy
labels are not changed. Reopen Phy to see `good_probability` as a cluster
column.

If a probability file already exists, the script stops before writing. Use
`--overwrite` only when you intentionally want to replace it.

## Interpreting the probability

The value is the model's estimate of my `good` label, conditional on this
training set and preprocessing pipeline. A sensible initial use is triage:

- high probability: prioritize as likely good;
- low probability: deprioritize or inspect quickly;
- intermediate probability: review carefully.

The pretrained model's held-out-session performance should not be assumed to
transfer unchanged to a different lab, recording setup, Kilosort version,
brain region, or curator. For a different curation policy, retrain or calibrate
the model on your own labels.

## Train your own Stine-Score-style model

Give the training extractor a base directory containing one or more Phy output
folders. It recursively discovers them and uses only clusters with saved
`good`, `mua`, or `noise` labels; unlabeled folders are skipped automatically.

For training, point the command at a directory containing **manually curated**
outputs. Without a manifest, Stine-Score cannot reliably distinguish complete
manual labels from any pre-existing or partial Kilosort/Phy labels.

```powershell
.\.venv\Scripts\python extract_features.py "D:\your-data-root"
.\.venv\Scripts\python train_model.py artifacts\discovered_features\features.csv
```

Training compares a logistic baseline, nonlinear additive boosting, and an
interaction-capable boosted-tree model. It validates by holding out whole
sessions and selects the model with the best Brier score.

Useful outputs include:

- `artifacts/model/model_comparison.csv`
- `artifacts/model/held_out_session_predictions.csv`
- `artifacts/model/reliability_boosted_interactions.csv`
- `artifacts/model/unit_quality_model.joblib`

For descriptive feature-family importance:

```powershell
.\.venv\Scripts\python analyze_model.py `
  artifacts\model\unit_quality_model.joblib `
  artifacts\features.csv
```

## Utilities

`fix_phy_params_paths.py` can repair invalid `dat_path` values in Phy
`params.py` files by replacing them with a portable relative raw-trace path.
It is a dry run unless `--apply` is supplied.

## License

Released under the [MIT License](LICENSE).
