# Collecting a real benchmark

The synthetic walkthrough verifies application behavior. It does not measure the
detector, tracker or OSNet. Keep it separate from real evidence and real model data.

1. Record consented volunteers in distinct sessions and cameras. Include hard
   negatives with similar clothing, low light, occlusion and viewpoint changes.
2. Preserve original hashes, camera configuration and clock-alignment notes. Keep
   consent and a retention date with each case. Do not use adjacent frames as
   independent examples.
3. Assign consented subject IDs, original sequence IDs and case/session IDs. Place
   all occurrences of each identity and sequence in only one split. If identities
   connect several sessions, split the connected components together.
4. Generate cross-source positive and negative pairs after assigning splits.
   Capture the exact features from proposal records in the exported case JSON.
   Do not infer true identity labels from the current model or from acceptance alone.
5. Supply four separate sets: train, calibration, validation, test. Calibration fits
   the sigmoid mapping; validation selects the classifier and freezes the rejection
   threshold; test is evaluated after model selection. Every split needs both classes.
6. Use Evaluation lab or `python -m fedsight evaluate pairs.json results.json`.
   The model artifact is numerical JSON; it does not execute pickle content.
7. Report counts and per-condition errors alongside aggregate metrics. For final
   results use a previously unseen benchmark; repeated development on a test set
   compromises its held-out status.

Download the JSON format from Evaluation lab or run:

```powershell
.venv\Scripts\python.exe -m fedsight sample-dataset examples.json
```

Each row has `feature_version`, `features`, `label`, `identity_a`, `identity_b`,
`sequence_a`, `sequence_b`, `case_id`, `split`, and `synthetic`. Keep real data
outside source control. Never relabel generated vectors as real observations.

## Planned empirical comparisons

Run frozen input manifests through baseline cosine, quality-weighted embeddings,
context features, calibrated classifier, and calibrated classifier with rejection.
Keep identical held-out IDs, source hashes and thresholds across comparisons.
The functions in `fedsight.evaluation` compute binary and retrieval metrics.
Detection/tracking metrics require bounding-box/track annotations and external
evaluation tooling; they are not inferred from candidate review rates.

## Model-weight provenance

Select an OSNet architecture and matching trusted Re-ID checkpoint in the processing
screen. The job records its SHA-256. A named checkpoint is not automatically a
validated checkpoint: compare its results on the target camera domain. ImageNet
pretraining alone is not person Re-ID training.

## Security demonstrations

Run `python -m pytest tests/test_security.py -q`. The tests mutate disposable
copies and cover ciphertext alteration, wrong keys/AAD, audit edit/delete/reorder,
checkpoint-anchored tail deletion, persistent stale/duplicate rejection, and record
inventory tampering. Do not perform attack demonstrations on real case data.
