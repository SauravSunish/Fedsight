# Person matching research baseline

The project has an optional OSNet x1.0 checkpoint trained by the Torchreid
authors on Market-1501. It is an existing pretrained baseline, not a model
trained on your cameras and not evidence that two people are the same.

1. In **Evidence & processing**, select `osnet_x1_0`.
2. Enter the full path to `models/osnet_x1_0_market1501.pth.tar` in the
   trusted local OSNet weights field.
3. Reprocess the intended sources to create descriptors using that checkpoint.
4. Review proposed matches against the original footage. Validate on separately
   labelled footage from the intended cameras before relying on its results.

Source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md
The Market-1501 dataset is for research use; verify its terms and checkpoint
licensing before any other use. The extracted dataset is under `data/reid/`.
This checkpoint was downloaded from the model zoo's Market-1501 column.
SHA-256: `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`.

Fine-tuning for a particular environment requires appropriately authorized,
identity-labelled camera data and a separate held-out evaluation set. The
installed PyTorch environment currently reports no CUDA, so training on this
computer may be slow. See EVALUATION_GUIDE.md for the separate pair classifier.
