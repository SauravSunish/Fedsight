# Local open-vocabulary preview search

This experimental feature ranks existing tracklet previews using MobileCLIP2-S0.
It is separate from OSNet's person-to-person embeddings. Ranking scores are not
probabilities, and an unreviewed candidate is not proof of identity or presence.

Install `requirements-semantic.txt` in the project's environment and obtain a
trusted, compatible MobileCLIP2-S0 OpenCLIP checkpoint. The model's publisher
lists its weights and license at
https://huggingface.co/timm/MobileCLIP2-S0-OpenCLIP. Review the model license
for your intended use. Select the local checkpoint in **FedSight Investigator →
Open-vocabulary visual search**, then click **Index current case previews**.
You can set `FEDSIGHT_CLIP_WEIGHTS` to that path before starting FedSight to
automatically index previews after future successful video-processing jobs.

Vectors are persisted in encrypted, case-scoped, audited records; the current
implementation performs exact NumPy cosine search rather than storing plaintext
vectors in LanceDB. Search rebuilds its matrix from encrypted records and has
not been benchmarked at large scale. Case, active job, checkpoint hash and crop
hash constrain which vectors can be searched. Rebuilding skips matching records;
reprocessing a video generates new track IDs and needs indexing again.

Enable MobileCLIP search and ask a visual question. Use the camera, clip-relative
time and minimum box-height controls to narrow results. Disable local vision to
see ranked candidates directly, or enable it to send only those candidates to
the local Ollama vision model. Scores and VLM verdicts require review against
the original video. Small crops, occlusions, missing objects, the preview limit,
and models' domain differences can cause missed or incorrect matches. Time
filters are relative to each recording and do not establish cross-camera timing.

On the development workstation, Windows application control blocked the native
`regex` library required by OpenCLIP and the LanceDB native library. As a result,
the real model and large-scale performance have not been verified there; the
application reports the indexing error instead of silently inventing results.
