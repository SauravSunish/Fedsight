# FedSight 2.0 implementation status

This build implements the roadmap's local forensic workflow. It does not claim
that an eight-week research programme or production validation has been completed.

## Implemented in the local application

- Ten-workspace Streamlit dashboard backed by persistent encrypted SQLite records.
- Multiple cases, up to 32 recordings per case, source SHA-256 manifests,
  encrypted immutable original files, derived preview provenance, and case states.
- Existing YOLO / ByteTrack / BoT-SORT / OSNet pipeline retained. Processing jobs
  checkpoint completed sources; retries skip completed sources. Jobs are synchronous
  and resume at source boundaries, not at individual frames.
- Auto tracker selection prefers ByteTrack. When the native LAP extension is
  unavailable, Auto uses a distinct portable motion/IoU tracker with Hungarian
  assignment. The effective backend is recorded. Explicit native selections fail
  clearly instead of silently substituting a different tracker.
- Quality components for sharpness, brightness, contrast, crop area, confidence,
  occlusion and visibility; diverse crop selection and quality-weighted embeddings.
- Multi-shot ranking, mutual-neighbour and assignment hints, raw cosine retained
  separately from centered/CSLS ranking. NumPy exact retrieval; optional FAISS.
- Versioned pair features, transparent uncalibrated AICE terms, unknown-clock
  handling, uncertainty margins, explicit low-score rejection and human decisions.
- Numerical JSON logistic/gradient-boosting/MLP model comparison. Independent
  training/calibration/validation/test splits, two-sided identity and sequence
  leakage checks, validation-only model/threshold selection, held-out confusion
  counts, precision/recall/F1, PR-AUC, ROC-AUC, Brier, ECE and reliability plots.
- Camera registry, directed travel-time bounds with provenance, offset/drift-aware
  time anchors, bounded multi-hop route search, plausible/implausible/unknown context.
- Append-only accept/reject/defer decisions with actors, reasons, timestamps,
  immutable machine proposals and versioned runs. No automatic appearance merge.
- Human-confirmed appearance graph and conflict detection, original-video context
  playback, clip-relative/corrected timeline, notes and bookmarks.
- Search by source, time, quality, approximate central-crop colour, motion and
  duration. Colour is an unvalidated heuristic; bag and hat remain explicitly unknown.
- Evidence digest with exact source/frame/time references and thumbnails. Original
  video playback is range-bounded. No proprietary synopsis or derived video montage.
- Batch stationary-duration, rectangle-zone and line-crossing rules with review
  records and deduplication; sampled image-plane motion density and review analytics.
- AES-256-GCM for original files, previews, embeddings and database record payloads;
  fresh random nonces, metadata AAD, independent encryption/audit keys.
- HMAC-SHA3-256 audit chain, record inventory/content verification, signed exportable
  checkpoints and persisted replay detection. Security attack tests are in tests/.
- ZIP exports with PDF, JSON, CSV, thumbnails, source/derived hashes and review history.
  CSV formula injection is neutralised. Export fails if integrity checks fail.
- Three allowed read-only evidence questions. No model-generated Cypher execution.
- CLI, optional bearer-token local API, and optional parameterised Neo4j projection.
- Clearly isolated synthetic workflow and synthetic feature datasets; in-process
  FedAvg/FedProx logistic training sandbox.

## Requires supplied data or infrastructure

- Actual person-Re-ID checkpoint and consented, identity-labelled target footage.
  Default Torchreid ImageNet weights are not person-Re-ID-trained weights.
- Real held-out accuracy, tracking HOTA/IDF1/ID-switch annotations, detector mAP,
  route-quality labels, workflow timing studies and controlled model ablations.
- A running Neo4j instance and credentials. Sync adapter is implemented; remote
  deployment cannot be verified without a server. SQLite is authoritative and the
  Neo4j graph is a decrypted, explicitly requested projection, not encrypted storage.
- Verified cameras, event anchors and timezone/offset provenance for route claims.
- Institutional collection/retention procedures and independent review for deployment.

## Deliberately deferred / not claimed

- Multi-user login/RBAC/SSO, production background queue, distributed processing,
  central key custody/rotation, automated retention deletion and legal admissibility.
  The dashboard is a trusted local workstation. Reviewer names are self-attribution.
- Live MJPEG/RTSP/ONVIF/VMS adapters, continuous notifications and scheduled analytics.
- Trained bag/hat classifiers, calibrated floorplan occupancy, video clip rendering.
- Normal-only convolutional autoencoder training; stationary-duration rules provide
  the roadmap's basic anomaly-triage fallback without semantic crime labels.
- Flower network federation, secure aggregation, differential privacy and realistic
  non-IID federation benchmarks. The simulation is one process, not a deployed FL system.
- GAT/TGN, CKKS, face recognition, digital twins and replacement Re-ID backbones.

## Known operating bounds

- Local single-trust-domain storage; audit metadata retains actor, case and record IDs
  in plaintext. Record content and media are encrypted. An attacker with both keys
  and the database can rebuild history. Keep retained checkpoints and keys separately.
- Case exports are deliberately plaintext for review and must be handled accordingly.
- Graph groups are derived from the latest complete proposal run. Reranking creates
  new proposals for review; earlier review history stays accessible and exportable.
- A failed source can leave encrypted, audited partial derivatives; only completed
  jobs contribute active tracklets. Interrupted workers require explicit recovery.
- Route maps may be incomplete. Route contradictions influence review explanations
  without hard rejection of an otherwise plausible visual candidate.
- Training results on supplied synthetic examples are not real-world performance.

## Implementation references

- AES-GCM: https://cryptography.io/en/latest/hazmat/primitives/aead/
- Calibration: https://scikit-learn.org/stable/modules/calibration.html
- Neo4j managed transactions: https://neo4j.com/docs/python-manual/current/transactions/
- Streamlit video controls: https://docs.streamlit.io/develop/api-reference/media/st.video
