# Responsible use

FedSight is an exploratory, human-in-the-loop prototype. It is not a validated surveillance, biometric-identification, safety, or law-enforcement system.

## Before using non-synthetic material

You are responsible for confirming that you have a lawful basis and permission to collect, store, process, and share every recording and related record. Requirements vary by jurisdiction and context; this document is not legal advice.

At minimum:

- prefer the built-in synthetic walkthrough for demonstrations;
- collect only the material needed for a clearly defined purpose;
- avoid covert monitoring and sensitive locations;
- disclose recording and analysis where required;
- restrict access to case data, encryption keys, exports, and backups;
- define retention and deletion periods before collection;
- review the terms of models, datasets, cameras, and third-party providers;
- document model configuration, sampling, limitations, and reviewer decisions.

## Human review is required

Detections, similarity scores, tracklets, inferred routes, natural-language responses, and alerts can all be wrong. They must be treated as uncertain leads linked back to original source material.

Do not use FedSight output by itself to:

- identify a person;
- infer intent, guilt, health, emotion, protected characteristics, or risk;
- make employment, education, housing, credit, insurance, immigration, policing, or other consequential decisions;
- trigger an automated intervention;
- claim that an event did or did not occur.

## Security limitations

The prototype encrypts selected local assets and maintains an HMAC-linked audit trail, but this does not make the system production secure. Local metadata, configuration, process memory, exports, logs, operating-system access, backups, and optional external services can create additional exposure.

Never commit recordings, databases, model weights, camera addresses, credentials, tokens, or encryption keys. Protect keys separately; losing them can make encrypted evidence unreadable.

## Scope of this repository

The repository exists to support technical learning, portfolio review, and controlled research. A production application would require independent security, privacy, legal, accessibility, bias, performance, and operational review, plus evaluation with representative data and trained users.
