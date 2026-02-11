# Fact KG Build Plan

Date: 2026-02-10

## Goal

Build a reliable personal knowledge graph ingestion pipeline from messy iMessage conversations:

1. Extract durable personal facts.
2. Resolve attribution (who the fact is about).
3. Resolve temporal status (current/past/future).
4. Upsert clean facts into the graph with provenance and confidence.

## Current State (What Already Exists)

### Implemented

1. Stage-1 candidate extractor (GLiNER):
   - `jarvis/contacts/candidate_extractor.py`
   - Supports label profiles (`high_recall`, `balanced`, `high_precision`) and natural-language labels.

2. Candidate extraction/evaluation tooling:
   - `scripts/extract_candidates.py`
   - `scripts/eval_gliner_candidates.py`
   - `scripts/run_gliner_eval_compat.sh`

3. Candidate goldsets and merged dataset:
   - `training_data/gliner_goldset/candidate_gold_merged_r4.json` (796 rows)
   - `training_data/gliner_goldset/gliner_metrics_high_precision_merged_r4_cleaned.json`

4. Stage-2 fact gate classifier (candidate keep/discard):
   - Trainer: `scripts/train_fact_filter.py`
   - Dataset builder: `scripts/build_fact_filter_dataset.py`
   - Latest model: `models/fact_filter_r4_clean.pkl`
   - Latest dev metrics from `logs/train_fact_filter.log`:
     - Accuracy: 0.693
     - Precision: 0.329
     - Recall: 0.781
     - F1: 0.463

5. Message-level gate model (upstream keep/discard):
   - Trainer: `scripts/train_message_gate.py`
   - Model artifact exists: `models/message_gate.pkl`

6. Existing production fact pipeline (legacy):
   - Extractor: `jarvis/contacts/fact_extractor.py` (regex + spaCy + optional NLI)
   - Watcher integration currently uses legacy extractor:
     - `jarvis/watcher.py` (`_extract_facts`)

7. Graph rendering layer:
   - `jarvis/graph/knowledge_graph.py` (reads `contact_facts` and builds graph output)

### Not Implemented Yet

1. Candidate extractor + fact gate are not integrated into live runtime ingestion.
2. No production attribution resolver (`self`, `other_participant`, `named_entity`).
3. No production temporal resolver (`current`, `past`, `future`, `unknown`).
4. No end-to-end fact ingestion contract with provenance-first upsert semantics.
5. No shadow-run gate for safe rollout before replacing legacy extraction.

## Baseline Metrics Snapshot

GLiNER stage-1 (`high_precision` profile):

1. `training_data/gliner_goldset/gliner_metrics_high_precision.json`
   - P 0.3476, R 0.4939, F1 0.4081 (250-set baseline)
2. `training_data/gliner_goldset/gliner_metrics_high_precision_merged_r3.json`
   - P 0.2628, R 0.4696, F1 0.3370 (593-set)
3. `training_data/gliner_goldset/gliner_metrics_high_precision_merged_r4_cleaned.json`
   - P 0.2343, R 0.4776, F1 0.3143 (796-set cleaned)

Interpretation: stage-1 recall held roughly stable while precision dropped as harder negatives were added. Stage-1 remains the bottleneck.

## Target Definition of Done

1. Accepted-fact precision >= 0.80 on a blinded 200-message audit.
2. Accepted-fact recall >= 0.45 on frozen dev goldset.
3. Attribution accuracy >= 0.85 overall.
4. Temporal status accuracy >= 0.80.
5. Duplicate upsert rate <= 5% on replay.
6. Runtime <= 250 ms/message average on-device.

## Delivery Plan

### Phase 0: Freeze Evaluation Contract (1 day)

Deliverables:

1. Freeze current gold/eval artifacts as baseline:
   - `training_data/gliner_goldset/candidate_gold_merged_r4.json`
   - `training_data/gliner_goldset/gliner_metrics_high_precision_merged_r4_cleaned.json`
2. Freeze train/dev filter splits:
   - `training_data/fact_candidates_train_r4_clean.jsonl`
   - `training_data/fact_candidates_dev_r4_clean.jsonl`
3. Document fixed command matrix in repo docs.

Exit gate:

1. Two consecutive reruns produce same metrics (+/-0.01).

### Phase 1: Candidate Extractor Bakeoff (2-3 days)

Deliverables:

1. Add extractor adapters to common candidate output schema:
   - GLiNER baseline
   - GLiNER2
   - NuExtract (schema-driven extraction)
2. Evaluate all extractors on the same frozen goldset.
3. Compare using precision-weighted score (F0.5) plus recall floor.

Exit gate:

1. Select primary extractor and fallback extractor with written benchmark evidence.

### Phase 2: Production Candidate + Fact Gate Path (2 days)

Deliverables:

1. Add production pipeline module:
   - Message gate -> candidate extraction -> fact gate.
2. Store candidate decisions with provenance:
   - message_id, span_text, span_label, fact_type, extractor_score, gate_score.
3. Add threshold config with explicit profiles for online/offline runs.

Exit gate:

1. Pipeline runs end-to-end on a 10k-message replay without crashes.
2. Candidate decision logging enabled for audit.

### Phase 3: Attribution and Temporal Resolver (4 days)

Deliverables:

1. Build subject resolver:
   - labels: `self`, `other_participant`, `named_entity`
2. Build temporal resolver:
   - labels: `current`, `past`, `future`, `unknown`
3. Add context-aware features:
   - speaker turn, pronouns, anchor verbs, neighboring messages, thread state.
4. Add resolver eval script and test fixtures.

Exit gate:

1. Attribution >= 0.85 overall.
2. Temporal >= 0.80 on labeled resolver dev set.

### Phase 4: KG Upsert Contract and Storage (2-3 days)

Deliverables:

1. Canonical accepted-fact schema:
   - subject_id, fact_type, value, temporal_status, confidence, source_message_ids.
2. Idempotent upsert key and conflict strategy.
3. Keep full provenance trail for every accepted fact.
4. Ensure graph loader consumes new accepted-fact records.

Exit gate:

1. Replay test shows <= 5% duplicate facts and deterministic upserts.

### Phase 5: Shadow Run and Rollout (3 days)

Deliverables:

1. Shadow mode in watcher:
   - run new pipeline without replacing legacy writes.
2. Build audit report:
   - false positives by label and fact type
   - false negatives by slice
3. Tune thresholds once using shadow outputs.
4. Controlled switch to new pipeline as primary.

Exit gate:

1. Meets all Definition of Done metrics for two consecutive shadow windows.

## What Is Trained (And Why)

1. Message gate classifier:
   - Filters obviously non-factual messages early.
2. Fact gate classifier:
   - Filters false candidate spans after extractor.
3. Attribution resolver:
   - Predicts who each accepted fact belongs to.
4. Temporal resolver:
   - Predicts current/past/future status of fact.

Note: these are separate tasks; one model is not enough for all four reliably.

## Immediate Next 3 Tasks

1. Add a common extractor adapter interface and run GLiNER vs GLiNER2 vs NuExtract bakeoff.
2. Implement production candidate+fact-gate module and wire a shadow path in `jarvis/watcher.py`.
3. Define resolver annotation schema and create first 300 attribution/temporal labels.
