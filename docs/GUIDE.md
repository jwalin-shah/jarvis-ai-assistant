# Documentation Guide

This guide is the canonical index for JARVIS documentation.

## Quick Start

| Doc | Purpose |
|-----|---------|
| `README.md` | Project overview and setup |
| `CLAUDE.md` | Development instructions and component status |
| `docs/CLI_GUIDE.md` | Complete CLI reference |
| `docs/API_REFERENCE.md` | REST API documentation |

## Core Documentation

### Usage & Integration
- `docs/CLI_GUIDE.md` - CLI commands and examples
- `docs/API_REFERENCE.md` - REST API endpoints
- `docs/MCP_INTEGRATION.md` - MCP server integration
- `docs/PERFORMANCE.md` - Performance tuning

### System Design
- `docs/JARVIS-v1-Design-Document.md` - Architecture overview
- `docs/JARVIS-v1-Development-Guide.md` - Implementation guidance
- `docs/CLASSIFIER_SYSTEM.md` - Trigger & response classifiers

### Quality & Evaluation
- `docs/EVALUATION_AND_KNOWN_ISSUES.md` - Evaluation pipeline + known issues
- `docs/PLAN.md` - Current implementation plan
- `BENCHMARKS.md` - Latest benchmark results

### Reference
- `docs/MODEL_SELECTION_STRATEGY.md` - Model selection research
- `docs/TRIGGER_LABELING_GUIDE.md` - Labeling guide for trigger data

## Scripts

Core scripts in `scripts/`:
- `train_trigger_classifier.py` - Train trigger SVM
- `build_da_classifier.py` - Build DA classifier
- `eval_pipeline.py` - Evaluation pipeline
- `eval_full_classifier.py` - Full classifier evaluation
- `overnight_evaluation.py` - Full benchmark suite
- `generate_report.py` - Generate benchmark reports

Archived scripts in `scripts/archive/` - one-off data prep and experimentation.

## Archived Documentation

Historical docs moved to `docs/archive/`:
- Implementation plans (FROM_SCRATCH_*.md)
- Refactoring plans (REFACTOR*.md)
- Phase summaries (PHASES_*.md, FIXES_SUMMARY.md)
- Experiment notes (CLASSIFIER_EXPERIMENTS.md)

These are preserved for reference but superseded by current docs.
