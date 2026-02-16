# Scripts Directory

Organized directory of utility, production, and development scripts.

## Directory Structure

- **`production/`** - Production and operational scripts
- **`evaluation/`** - Evaluation, benchmarking, and bakeoff scripts
- **`training/`** - ML training and data preparation scripts
- **`analysis/`** - One-off analysis, debugging, and diagnostic scripts
- **`archived/`** - Deprecated scripts kept for reference
- **`utils/`** - Shared utility modules

## Production Scripts

Located in `production/`:

- `launch.sh` - Launch full application (API + desktop)
- `health_check.py` - System health validation
- `self_test.py` - Complete self-test with all components
- `backfill_complete.py` - Full backfill: segments + facts
- `backfill_messages.py` - Simple message backfill
- `sync_contacts.py` - Sync contacts from macOS
- `sync_calendar.py` - Sync calendar events
- `batch_process_knowledge.py` - Batch knowledge graph processing
- `start_worker_loop.py` - Background worker daemon
- `autonomous_loop.sh` - Autonomous processing loop
- `start_mlx_server.sh` - MLX model server

## Evaluation Scripts

Located in `evaluation/`:

- `eval_*.py` - Various evaluation pipelines
- `extraction_bakeoff*.py` - Fact extraction experiments
- `prompt_bakeoff*.py` - Prompt engineering experiments  
- `reply_bakeoff.py` - Reply generation experiments
- `benchmark_speculative.py` - Speculative decoding benchmark

## Training Scripts

Located in `training/`:

- `prepare_*.py` - Data preparation for training
- `generate_*.py` - Generate training configs/data
- `build_all_profiles.py` - Build contact profiles

## Analysis Scripts

Located in `analysis/`:

- `analyze_*.py` - Analysis scripts
- `diagnose_*.py` - Diagnostic scripts
- `check_*.py` - Validation checks
- `test_*.py` - One-off test scripts (not unit tests)
- `trace_*.py`, `verify_*.py` - Debugging utilities

## Other Files

- `gen_ref_pages.py` - Generate documentation reference pages
- `check_swap.sh`, `check_training_status.sh` - Shell utilities
- `extraction_lab_configs.json` - Config for extraction experiments
