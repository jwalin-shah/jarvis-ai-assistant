# Debug Logging Guide

## Quick Start

Enable debug logging via environment variable:

```bash
# Enable DEBUG logging for ALL modules
export JARVIS_LOG_LEVEL=DEBUG
jarvis watch

# Or inline
JARVIS_LOG_LEVEL=DEBUG jarvis watch
```

## Logging Levels

```bash
DEBUG    # Detailed diagnostic info (verbose)
INFO     # General operational messages (default)
WARNING  # Warnings (non-critical issues)
ERROR    # Errors that don't stop execution
CRITICAL # Fatal errors
```

## Per-Module Debug Logging

Enable debug for specific modules only:

```python
import logging

# In your script or config
logging.getLogger("jarvis.contacts").setLevel(logging.DEBUG)
logging.getLogger("jarvis.topics").setLevel(logging.DEBUG)
logging.getLogger("jarvis.embedding").setLevel(logging.DEBUG)
```

## Configuration File

Edit `~/.jarvis/config.json`:

```json
{
  "logging": {
    "level": "DEBUG",
    "file": "~/.jarvis/logs/jarvis.log"
  }
}
```

## CLI Arguments

Many scripts accept `--log-level`:

```bash
# Evaluation with debug logging
make eval LOG_LEVEL=DEBUG

# Manual fact extraction
python -m jarvis.contacts.extract_facts --log-level DEBUG --chat-id <id>
```

## Log File Locations

- Main log: `~/.jarvis/logs/jarvis.log`
- Script logs: `~/.jarvis/logs/<script_name>.log`
- Evaluation logs: `logs/<evaluation_name>.log`

## What Gets Logged at DEBUG Level

- **Embedding**: Model load times, batch sizes, cache hits
- **NLI**: Entailment scores, rejection reasons
- **Contacts**: Fact extraction prompts, LLM responses
- **Topics**: Segment boundaries, topic discovery
- **Search**: Query rewrites, ranking scores
- **Database**: SQL queries, connection pool stats

## Example: Debugging Fact Extraction

```bash
# Full debug output for fact extraction pipeline
JARVIS_LOG_LEVEL=DEBUG python -c "
import logging
logging.basicConfig(level=logging.DEBUG)

from jarvis.contacts.instruction_extractor import get_instruction_extractor
extractor = get_instruction_extractor(tier='0.7b')
extractor.load()
# ... your debugging code ...
"
```

## Production Use

**Don't use DEBUG in production!** It:

- Logs PII (message content, contact names)
- Creates huge log files
- Degrades performance

Use INFO (default) or WARNING for production.

## Clean Logs

```bash
# Clear old logs
rm -rf ~/.jarvis/logs/*.log
rm -rf logs/*.log
```

## Structured Logging

All modules use Python's standard logging:

```python
import logging

logger = logging.getLogger(__name__)

# Standard log calls
logger.debug("Detailed diagnostic: %s", data)
logger.info("Normal operation: %s", event)
logger.warning("Non-critical issue: %s", issue)
logger.error("Error occurred: %s", error)
logger.exception("Error with traceback")  # Includes stack trace
```

## Performance Impact

| Level   | Performance Impact   | Use Case                          |
| ------- | -------------------- | --------------------------------- |
| DEBUG   | High (10-30% slower) | Development, debugging            |
| INFO    | Low (1-5% slower)    | Production, monitoring            |
| WARNING | Minimal (\u003c1%)   | Production (quiet)                |
| ERROR   | None                 | Production (silent unless errors) |
