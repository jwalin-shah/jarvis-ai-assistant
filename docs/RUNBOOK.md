# JARVIS Operational Runbook

**Purpose:** Step-by-step procedures for operating, troubleshooting, and recovering JARVIS in production.

**Audience:** On-call engineers, system administrators

**Last Updated:** 2026-02-10

---

## Quick Reference

| Issue           | Command                                                    | Response Time |
| --------------- | ---------------------------------------------------------- | ------------- |
| Health check    | `jarvis health`                                            | <5s           |
| Check circuits  | `curl http://localhost:8742/circuits`                      | <1s           |
| Reset circuit   | `curl -X POST http://localhost:8742/circuits/{name}/reset` | <1s           |
| Clear queue     | `jarvis tasks clear-completed`                             | <5s           |
| Memory status   | `jarvis health \| grep memory`                             | <5s           |
| Restart service | `jarvis restart`                                           | 30-60s        |

---

## 1. Alert: Circuit Breaker Opened

### Severity: WARNING

### Symptoms

- Feature degraded to fallback mode
- Users seeing template responses instead of AI-generated content
- `/circuits` endpoint showing `state: "open"`

### Diagnosis

```bash
# 1. Check which circuits are open
curl -s http://localhost:8742/circuits | jq '
  .circuits
  | to_entries[]
  | select(.value.state == "open")
  | {name: .key, state: .value.state, failures: .value.total_failures}'

# 2. Check recent errors
jarvis logs --since 1h | grep -E "(circuit|error|fail|exception)" | tail -20

# 3. Check system resources
jarvis health --verbose

# 4. Check model status
jarvis models status
```

### Common Causes

| Cause            | Detection                       | Fix                      |
| ---------------- | ------------------------------- | ------------------------ |
| Memory pressure  | `memory_pressure: red/critical` | Free memory, restart     |
| Model corruption | `ModelLoadError` in logs        | Verify model files       |
| Database lock    | `iMessageQueryError`            | Wait or restart iMessage |
| MLX GPU error    | Metal assertion in logs         | Restart JARVIS           |

### Resolution Steps

**Step 1: Identify Root Cause**

```bash
# Check memory
free_mb=$(jarvis health --json | jq '.memory.available_mb')
if (( $(echo "$free_mb < 500" | bc -l) )); then
    echo "MEMORY PRESSURE - Free memory: ${free_mb}MB"
fi

# Check model
if jarvis logs --since 10m | grep -q "ModelLoadError"; then
    echo "MODEL ERROR detected"
fi

# Check database access
if ! jarvis db ping; then
    echo "DATABASE ACCESS ISSUE"
fi
```

**Step 2: Apply Fix Based on Cause**

_Memory Pressure:_

```bash
# 1. Trigger emergency mode
jarvis admin emergency-mode

# 2. Clear caches
jarvis cache clear --all

# 3. Unload models to free memory
jarvis models unload

# 4. Check recovery
sleep 5
jarvis health
```

_Model Error:_

```bash
# 1. Verify model files
jarvis models verify

# 2. If corrupted, re-download
jarvis models download --force lfm-1.2b

# 3. Reload
jarvis models reload
```

_Database Lock:_

```bash
# Check if iMessage is running
if pgrep -x "Messages" > /dev/null; then
    echo "iMessage running - waiting for lock release"
    sleep 30
fi

# Verify access
jarvis db ping
```

**Step 3: Reset Circuit**

```bash
# Reset specific circuit
curl -X POST http://localhost:8742/circuits/model_generation/reset

# Verify circuit closed
curl -s http://localhost:8742/circuits/model_generation | jq '.state'
# Should return "closed"
```

**Step 4: Verify Recovery**

```bash
# Test generation
jarvis draft test-chat-id --instruction "Say hello"

# Check health
jarvis health
```

### Escalation

- If circuit keeps opening after reset: **Escalate to engineering** (possible bug)
- If multiple circuits open simultaneously: **P0 incident** (systemic failure)

---

## 2. Alert: High Memory Usage

### Severity: CRITICAL (>95%), WARNING (>85%)

### Symptoms

- Memory pressure level at `red` or `critical`
- Slow response times
- Model generation timeouts
- System becoming unresponsive

### Diagnosis

```bash
# Check current memory state
jarvis health --verbose

# Detailed memory breakdown
python3 << 'EOF'
import psutil
mem = psutil.virtual_memory()
print(f"Total: {mem.total / 1e9:.1f} GB")
print(f"Available: {mem.available / 1e9:.1f} GB")
print(f"Used: {mem.used / 1e9:.1f} GB ({mem.percent}%)")
print(f"Active: {mem.active / 1e9:.1f} GB")
print(f"Inactive: {mem.inactive / 1e9:.1f} GB")
EOF

# Check MLX/Metal memory
python3 << 'EOF'
import mlx.core as mx
if hasattr(mx.metal, 'get_active_memory'):
    active = mx.metal.get_active_memory()
    print(f"Metal Active: {active / 1e6:.0f} MB")
EOF

# Check JARVIS processes
ps aux | grep -i jarvis | grep -v grep

# Find memory-hungry processes
ps aux | sort -nr -k 4 | head -10
```

### Immediate Response (Critical >95%)

```bash
#!/bin/bash
# emergency_memory_response.sh

echo "Triggering emergency memory response..."

# 1. Emergency mode (disables non-essential features)
javis admin emergency-mode

# 2. Clear all caches
echo "Clearing caches..."
javis cache clear --all

# 3. Unload models
echo "Unloading models..."
javis models unload

# 4. Cancel non-essential tasks
echo "Cancelling background tasks..."
javis tasks cancel --type=batch_export --type=embedding_sync

# 5. Clear completed tasks
echo "Clearing completed task history..."
javis tasks clear-completed

# 6. Monitor recovery
echo "Monitoring recovery..."
for i in {1..12}; do
    mem_pct=$(psutil virtual_memory percent 2>/dev/null || echo "N/A")
    echo "Memory usage: ${mem_pct}%"
    if [[ "$mem_pct" != "N/A" && $(echo "$mem_pct < 85" | bc) -eq 1 ]]; then
        echo "Memory pressure relieved"
        break
    fi
    sleep 5
done

echo "Emergency response complete"
```

### Resolution

**Step 1: Identify Memory Leak Source**

```bash
# Check for memory growth over time
for i in {1..6}; do
    echo "$(date): $(ps aux | grep jarvis | grep -v grep | awk '{sum+=$6} END {print sum/1024 " MB"}')"
    sleep 10
done

# Check log for memory-related errors
jarvis logs | grep -i "memory\|leak\|unload\|gc"
```

**Step 2: If Memory Not Recovering**

```bash
# Graceful restart
jarvis stop
sleep 5

# Clear any stale lock files
rm -f ~/.jarvis/*.lock

# Start fresh
jarvis start

# Verify
sleep 10
jarvis health
```

**Step 3: Prevent Recurrence**

```bash
# Adjust memory thresholds (if consistently hitting limits)
javis config set memory.full_mode_mb 6000
javis config set memory.lite_mode_mb 400

# Enable aggressive cleanup
javis config set tasks.auto_cleanup true
javis config set cache.ttl_seconds 300
```

---

## 3. Alert: Task Queue Backlog

### Severity: WARNING

### Symptoms

- High number of pending tasks
- Delayed task execution
- Growing queue size metric

### Diagnosis

```bash
# Queue statistics
jarvis tasks stats

# List pending tasks (oldest first)
javis tasks list --status pending --sort oldest | head -20

# Check worker status
jarvis workers status

# Check for stuck workers
jarvis workers list | grep -E "(stuck|hung|timeout)"
```

### Resolution

**Step 1: Assess the Situation**

```bash
#!/bin/bash
# assess_queue.sh

stats=$(jarvis tasks stats --json)
total=$(echo "$stats" | jq '.total')
pending=$(echo "$stats" | jq '.by_status.pending // 0')
running=$(echo "$stats" | jq '.by_status.running // 0')
failed=$(echo "$stats" | jq '.by_status.failed // 0')

echo "Queue Status:"
echo "  Total: $total"
echo "  Pending: $pending"
echo "  Running: $running"
echo "  Failed: $failed"

# Check if workers are processing
if [[ $running -eq 0 && $pending -gt 10 ]]; then
    echo "WARNING: Workers not processing tasks!"
fi
```

**Step 2: If Workers Stuck**

```bash
# Restart workers
jarvis workers restart

# Verify
sleep 5
jarvis workers status
```

**Step 3: Clear Stale Tasks**

```bash
# Cancel tasks older than 1 hour
jarvis tasks cancel --older-than 1h --status pending

# Clear failed tasks that won't retry
jarvis tasks clear --status failed --max-retries-exceeded

# Clear completed task history
jarvis tasks clear-completed
```

**Step 4: If Queue Corrupted**

```bash
# Backup first
cp ~/.jarvis/task_queue.json ~/.jarvis/task_queue.json.bak.$(date +%s)

# Reset queue
rm ~/.jarvis/task_queue.json
jarvis restart

# Note: All pending tasks will be lost
```

---

## 4. Alert: Database Connection Failed

### Severity: CRITICAL

### Symptoms

- Cannot read iMessage conversations
- `iMessageAccessError` in logs
- `MSG_ACCESS_DENIED` error code

### Diagnosis

```bash
# Check permissions
jarvis setup --check-permissions

# Test database access
sqlite3 ~/Library/Messages/chat.db "SELECT COUNT(*) FROM message;" 2>&1

# Check database file exists and is readable
ls -la ~/Library/Messages/chat.db
file ~/Library/Messages/chat.db

# Check if database is locked
lsof ~/Library/Messages/chat.db 2>/dev/null || echo "Not locked"

# Check iMessage process
pgrep -x "Messages" > /dev/null && echo "iMessage running" || echo "iMessage not running"
```

### Resolution

**Step 1: Permission Issue**

```bash
# Re-request Full Disk Access
echo "Opening System Settings..."
open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"

echo "Please grant Full Disk Access to Terminal/Terminal emulator"
echo "Then press Enter to continue..."
read

# Verify
jarvis setup --check-permissions
```

**Step 2: Database Locked**

```bash
# Wait for iMessage to release lock
attempts=0
while lsof ~/Library/Messages/chat.db 2>/dev/null | grep -q "Messages"; do
    attempts=$((attempts + 1))
    if [[ $attempts -gt 30 ]]; then
        echo "Lock not released after 5 minutes"
        echo "Consider closing iMessage app temporarily"
        break
    fi
    echo "Waiting for database lock release... ($attempts/30)"
    sleep 10
done

# Verify access
jarvis db ping
```

**Step 3: Database Corruption**

```bash
# Run integrity check
sqlite3 ~/Library/Messages/chat.db "PRAGMA integrity_check;"

# Check schema version
sqlite3 ~/Library/Messages/chat.db "SELECT * FROM _SqliteDatabaseProperties WHERE key = 'schema_version';"

# If corrupted, there's no recovery for iMessage DB
# User must restore from Time Machine
```

---

## 5. Alert: Rate Limit Exceeded

### Severity: WARNING

### Symptoms

- HTTP 429 responses
- `Retry-After` header in responses
- Client complaints about throttling

### Diagnosis

```bash
# Check rate limit status
curl -s http://localhost:8742/metrics | grep ratelimit

# Check recent request volume
jarvis logs --since 5m | grep -c "request"

# Identify top clients
jarvis logs --since 10m | grep "request" | awk '{print $4}' | sort | uniq -c | sort -nr | head -10
```

### Resolution

**Step 1: If Legitimate Traffic**

```bash
# Temporarily increase limits
curl -X POST http://localhost:8742/admin/ratelimit \
  -H "Content-Type: application/json" \
  -d '{"generation": "20/minute", "read": "120/minute"}'

# Monitor
watch -n 5 'curl -s http://localhost:8742/metrics | grep ratelimit'
```

**Step 2: If Abuse/Loop**

```bash
# Identify problematic client
jarvis logs --since 10m | grep "request" | awk '{print $4}' | sort | uniq -c | sort -nr | head -5

# Block client (if necessary)
curl -X POST http://localhost:8742/admin/block \
  -H "Content-Type: application/json" \
  -d '{"client": "<client_id>", "duration_minutes": 60}'
```

**Step 3: Permanent Adjustment**

```bash
# Update config
javis config set rate_limit.generation "15/minute"
javis config set rate_limit.read "90/minute"
```

---

## 6. Full System Recovery

### Severity: CRITICAL (Complete Outage)

### When to Use

- JARVIS completely unresponsive
- Multiple component failures
- Data corruption suspected

### Recovery Procedure

```bash
#!/bin/bash
# full_recovery.sh

set -e

echo "=== JARVIS Full System Recovery ==="
echo "Start time: $(date)"

# Step 1: Stop all services
echo "[1/10] Stopping services..."
javis stop 2>/dev/null || true
pkill -f jarvis 2>/dev/null || true
sleep 5

# Step 2: Check for zombie processes
echo "[2/10] Checking for zombie processes..."
if pgrep -f jarvis > /dev/null; then
    echo "Force killing remaining processes..."
    pkill -9 -f jarvis 2>/dev/null || true
    sleep 2
fi

# Step 3: Clear temporary files
echo "[3/10] Clearing temporary files..."
rm -rf ~/.jarvis/temp/*
rm -f ~/.jarvis/*.lock

# Step 4: Backup current state
echo "[4/10] Backing up current state..."
backup_dir="~/.jarvis/backups/recovery_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
cp ~/.jarvis/task_queue.json "$backup_dir/" 2>/dev/null || true
cp ~/.jarvis/scheduler_queue.json "$backup_dir/" 2>/dev/null || true
cp ~/.jarvis/config.yaml "$backup_dir/" 2>/dev/null || true

# Step 5: Verify permissions
echo "[5/10] Verifying permissions..."
if ! jarvis setup --check-permissions --quiet; then
    echo "WARNING: Permission issues detected"
    echo "Run: jarvis setup"
fi

# Step 6: Clear corrupted caches (optional)
read -p "Clear all caches? (y/N): " clear_caches
if [[ $clear_caches == "y" ]]; then
    echo "[6/10] Clearing caches..."
    rm -rf ~/.jarvis/cache/*
    rm -rf ~/.jarvis/embedding_cache/*
fi

# Step 7: Reset queues (optional)
read -p "Reset task queues? All pending tasks will be lost (y/N): " reset_queues
if [[ $reset_queues == "y" ]]; then
    echo "[7/10] Resetting queues..."
    rm -f ~/.jarvis/task_queue.json
    rm -f ~/.jarvis/scheduler_queue.json
fi

# Step 8: Rebuild indexes (if database issues suspected)
read -p "Rebuild search indexes? (y/N): " rebuild_indexes
if [[ $rebuild_indexes == "y" ]]; then
    echo "[8/10] Rebuilding indexes..."
    jarvis db rebuild-indexes
fi

# Step 9: Start services
echo "[9/10] Starting services..."
jarvis start --wait

# Step 10: Verify
echo "[10/10] Verifying recovery..."
sleep 5

if jarvis health --quiet; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed - check logs"
    jarvis health --verbose
    exit 1
fi

# Test basic functionality
echo "Testing basic functionality..."
if jarvis search-messages --limit 1 "test" > /dev/null 2>&1; then
    echo "✓ Search functional"
else
    echo "✗ Search failed"
fi

echo ""
echo "=== Recovery Complete ==="
echo "End time: $(date)"
echo "Backup location: $backup_dir"
```

---

## 7. Post-Incident Review Template

```markdown
## Post-Incident Review: INC-YYYY-MM-DD-XXX

### Summary

One-line description of the incident.

### Timeline (All times local)

| Time  | Event                          | Owner   |
| ----- | ------------------------------ | ------- |
| HH:MM | Alert fired                    | System  |
| HH:MM | Investigation started          | @oncall |
| HH:MM | Root cause identified          | @oncall |
| HH:MM | Mitigation applied             | @oncall |
| HH:MM | Service restored               | @oncall |
| HH:MM | Post-incident review completed | @oncall |

### Impact

- **Duration:** X minutes
- **Severity:** [P0/P1/P2/P3]
- **Features Affected:** List affected features
- **Users Affected:** Approximate count
- **Data Loss:** Yes/No - details if yes

### Root Cause

Detailed explanation of why the incident occurred.

### Resolution

Steps taken to resolve the incident.

### What Went Well

- Item 1
- Item 2

### What Could Be Improved

- Item 1
- Item 2

### Action Items

| ID  | Action | Owner | Due Date |
| --- | ------ | ----- | -------- |
| 1   |        |       |          |
| 2   |        |       |          |

### Related Links

- Alert: [link]
- Dashboard: [link]
- Logs: [link]
```

---

## 8. Preventive Maintenance

### Daily Checks

```bash
#!/bin/bash
# daily_health_check.sh

echo "=== JARVIS Daily Health Check ==="
echo "Date: $(date)"

# 1. Basic health
echo "[1/5] Health check..."
jarvis health --quiet || exit 1

# 2. Circuit breaker status
echo "[2/5] Circuit breakers..."
open_circuits=$(curl -s http://localhost:8742/circuits | jq '[.circuits | to_entries[] | select(.value.state == "open")] | length')
if [[ $open_circuits -gt 0 ]]; then
    echo "WARNING: $open_circuits circuit(s) open"
fi

# 3. Queue status
echo "[3/5] Queue status..."
pending=$(jarvis tasks stats --json | jq '.by_status.pending // 0')
if [[ $pending -gt 50 ]]; then
    echo "WARNING: $pending pending tasks"
fi

# 4. Memory
echo "[4/5] Memory check..."
mem_pct=$(python3 -c "import psutil; print(psutil.virtual_memory().percent)")
if (( $(echo "$mem_pct > 85" | bc -l) )); then
    echo "WARNING: Memory usage at ${mem_pct}%"
fi

# 5. Disk space
echo "[5/5] Disk space..."
disk_pct=$(df ~/.jarvis | tail -1 | awk '{print $5}' | tr -d '%')
if [[ $disk_pct -gt 85 ]]; then
    echo "WARNING: Disk usage at ${disk_pct}%"
fi

echo "=== Check Complete ==="
```

### Weekly Maintenance

```bash
#!/bin/bash
# weekly_maintenance.sh

echo "=== JARVIS Weekly Maintenance ==="

# 1. Clear old logs
echo "[1/5] Clearing old logs..."
find ~/.jarvis/logs -name "*.log" -mtime +7 -delete

# 2. Clear old completed tasks
echo "[2/5] Clearing old completed tasks..."
jarvis tasks clear-completed --older-than 7d

# 3. Verify backups
echo "[3/5] Verifying backups..."
ls -la ~/.jarvis/backups/ | tail -5

# 4. Update model cache
echo "[4/5] Updating model cache..."
jarvis models verify

# 5. Performance check
echo "[5/5] Performance check..."
jarvis benchmark quick

echo "=== Maintenance Complete ==="
```

---

## Appendix: Quick Commands

```bash
# Health and Status
jarvis health                    # Basic health check
jarvis health --verbose          # Detailed health
jarvis health --json             # JSON output

# Circuit Breakers
curl http://localhost:8742/circuits
curl -X POST http://localhost:8742/circuits/{name}/reset

# Tasks
jarvis tasks list
jarvis tasks stats
jarvis tasks cancel <id>
jarvis tasks clear-completed

# Models
jarvis models status
jarvis models unload
jarvis models reload
jarvis models verify

# Database
jarvis db ping
jarvis db check
jarvis db rebuild-indexes

# Cache
jarvis cache clear --all
jarvis cache stats

# Logs
jarvis logs --follow
jarvis logs --since 1h
jarvis logs --level error

# Metrics
curl http://localhost:8742/metrics
curl http://localhost:8742/health
```

---

_End of Runbook_
