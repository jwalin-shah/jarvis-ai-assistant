# JARVIS API Reference

Complete REST API documentation for the JARVIS iMessage Assistant backend.

**Base URL:** `http://localhost:8000` (default) or `http://localhost:8742` (common dev setup)

**Documentation URLs:** (when server is running)
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

---

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Error Responses](#error-responses)
- [Endpoints](#endpoints)
  - [Health](#health)
  - [Conversations](#conversations)
  - [Contacts](#contacts)
  - [Search](#search)
  - [Drafts (AI Generation)](#drafts-ai-generation)
  - [Suggestions (Quick Replies)](#suggestions-quick-replies)
  - [Threads](#threads)
  - [Topics](#topics)
  - [Statistics](#statistics)
  - [Insights](#insights)
  - [Priority Inbox](#priority-inbox)
  - [Calendar](#calendar)
  - [Attachments](#attachments)
  - [Embeddings](#embeddings)
  - [Relationships](#relationships)
  - [Feedback](#feedback)
  - [Experiments (A/B Testing)](#experiments-ab-testing)
  - [Quality Metrics](#quality-metrics)
  - [Digests](#digests)
  - [Export](#export)
  - [PDF Export](#pdf-export)
  - [Batch Operations](#batch-operations)
  - [Tasks (Async Queue)](#tasks-async-queue)
  - [Custom Templates](#custom-templates)
  - [Template Analytics](#template-analytics)
  - [Metrics](#metrics)
  - [Settings](#settings)
  - [WebSocket](#websocket)

---

## Authentication

This API is designed for local use by the JARVIS desktop application. **No authentication is required** as the API only binds to localhost by default.

## Rate Limiting

Rate limiting is applied to protect system resources:

| Endpoint Type | Limit |
|---------------|-------|
| Read endpoints (GET) | 60 requests/minute |
| Write endpoints (POST, PUT, DELETE) | 30 requests/minute |
| AI generation endpoints | 10 requests/minute |

Exceeding limits returns HTTP 429 with a `Retry-After` header.

## Error Responses

All errors return JSON with this structure:

```json
{
  "error": "Brief error message",
  "detail": "Detailed explanation of the error",
  "code": "MACHINE_READABLE_CODE"
}
```

### Common Error Codes

| HTTP Status | Code | Description |
|-------------|------|-------------|
| 400 | `VALIDATION_ERROR` | Invalid request parameters |
| 404 | `NOT_FOUND` | Resource not found |
| 429 | `RATE_LIMITED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Server error |
| 503 | `MODEL_UNAVAILABLE` | ML model not loaded |

---

## Endpoints

### Health

System health monitoring and status.

#### GET /health

Get comprehensive system health status.

**Response:**
```json
{
  "status": "healthy",
  "memory": {
    "available_mb": 6234,
    "used_mb": 1766,
    "mode": "FULL",
    "pressure": "normal"
  },
  "features": {
    "chat": {"status": "healthy", "details": "OK"},
    "imessage": {"status": "healthy", "details": "OK"}
  },
  "model": {
    "loaded": false,
    "id": "qwen-1.5b"
  }
}
```

#### GET /health/detailed

Get detailed health information including circuit breaker states.

---

### Conversations

Access iMessage conversations and messages.

#### GET /conversations

List recent conversations sorted by last message date.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Maximum conversations to return |
| `offset` | integer | 0 | Pagination offset |

**Response:**
```json
{
  "conversations": [
    {
      "chat_id": "chat123456",
      "display_name": "John Doe",
      "last_message": "Hey, are you free tomorrow?",
      "last_message_date": "2024-01-15T18:30:00Z",
      "unread_count": 2,
      "is_group": false,
      "participant_count": 2
    }
  ],
  "total": 150,
  "has_more": true
}
```

#### GET /conversations/{chat_id}/messages

Get messages for a specific conversation.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `chat_id` | string | Conversation identifier |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Maximum messages to return |
| `before` | datetime | null | Return messages before this date |
| `after` | datetime | null | Return messages after this date |

**Response:**
```json
{
  "messages": [
    {
      "id": "msg123",
      "text": "Hey, are you free tomorrow?",
      "sender": "John Doe",
      "is_from_me": false,
      "date": "2024-01-15T18:30:00Z",
      "attachments": [],
      "reactions": []
    }
  ],
  "chat_id": "chat123456",
  "total": 500,
  "has_more": true
}
```

#### GET /conversations/{chat_id}/context

Get conversation context around a specific message (for RAG).

#### POST /conversations/{chat_id}/mark-read

Mark a conversation as read.

#### POST /conversations/{chat_id}/archive

Archive a conversation.

---

### Contacts

Access contact information.

#### GET /contacts

List all contacts with iMessage conversations.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 100 | Maximum contacts to return |
| `search` | string | null | Filter by name |

**Response:**
```json
{
  "contacts": [
    {
      "id": "contact123",
      "name": "John Doe",
      "phone": "+1234567890",
      "email": "john@example.com",
      "avatar_url": null,
      "last_contacted": "2024-01-15T18:30:00Z"
    }
  ]
}
```

#### GET /contacts/{contact_id}

Get detailed information for a specific contact.

---

### Search

Search through messages.

#### POST /search/semantic

Perform semantic search using AI embeddings.

**Request:**
```json
{
  "query": "dinner plans next week",
  "limit": 20,
  "chat_ids": ["chat123"],
  "date_range": {
    "start": "2024-01-01",
    "end": "2024-01-31"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "message_id": "msg123",
      "text": "Want to grab dinner next Tuesday?",
      "sender": "John",
      "date": "2024-01-10T18:30:00Z",
      "chat_id": "chat123",
      "score": 0.92
    }
  ],
  "query": "dinner plans next week",
  "total": 5
}
```

#### GET /search/history

Get recent search history.

#### DELETE /search/history

Clear search history.

---

### Drafts (AI Generation)

AI-powered message generation.

#### POST /drafts/reply

Generate AI-powered reply suggestions.

**Request:**
```json
{
  "chat_id": "chat123456",
  "instruction": "accept politely but ask about timing",
  "tone": "casual",
  "max_suggestions": 3
}
```

**Response:**
```json
{
  "suggestions": [
    {
      "id": "draft_1",
      "text": "Sounds great! What time works best for you?",
      "confidence": 0.89,
      "tone": "casual"
    },
    {
      "id": "draft_2",
      "text": "I'd love to! When were you thinking?",
      "confidence": 0.85,
      "tone": "casual"
    }
  ],
  "context": {
    "last_message": "Want to grab coffee tomorrow?",
    "sender": "John"
  },
  "generation_time_ms": 450
}
```

#### POST /drafts/summarize

Generate a conversation summary.

**Request:**
```json
{
  "chat_id": "chat123456",
  "message_count": 50,
  "style": "bullet_points"
}
```

**Response:**
```json
{
  "summary": "• Discussed project deadline moved to Friday\n• Scheduled coffee meeting for Tuesday\n• John recommended a new restaurant",
  "message_count": 50,
  "date_range": {
    "start": "2024-01-10T00:00:00Z",
    "end": "2024-01-15T18:30:00Z"
  },
  "generation_time_ms": 800
}
```

---

### Suggestions (Quick Replies)

Fast pattern-based suggestions without AI model invocation.

#### POST /suggestions

Get quick reply suggestions.

**Request:**
```json
{
  "chat_id": "chat123456",
  "context_messages": 5
}
```

**Response:**
```json
{
  "suggestions": [
    {"text": "Sounds good!", "category": "agreement"},
    {"text": "Let me check and get back to you", "category": "deferral"},
    {"text": "Thanks!", "category": "gratitude"}
  ],
  "matched_template": "scheduling_response",
  "is_group_chat": false
}
```

---

### Threads

Conversation thread management.

#### GET /threads

List conversation threads with smart grouping.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chat_id` | string | required | Conversation ID |
| `limit` | integer | 20 | Maximum threads |

**Response:**
```json
{
  "threads": [
    {
      "id": "thread_1",
      "topic": "Project Discussion",
      "message_count": 15,
      "participants": ["John", "Sarah"],
      "last_activity": "2024-01-15T18:30:00Z",
      "summary": "Discussing Q1 deliverables"
    }
  ]
}
```

#### GET /threads/{thread_id}

Get a specific thread with all messages.

#### POST /threads/detect

Detect and create threads from conversation messages.

---

### Topics

Topic/label management for conversations.

#### POST /topics/analyze

Analyze and extract topics from a conversation.

**Request:**
```json
{
  "chat_id": "chat123456",
  "message_count": 100
}
```

**Response:**
```json
{
  "topics": [
    {"name": "Work", "confidence": 0.85, "message_count": 45},
    {"name": "Social", "confidence": 0.72, "message_count": 30}
  ]
}
```

---

### Statistics

Conversation and messaging statistics.

#### GET /stats

Get comprehensive messaging statistics.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | "30d" | Time period (7d, 30d, 90d, all) |

**Response:**
```json
{
  "total_messages": 15420,
  "total_conversations": 89,
  "messages_sent": 7234,
  "messages_received": 8186,
  "active_days": 28,
  "average_response_time_minutes": 12.5,
  "busiest_hour": 14,
  "top_contacts": [
    {"name": "John", "message_count": 1250}
  ]
}
```

#### DELETE /stats/cache

Clear statistics cache.

---

### Insights

AI-generated conversation insights.

#### GET /insights

Get AI-generated insights about messaging patterns.

**Response:**
```json
{
  "insights": [
    {
      "type": "response_pattern",
      "title": "Quick Responder",
      "description": "You typically respond within 10 minutes to John",
      "confidence": 0.88
    }
  ]
}
```

#### GET /insights/communication-style

Get analysis of your communication style.

#### GET /insights/relationship/{contact_id}

Get insights about a specific relationship.

#### GET /insights/trends

Get trending topics and patterns.

#### GET /insights/action-items

Get pending action items extracted from conversations.

#### DELETE /insights/cache

Clear insights cache.

---

### Priority Inbox

Smart inbox prioritization.

#### GET /priority/inbox

Get prioritized inbox with smart sorting.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 20 | Maximum items |

**Response:**
```json
{
  "items": [
    {
      "chat_id": "chat123",
      "contact_name": "Boss",
      "priority_score": 0.95,
      "reason": "Urgent keyword detected",
      "last_message": "Need this ASAP",
      "requires_response": true
    }
  ]
}
```

#### POST /priority/rules

Add a priority rule.

#### DELETE /priority/rules/{rule_id}

Delete a priority rule.

#### POST /priority/snooze/{chat_id}

Snooze a conversation.

#### GET /priority/snoozed

Get snoozed conversations.

#### POST /priority/refresh

Refresh priority calculations.

---

### Calendar

Calendar integration for event detection.

#### GET /calendar/events

Get upcoming events extracted from messages.

#### GET /calendar/events/{event_id}

Get a specific event.

#### GET /calendar/today

Get today's events.

#### POST /calendar/detect

Detect events in recent messages.

#### POST /calendar/sync

Sync detected events to system calendar.

#### POST /calendar/remind

Set a reminder for an event.

#### POST /calendar/export

Export events to ICS format.

---

### Attachments

Attachment management.

#### GET /attachments/stats/{chat_id}

Get attachment statistics for a conversation.

**Response:**
```json
{
  "total_count": 150,
  "total_size_mb": 245.5,
  "by_type": {
    "image": {"count": 100, "size_mb": 180.2},
    "video": {"count": 20, "size_mb": 50.0},
    "document": {"count": 30, "size_mb": 15.3}
  }
}
```

#### GET /attachments/storage/summary

Get storage usage summary across all conversations.

#### GET /attachments/{chat_id}

List attachments for a conversation.

#### GET /attachments/{chat_id}/timeline

Get attachments organized by date.

#### GET /attachments/{attachment_id}/download

Download a specific attachment.

---

### Embeddings

Semantic embedding management.

#### POST /embeddings/index

Index messages for semantic search.

**Request:**
```json
{
  "chat_ids": ["chat123", "chat456"],
  "force_reindex": false
}
```

**Response:**
```json
{
  "indexed_count": 500,
  "duration_ms": 2500,
  "status": "completed"
}
```

#### GET /embeddings/search

Search using embeddings (alias for /search/semantic).

#### GET /embeddings/relationship/{contact_id}

Get relationship profile based on communication embeddings.

#### GET /embeddings/stats

Get embedding index statistics.

---

### Relationships

Relationship learning and communication profiles.

#### GET /relationships/profile/{contact_id}

Get learned communication profile for a contact.

**Response:**
```json
{
  "contact_id": "contact123",
  "name": "John Doe",
  "communication_style": {
    "formality": "casual",
    "emoji_usage": "moderate",
    "average_message_length": 45,
    "response_time_preference": "quick"
  },
  "topics": ["work", "sports", "restaurants"],
  "last_updated": "2024-01-15T00:00:00Z"
}
```

#### GET /relationships/style-guide/{contact_id}

Get natural language style guide for a contact.

**Response:**
```json
{
  "contact_id": "contact123",
  "guide": "When messaging John, keep it casual and use occasional emojis. He prefers quick, direct messages and responds well to humor. Avoid overly formal language."
}
```

#### POST /relationships/refresh/{contact_id}

Force refresh of relationship profile.

#### DELETE /relationships/profile/{contact_id}

Delete a relationship profile.

---

### Feedback

User feedback for AI improvement.

#### POST /feedback/record

Record feedback on a suggestion.

**Request:**
```json
{
  "suggestion_id": "draft_1",
  "action": "sent",
  "edited_text": null,
  "rating": 5
}
```

#### GET /feedback/stats

Get overall feedback statistics.

**Response:**
```json
{
  "total_suggestions": 500,
  "sent_count": 350,
  "edited_count": 100,
  "dismissed_count": 50,
  "acceptance_rate": 0.70,
  "average_rating": 4.2
}
```

#### GET /feedback/improvements

Get AI-generated suggestions for improving prompts.

#### GET /feedback/by-contact/{contact_id}

Get feedback statistics for a specific contact.

#### POST /feedback/export

Export feedback data.

---

### Experiments (A/B Testing)

Prompt experimentation infrastructure.

#### GET /experiments/list

List all experiments.

**Response:**
```json
{
  "experiments": [
    {
      "name": "casual_tone_v2",
      "status": "active",
      "variants": ["control", "treatment"],
      "start_date": "2024-01-01",
      "sample_size": 100
    }
  ]
}
```

#### GET /experiments/{experiment_name}

Get experiment details.

#### GET /experiments/{experiment_name}/results

Get experiment results with statistical analysis.

#### POST /experiments

Create a new experiment.

#### POST /experiments/{experiment_name}/record

Record a data point for an experiment.

#### PUT /experiments/{experiment_name}

Update experiment configuration.

#### DELETE /experiments/{experiment_name}

Delete an experiment.

#### DELETE /experiments/{experiment_name}/data

Clear experiment data.

#### GET /experiments/active

Get only active experiments.

---

### Quality Metrics

Quality monitoring dashboard.

#### GET /quality/dashboard

Get comprehensive quality dashboard.

**Response:**
```json
{
  "overall_score": 0.85,
  "hhem_score": 0.72,
  "acceptance_rate": 0.70,
  "average_latency_ms": 450,
  "trends": {
    "7d": {"score": 0.83, "change": 0.02}
  },
  "by_intent": [
    {"intent": "reply", "score": 0.88, "count": 200}
  ]
}
```

#### GET /quality/summary

Get quality summary metrics.

#### GET /quality/trends

Get quality trends over time.

#### GET /quality/contact/{contact_id}

Get quality metrics for a specific contact.

#### GET /quality/contacts

Get quality metrics for all contacts.

#### GET /quality/time-of-day

Get quality metrics by time of day.

#### GET /quality/by-intent

Get quality metrics by intent type.

#### GET /quality/by-conversation-type

Get quality metrics by conversation type (1:1 vs group).

#### GET /quality/recommendations

Get AI recommendations for improving quality.

#### POST /quality/record/response

Record a response quality data point.

#### POST /quality/record/acceptance

Record an acceptance event.

#### POST /quality/reset

Reset quality metrics.

---

### Digests

Daily/weekly message digests.

#### POST /digest/generate

Generate a digest.

**Request:**
```json
{
  "period": "daily",
  "include_action_items": true,
  "include_summary": true
}
```

**Response:**
```json
{
  "period": "daily",
  "date_range": {
    "start": "2024-01-14T00:00:00Z",
    "end": "2024-01-15T00:00:00Z"
  },
  "summary": "You had 45 messages across 12 conversations...",
  "action_items": [
    {"text": "Reply to John about dinner", "priority": "high"}
  ],
  "highlights": [
    {"contact": "Mom", "topic": "Birthday planning"}
  ]
}
```

#### POST /digest/export

Export digest to Markdown or HTML.

#### GET /digest/preferences

Get digest preferences.

#### PUT /digest/preferences

Update digest preferences.

#### GET /digest/daily

Get or generate today's daily digest.

#### GET /digest/weekly

Get or generate this week's digest.

---

### Export

Conversation export functionality.

#### POST /export/conversation/{chat_id}

Export a conversation.

**Request:**
```json
{
  "format": "json",
  "limit": 1000,
  "include_attachments": false
}
```

**Response:**
```json
{
  "filename": "conversation_chat123_20240115.json",
  "path": "/Users/user/.jarvis/exports/conversation_chat123_20240115.json",
  "format": "json",
  "message_count": 500,
  "size_bytes": 125000
}
```

#### POST /export/search

Export search results.

#### POST /export/backup

Create a full backup of all conversations.

---

### PDF Export

Export conversations to PDF.

#### POST /pdf/{chat_id}

Generate PDF export of a conversation.

**Request:**
```json
{
  "limit": 100,
  "include_attachments": true,
  "page_size": "letter"
}
```

#### POST /pdf/{chat_id}/download

Download the generated PDF.

---

### Batch Operations

Bulk operations for efficiency.

#### POST /batch/export

Batch export multiple conversations.

**Request:**
```json
{
  "chat_ids": ["chat1", "chat2", "chat3"],
  "format": "json"
}
```

#### POST /batch/export/all

Export all conversations.

#### POST /batch/summarize

Generate summaries for multiple conversations.

#### POST /batch/summarize/recent

Summarize all recent conversations.

#### POST /batch/generate-replies

Generate replies for multiple conversations.

---

### Tasks (Async Queue)

Asynchronous task management.

#### POST /tasks

Create a new async task.

**Request:**
```json
{
  "type": "batch_export",
  "params": {
    "chat_ids": ["chat1", "chat2"]
  }
}
```

**Response:**
```json
{
  "task_id": "task_abc123",
  "status": "pending",
  "created_at": "2024-01-15T18:30:00Z"
}
```

#### GET /tasks

List all tasks.

#### GET /tasks/stats

Get task queue statistics.

#### GET /tasks/{task_id}

Get task status and result.

#### DELETE /tasks/{task_id}

Cancel a task.

#### POST /tasks/{task_id}/retry

Retry a failed task.

#### POST /tasks/worker/start

Start the background worker.

#### POST /tasks/worker/stop

Stop the background worker.

#### DELETE /tasks/completed/clear

Clear completed tasks.

---

### Custom Templates

User-defined response templates.

#### GET /templates

List all custom templates.

**Response:**
```json
{
  "templates": [
    {
      "id": "template_1",
      "name": "Quick Decline",
      "pattern": "decline.*politely",
      "response": "Thanks for thinking of me, but I'll have to pass this time!",
      "category": "social",
      "usage_count": 25
    }
  ]
}
```

#### POST /templates

Create a new template.

#### GET /templates/{template_id}

Get a specific template.

#### PUT /templates/{template_id}

Update a template.

#### DELETE /templates/{template_id}

Delete a template.

#### GET /templates/categories

List template categories.

#### POST /templates/match

Test template matching against a query.

#### POST /templates/import

Import templates from JSON.

#### POST /templates/export

Export templates to JSON.

---

### Template Analytics

Analytics for template usage.

#### GET /template-analytics

Get overall template analytics.

#### GET /template-analytics/top

Get most used templates.

#### GET /template-analytics/missed

Get queries that didn't match any template.

#### GET /template-analytics/categories

Get analytics by category.

#### GET /template-analytics/templates

Get per-template analytics.

#### GET /template-analytics/coverage

Get template coverage statistics.

#### GET /template-analytics/export

Export analytics data.

#### POST /template-analytics/reset

Reset analytics data.

#### GET /template-analytics/dashboard

Get analytics dashboard data.

---

### Metrics

Prometheus-compatible metrics.

#### GET /metrics

Get Prometheus-formatted metrics.

**Response:**
```text
# HELP jarvis_requests_total Total requests
# TYPE jarvis_requests_total counter
jarvis_requests_total{endpoint="/health"} 150
jarvis_memory_usage_bytes 1850000000
jarvis_generation_latency_seconds_bucket{le="0.5"} 45
```

#### GET /metrics/memory

Get memory metrics.

**Response:**
```json
{
  "rss_mb": 1850,
  "vms_mb": 3200,
  "available_mb": 6234,
  "percent_used": 22.5
}
```

#### GET /metrics/latency

Get latency metrics.

#### GET /metrics/requests

Get request count metrics.

#### POST /metrics/gc

Trigger garbage collection.

#### POST /metrics/sample

Record a metrics sample.

#### POST /metrics/reset

Reset metrics counters.

---

### Settings

Application settings management.

#### GET /settings

Get current settings.

**Response:**
```json
{
  "model": {
    "model_id": "qwen-1.5b",
    "auto_select": true,
    "temperature": 0.7
  },
  "ui": {
    "theme": "system",
    "font_size": 14
  },
  "search": {
    "default_limit": 50
  }
}
```

#### PUT /settings

Update settings.

#### GET /settings/schema

Get settings JSON schema.

#### POST /settings/reset

Reset settings to defaults.

#### POST /settings/export

Export settings to file.

---

### WebSocket

Real-time updates via WebSocket.

#### WS /ws

WebSocket endpoint for real-time events.

**Connection:** `ws://localhost:8000/ws`

**Message Types:**

```json
// New message notification
{
  "type": "new_message",
  "data": {
    "chat_id": "chat123",
    "message_id": "msg456",
    "sender": "John",
    "text": "Hey!"
  }
}

// Generation complete
{
  "type": "generation_complete",
  "data": {
    "request_id": "req123",
    "suggestions": [...]
  }
}

// Health update
{
  "type": "health_update",
  "data": {
    "memory_mode": "FULL",
    "model_loaded": true
  }
}
```

---

## Privacy

All data processing happens locally on your Mac. No conversation data, messages, or personal information is ever sent to external servers. The MLX language models run entirely on Apple Silicon.

---

## Example: Complete Workflow

```bash
# 1. Check health
curl http://localhost:8000/health

# 2. List conversations
curl http://localhost:8000/conversations?limit=10

# 3. Search messages
curl -X POST http://localhost:8000/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "dinner plans", "limit": 5}'

# 4. Generate reply
curl -X POST http://localhost:8000/drafts/reply \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "chat123", "tone": "casual"}'

# 5. Record feedback
curl -X POST http://localhost:8000/feedback/record \
  -H "Content-Type: application/json" \
  -d '{"suggestion_id": "draft_1", "action": "sent"}'
```

---

For interactive API exploration, start the server with `jarvis serve` and visit http://localhost:8000/docs for Swagger UI.
