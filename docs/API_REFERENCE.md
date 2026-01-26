# JARVIS API Reference

Complete REST API documentation for the JARVIS iMessage Assistant backend.

**Base URL:** `http://localhost:8742`

**Documentation URLs:**
- Swagger UI: http://localhost:8742/docs
- ReDoc: http://localhost:8742/redoc
- OpenAPI JSON: http://localhost:8742/openapi.json

---

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Error Responses](#error-responses)
- [Endpoints](#endpoints)
  - [Health](#health)
  - [iMessage Data](#imessage-data)
    - [Conversations](#conversations)
    - [Contacts](#contacts)
    - [Topics](#topics)
    - [Statistics](#statistics)
    - [Insights](#insights)
    - [Priority Inbox](#priority-inbox)
    - [Calendars](#calendars)
  - [AI Generation & Search](#ai-generation--search)
    - [Drafts](#drafts-ai-generation)
    - [Suggestions](#suggestions-quick-replies)
    - [Semantic Search](#semantic-search)
  - [System Operations](#system-operations)
    - [Export](#export)
    - [PDF Export](#pdf-export)
    - [Batch Operations](#batch-operations)
    - [Task Queue](#task-queue)
  - [Management](#management)
    - [Metrics](#metrics)
    - [Template Analytics](#template-analytics)
    - [Settings](#settings)
    - [WebSocket](#websocket)

---

## Authentication

This API is designed for local use by the JARVIS desktop application. **No authentication is required** as the API only binds to localhost.

## Rate Limiting

Rate limiting is applied to protect system resources and ensure stability during heavy AI generation tasks.

- **Read endpoints** (GET): 60 requests per minute
- **Write endpoints** (POST, PUT, DELETE): 30 requests per minute
- **Generation endpoints** (AI-powered): 10 requests per minute

Exceeding these limits returns HTTP 429 with a `Retry-After` header.

## Error Responses

All errors return a JSON response with the following structure:

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
| 403 | `PERMISSION_DENIED` | Full Disk Access or Calendar permission not granted |
| 404 | `NOT_FOUND` | Resource not found |
| 408 | `REQUEST_TIMEOUT` | Generation task took too long |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Server error |
| 503 | `SERVICE_UNAVAILABLE` | Model not loaded or system resource issues |

---

## Endpoints

### Health

System health monitoring and status checks.

---

#### GET /

Root endpoint - simple health ping.

**Response:**
```json
{
    "status": "ok",
    "service": "jarvis-api"
}
```

---

#### GET /health

Get comprehensive system health status.

**Response:**
```json
{
    "status": "healthy",
    "imessage_access": true,
    "memory_available_gb": 12.5,
    "memory_used_gb": 3.5,
    "memory_mode": "FULL",
    "model_loaded": true,
    "permissions_ok": true,
    "jarvis_rss_mb": 256.5,
    "jarvis_vms_mb": 1024.0,
    "model": {
        "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "display_name": "Qwen 0.5B (Fast)",
        "loaded": true,
        "memory_usage_mb": 450.5,
        "quality_tier": "basic"
    },
    "recommended_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "system_ram_gb": 16.0
}
```

---

### iMessage Data

#### Conversations

---

#### GET /conversations

List recent conversations sorted by last message date.

**Query Parameters:**
- `limit` (int): Max conversations (1-500, default 50)
- `since` (datetime): Only convos with messages after this date
- `before` (datetime): Pagination cursor

---

#### GET /conversations/{chat_id}

Get detailed metadata for a single conversation.

---

#### GET /conversations/{chat_id}/messages

Get messages for a specific conversation.

**Query Parameters:**
- `limit` (int): Max messages (1-1000, default 100)
- `before` (datetime): Only messages before this date

---

#### GET /conversations/search

Keyword search messages across all conversations.

---

#### POST /conversations/{chat_id}/send

Send a text message to a conversation.

---

#### POST /conversations/{chat_id}/send-attachment

Send a file attachment to a conversation.

---

#### Contacts

---

#### GET /contacts

List all iMessage contacts.

---

#### GET /contacts/{identifier}

Get details for a specific contact.

---

#### Topics

---

#### POST /conversations/{chat_id}/topics

Detect and retrieve topics for a conversation.

---

#### Statistics

---

#### GET /stats/{chat_id}

Get messaging patterns and activity statistics for a conversation.

**Query Parameters:**
- `time_range` (string): `week`, `month`, `three_months`, `all_time` (default `month`)
- `limit` (int): Max messages to analyze (default 500)

---

#### Insights

Advanced sentiment and relationship analytics.

---

#### GET /insights/{chat_id}

Get comprehensive relationship insights.

---

#### GET /insights/{chat_id}/sentiment

Get sentiment trends over time.

---

#### GET /insights/{chat_id}/health

Get relationship health scores.

---

#### Priority Inbox

---

#### GET /priority

Get messages prioritized by urgency and importance.

**Query Parameters:**
- `limit` (int): Max messages (default 50)
- `min_level` (string): `critical`, `high`, `medium`, `low`

---

#### POST /priority/handled

Mark a prioritized message as handled.

---

#### Calendars

---

#### GET /calendars

List available macOS calendars.

---

#### GET /calendars/events

List upcoming calendar events.

---

#### POST /calendars/detect

Detect potential events in arbitrary text.

---

#### POST /calendars/events

Create a new calendar event.

---

### AI Generation & Search

#### Drafts (AI Generation)

---

#### POST /drafts/reply

Generate AI-powered reply suggestions.

---

#### POST /drafts/summarize

Summarize a conversation using AI.

---

#### Suggestions (Quick Replies)

---

#### POST /suggestions

Fast pattern-based reply suggestions (no AI model required).

---

#### Semantic Search

AI-powered search based on meaning rather than keywords.

---

#### POST /search/semantic

Perform a semantic search.

**Request Body:**
```json
{
    "query": "when did we talk about the project?",
    "limit": 20,
    "threshold": 0.3
}
```

---

### System Operations

#### Export

---

#### POST /export/conversation/{chat_id}

Export a single conversation (JSON, CSV, TXT).

---

#### PDF Export

---

#### POST /pdf_export/pdf/{chat_id}

Generate a PDF document of a conversation.

---

#### Batch Operations

Bulk processing for multiple conversations.

---

#### POST /batch/summarize

Summarize multiple conversations at once.

---

#### Task Queue

Background task management.

---

#### GET /tasks

List active and completed background tasks.

---

#### GET /tasks/{task_id}

Get status of a specific task.

---

### Management

#### Metrics

---

#### GET /metrics

Prometheus-compatible performance metrics.

---

#### GET /metrics/memory

Detailed RAM and GPU usage.

---

#### Template Analytics

---

#### GET /metrics/templates/dashboard

Get performance analytics for prompt templates.

---

#### Settings

---

#### GET /settings

Get current application configuration.

---

#### PUT /settings

Update application configuration.

---

#### GET /settings/models

List available language models.

---

#### WebSocket

---

#### GET /ws

WebSocket endpoint for real-time streaming generation and status updates.

---

## Privacy

All data processing happens locally on your Mac. No conversation data, messages, or personal information is ever sent to external servers. The MLX language models run entirely on Apple Silicon.