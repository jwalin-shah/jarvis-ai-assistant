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
    - [Attachments](#attachments)
  - [AI Generation & Search](#ai-generation--search)
    - [Drafts](#drafts-ai-generation)
    - [Suggestions](#suggestions-quick-replies)
    - [Semantic Search & Embeddings](#semantic-search--embeddings)
    - [Relationship Learning](#relationship-learning)
  - [Evaluation & Optimization](#evaluation--optimization)
    - [Feedback](#feedback)
    - [A/B Testing](#ab-testing)
    - [Quality Metrics](#quality-metrics)
  - [System Operations](#system-operations)
    - [Digests](#digests)
    - [Export](#export)
    - [PDF Export](#pdf-export)
    - [Batch Operations](#batch-operations)
    - [Task Queue](#task-queue)
    - [MCP Server](#mcp-server)
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

---

## Endpoints

### Health

System health monitoring and status checks.

---

#### GET /health

Get comprehensive system health status.

---

### iMessage Data

#### Conversations

---

#### GET /conversations

List recent conversations sorted by last message date.

---

#### GET /conversations/{chat_id}/messages

Get messages for a specific conversation.

---

#### Attachments

---

#### GET /attachments/stats/{chat_id}

Get attachment statistics for a conversation.

---

#### GET /attachments/storage/summary

Get summary of storage usage by attachments across all conversations.

---

### AI Generation & Search

#### Drafts (AI Generation)

---

#### POST /drafts/reply

Generate AI-powered reply suggestions.

---

#### Suggestions (Quick Replies)

Supports both 1-on-1 chats and **group chats** with specialized patterns.

---

#### POST /suggestions

Fast pattern-based reply suggestions (no AI model required).

---

#### Semantic Search & Embeddings

---

#### POST /search/semantic

Perform a semantic search using AI-powered embeddings.

---

#### POST /embeddings/index/{chat_id}

Manually trigger indexing of a conversation's messages into the embedding store.

---

#### Relationship Learning

Learns communication patterns with each contact for personalized generation.

---

#### GET /relationships/profile/{contact_id}

Get the learned communication profile for a contact.

---

#### GET /relationships/style-guide/{contact_id}

Get a natural language style guide for communicating with this contact.

---

#### POST /relationships/refresh/{contact_id}

Force a refresh of the relationship profile based on recent messages.

---

### Evaluation & Optimization

#### Feedback

Record user feedback on AI suggestions to improve future performance.

---

#### POST /feedback/record

Record an action (sent, edited, dismissed) for a suggestion.

---

#### GET /feedback/stats

Get overall feedback statistics and acceptance rates.

---

#### GET /feedback/improvements

Get AI-generated suggestions for improving prompts based on feedback.

---

#### A/B Testing

Infrastructure for prompt experimentation.

---

#### GET /experiments/list

List all active and completed prompt experiments.

---

#### GET /experiments/results/{experiment_name}

Get results and statistical significance for an experiment.

---

#### Quality Metrics

Comprehensive dashboard for monitoring generation quality.

---

#### GET /quality/dashboard

Get all quality metrics for the dashboard (HHEM scores, acceptance, latency).

---

### System Operations

#### Digests

---

#### POST /digest/generate

Generate a daily or weekly digest of missed messages and action items.

---

#### POST /digest/export

Export a generated digest to Markdown or HTML.

---

#### MCP Server

JARVIS provides a Model Context Protocol (MCP) server for integration with other AI tools.

---

#### POST /mcp/rpc

JSON-RPC endpoint for MCP tool execution and resource access.

---

## Privacy

All data processing happens locally on your Mac. No conversation data, messages, or personal information is ever sent to external servers. The MLX language models run entirely on Apple Silicon.
