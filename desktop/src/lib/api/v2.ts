/**
 * @deprecated This client duplicates functionality in the main ApiClient (client.ts).
 * Only kept because SmartReplyChipsV2.svelte depends on it.
 * TODO: Migrate SmartReplyChipsV2 to use the main ApiClient, then delete this file.
 */

const V2_API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8742";

export interface V2HealthResponse {
  status: string;
  version: string;
  model_loaded: boolean;
  imessage_accessible: boolean;
}

export interface V2Conversation {
  chat_id: string;
  display_name: string | null;
  participants: string[];
  last_message_date: string | null;
  last_message_text: string | null;
  message_count: number;
  is_group: boolean;
}

export interface V2Message {
  id: number;
  text: string;
  sender: string;
  is_from_me: boolean;
  timestamp: string | null;
  chat_id: string;
}

export interface V2GeneratedReply {
  text: string;
  reply_type: string;
  confidence: number;
}

export interface V2GenerateRepliesResponse {
  replies: V2GeneratedReply[];
  chat_id: string;
  model_used: string;
  generation_time_ms: number;
  context_summary: string;
}

export interface V2ModelInfo {
  id: string;
  display_name: string;
  size_gb: number;
  quality: string;
  description: string;
}

class V2ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = V2_API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  // Health
  async getHealth(): Promise<V2HealthResponse> {
    return this.fetch<V2HealthResponse>("/health");
  }

  // Conversations
  async getConversations(limit: number = 50): Promise<{ conversations: V2Conversation[]; total: number }> {
    return this.fetch(`/conversations?limit=${limit}`);
  }

  async getMessages(chatId: string, limit: number = 50): Promise<{ messages: V2Message[]; chat_id: string; total: number }> {
    return this.fetch(`/conversations/${encodeURIComponent(chatId)}/messages?limit=${limit}`);
  }

  // Generation - uses /drafts/reply endpoint
  async generateReplies(chatId: string, numReplies: number = 3): Promise<V2GenerateRepliesResponse> {
    // Call the actual backend endpoint
    const response = await this.fetch<{
      suggestions: Array<{ text: string; confidence: number }>;
      context_used: {
        num_messages: number;
        participants: string[];
        last_message: string;
      };
    }>("/drafts/reply", {
      method: "POST",
      body: JSON.stringify({
        chat_id: chatId,
        num_suggestions: numReplies,
        context_messages: 20,
      }),
    });

    // Transform to V2 response format
    return {
      replies: response.suggestions.map((s) => ({
        text: s.text,
        reply_type: "ai",
        confidence: s.confidence,
      })),
      chat_id: chatId,
      model_used: "mlx-local",
      generation_time_ms: 0,
      context_summary: response.context_used.last_message,
    };
  }

  // Settings
  async getSettings(): Promise<{ model_id: string; auto_suggest: boolean; max_replies: number }> {
    return this.fetch("/settings");
  }

  async updateSettings(settings: { model_id?: string; auto_suggest?: boolean; max_replies?: number }): Promise<{ model_id: string; auto_suggest: boolean; max_replies: number }> {
    return this.fetch("/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    });
  }

  async getModels(): Promise<{ models: V2ModelInfo[] }> {
    return this.fetch("/settings/models");
  }
}

// Export singleton instance
export const v2Api = new V2ApiClient();
