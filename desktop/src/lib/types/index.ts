/**
 * Core Type Definitions for JARVIS
 * 
 * Strictly typed domain models with proper type guards
 */

// Primitive type helpers
export type Nullable<T> = T | null;
export type Optional<T> = T | undefined;

// Connection Status
export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting';

// Toast Types
export type ToastType = 'success' | 'error' | 'warning' | 'info';

// Theme Types
export type ThemeMode = 'dark' | 'light' | 'system';

export type AccentColorKey = 
  | 'blue' 
  | 'purple' 
  | 'pink' 
  | 'red' 
  | 'orange' 
  | 'yellow' 
  | 'green' 
  | 'teal' 
  | 'indigo';

// Priority Types
export type PriorityLevel = 'critical' | 'high' | 'medium' | 'low';

export type PriorityReason =
  | 'contains_question'
  | 'action_requested'
  | 'time_sensitive'
  | 'important_contact'
  | 'frequent_contact'
  | 'awaiting_response'
  | 'multiple_messages'
  | 'contains_urgency'
  | 'normal';

// Core Message Types
export interface Attachment {
  filename: string;
  file_path: Nullable<string>;
  mime_type: Nullable<string>;
  file_size: Nullable<number>;
}

export interface Reaction {
  type: string;
  sender: string;
  sender_name: Nullable<string>;
  date: string;
}

export interface Message {
  id: number;
  chat_id: string;
  sender: string;
  sender_name: Nullable<string>;
  text: string;
  date: string;
  is_from_me: boolean;
  attachments: Attachment[];
  reply_to_id: Nullable<number>;
  reactions: Reaction[];
  date_delivered: Nullable<string>;
  date_read: Nullable<string>;
  is_system_message: boolean;
}

// Optimistic Message Extension
export interface OptimisticMessage extends Message {
  _optimistic: true;
  _optimisticId: string;
  _optimisticStatus: 'sending' | 'sent' | 'failed';
  _optimisticError?: string;
}

// Type guard for optimistic messages
export function isOptimisticMessage(message: Message | OptimisticMessage): message is OptimisticMessage {
  return '_optimistic' in message && message._optimistic === true;
}

// Get optimistic status safely
export function getOptimisticStatus(message: Message): 'sending' | 'sent' | 'failed' | null {
  return isOptimisticMessage(message) ? message._optimisticStatus : null;
}

export function getOptimisticError(message: Message): string | undefined {
  return isOptimisticMessage(message) ? message._optimisticError : undefined;
}

export function getOptimisticId(message: Message): string | undefined {
  return isOptimisticMessage(message) ? message._optimisticId : undefined;
}

// Conversation Types
export interface Conversation {
  chat_id: string;
  participants: string[];
  display_name: Nullable<string>;
  last_message_date: string;
  message_count: number;
  is_group: boolean;
  last_message_text: Nullable<string>;
}

// Topic Types
export interface Topic {
  topic: string;
  confidence: number;
  color: string;
  display_name: string;
}

export interface TopicsResponse {
  chat_id: string;
  topics: Topic[];
  all_topics: Topic[];
  cached: boolean;
  message_count_analyzed: number;
}

// Draft Suggestion Types
export interface DraftSuggestion {
  text: string;
  confidence: number;
}

// Toast Types
export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  description?: string;
  duration: number;
  dismissible: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

// Keyboard Navigation
export type FocusZone = 'sidebar' | 'conversations' | 'messages' | 'compose' | 'modal' | null;

// Health Types
export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';

export interface HealthResponse {
  status: HealthStatus;
  imessage_access: Nullable<boolean>;
  memory_available_gb: Nullable<number>;
  memory_used_gb: Nullable<number>;
  memory_mode: Nullable<'FULL' | 'LITE' | 'MINIMAL'>;
  model_loaded: Nullable<boolean>;
  permissions_ok: Nullable<boolean>;
  details: Nullable<Record<string, string> | string>;
  jarvis_rss_mb: Nullable<number>;
  jarvis_vms_mb: Nullable<number>;
}

// Settings Types
export interface ModelInfo {
  model_id: string;
  name: string;
  size_gb: number;
  quality_tier: 'basic' | 'good' | 'best';
  ram_requirement_gb: number;
  is_downloaded: boolean;
  is_loaded: boolean;
  is_recommended: boolean;
  description: Nullable<string>;
}

export interface GenerationSettings {
  temperature: number;
  max_tokens_reply: number;
  max_tokens_summary: number;
}

export interface BehaviorSettings {
  auto_suggest_replies: boolean;
  suggestion_count: number;
  context_messages_reply: number;
  context_messages_summary: number;
}

export interface SystemInfo {
  system_ram_gb: number;
  current_memory_usage_gb: number;
  model_loaded: boolean;
  model_memory_usage_gb: number;
  imessage_access: boolean;
}

export interface SettingsResponse {
  model_id: string;
  generation: GenerationSettings;
  behavior: BehaviorSettings;
  system: SystemInfo;
}

// Search Types
export interface SearchFilters {
  sender?: string;
  after?: string;
  before?: string;
  has_attachments?: boolean;
}

export interface SearchResult extends Message {
  conversation_name?: string;
}

// Time Range
export type TimeRange = 'week' | 'month' | 'three_months' | 'all_time';

// API Error
export class APIError extends Error {
  status: number;
  detail: Nullable<string>;

  constructor(message: string, status: number, detail: Nullable<string> = null) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.detail = detail;
  }
}

// View Types
export type ViewType = 'messages' | 'dashboard' | 'health' | 'settings' | 'templates' | 'network';
