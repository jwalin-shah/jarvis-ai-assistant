/**
 * API type definitions for JARVIS frontend
 */

// Message types
export interface Attachment {
  filename: string;
  file_path: string | null;
  mime_type: string | null;
  file_size: number | null;
}

export interface Reaction {
  type: string;
  sender: string;
  sender_name: string | null;
  date: string;
}

export interface Message {
  id: number;
  chat_id: string;
  sender: string;
  sender_name: string | null;
  text: string;
  date: string;
  is_from_me: boolean;
  attachments: Attachment[];
  reply_to_id: number | null;
  reactions: Reaction[];
  date_delivered: string | null;
  date_read: string | null;
  is_system_message: boolean;
  // View-specific fields (for optimistic UI)
  _optimistic?: boolean;
  _optimisticId?: string;
  _optimisticStatus?: "sending" | "sent" | "failed" | null;
  _optimisticError?: string;
}

/** Pagination state for a conversation */
export interface PaginationState {
  hasMore: boolean;
  loadingMore: boolean;
}

export type ConnectionStatus = "connected" | "disconnected" | "connecting";

/** Optimistic message for immediate UI feedback during send */
export interface OptimisticMessage {
  id: string; // Temporary ID (e.g., "optimistic-{timestamp}")
  text: string;
  status: "sending" | "sent" | "failed";
  error?: string;
  timestamp: number;
  stableId: number; // Negative ID assigned once for stable rendering
}

// Conversation types
export interface Conversation {
  chat_id: string;
  participants: string[];
  display_name: string | null;
  last_message_date: string;
  message_count: number;
  is_group: boolean;
  last_message_text: string | null;
}

// Health types
export interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  imessage_access: boolean | null;
  memory_available_gb: number | null;
  memory_used_gb: number | null;
  memory_mode: "FULL" | "LITE" | "MINIMAL" | null;
  model_loaded: boolean | null;
  permissions_ok: boolean | null;
  details: Record<string, string> | string | null;
  jarvis_rss_mb: number | null;
  jarvis_vms_mb: number | null;
}

// Settings types
export interface ModelInfo {
  model_id: string;
  name: string;
  size_gb: number;
  quality_tier: "basic" | "good" | "best";
  ram_requirement_gb: number;
  is_downloaded: boolean;
  is_loaded: boolean;
  is_recommended: boolean;
  description: string | null;
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

export interface SettingsUpdateRequest {
  model_id?: string;
  generation?: Partial<GenerationSettings>;
  behavior?: Partial<BehaviorSettings>;
}

export interface DownloadStatus {
  model_id: string;
  status: "downloading" | "completed" | "failed";
  progress: number;
  error: string | null;
}

export interface ActivateResponse {
  success: boolean;
  model_id: string;
  error: string | null;
}

// Draft and Summary types
export interface DraftSuggestion {
  text: string;
  confidence: number;
}

export interface ContextInfo {
  num_messages: number;
  participants: string[];
  last_message: string | null;
}

export interface DraftReplyResponse {
  suggestions: DraftSuggestion[];
  context_used: ContextInfo;
  gated?: boolean;
  gated_confidence?: number;
}

export interface DateRange {
  start: string;
  end: string;
}

export interface SummaryResponse {
  summary: string;
  key_points: string[];
  date_range: DateRange;
  message_count?: number;
}

// Smart Reply Suggestion types
export interface SmartReplySuggestion {
  text: string;
  score: number;
}

export interface SmartReplySuggestionsResponse {
  suggestions: SmartReplySuggestion[];
}

// Search types
export interface SearchFilters {
  sender?: string;
  after?: string;
  before?: string;
  has_attachments?: boolean;
}

export interface SearchResult extends Message {
  conversation_name?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  query: string;
  filters_applied: SearchFilters;
}

// Grouped search results by conversation
export interface GroupedSearchResults {
  chat_id: string;
  conversation_name: string;
  results: SearchResult[];
}

// Error types
export interface ApiError {
  error: string;
  detail: string;
  code: string | null;
}

// Statistics types
export type TimeRange = "week" | "month" | "three_months" | "all_time";

export interface HourlyActivity {
  hour: number;
  count: number;
}

export interface WordFrequency {
  word: string;
  count: number;
}

export interface ConversationStats {
  chat_id: string;
  time_range: TimeRange;
  total_messages: number;
  sent_count: number;
  received_count: number;
  avg_response_time_minutes: number | null;
  hourly_activity: HourlyActivity[];
  daily_activity: Record<string, number>;
  message_length_distribution: Record<string, number>;
  top_words: WordFrequency[];
  emoji_usage: Record<string, number>;
  attachment_breakdown: Record<string, number>;
  first_message_date: string | null;
  last_message_date: string | null;
  participants: string[];
}

// Template Analytics types
export interface TemplateAnalyticsSummary {
  total_queries: number;
  template_hits: number;
  model_fallbacks: number;
  hit_rate_percent: number;
  cache_hit_rate: number;
  unique_templates_matched: number;
  queries_per_second: number;
  uptime_seconds: number;
}

export interface TopTemplateItem {
  template_name: string;
  match_count: number;
}

export interface MissedQueryItem {
  query_hash: string;
  similarity: number;
  best_template: string | null;
  timestamp: string;
}

export interface CategoryAverageItem {
  category: string;
  average_similarity: number;
}

export interface TemplateCoverage {
  total_templates: number;
  total_patterns: number;
  responses_from_templates: number;
  responses_from_model: number;
  coverage_percent: number;
}

export interface PieChartData {
  template_responses: number;
  model_responses: number;
}

export interface TemplateAnalyticsDashboard {
  summary: TemplateAnalyticsSummary;
  top_templates: TopTemplateItem[];
  missed_queries: MissedQueryItem[];
  category_averages: CategoryAverageItem[];
  coverage: TemplateCoverage;
  pie_chart_data: PieChartData;
}

// Semantic Search types
export interface SemanticSearchFilters {
  sender?: string;
  chat_id?: string;
  after?: string;
  before?: string;
  has_attachments?: boolean;
}

export interface SemanticSearchRequest {
  query: string;
  limit?: number;
  threshold?: number;
  index_limit?: number;
  filters?: SemanticSearchFilters;
}

export interface SemanticSearchResultItem {
  message: Message;
  similarity: number;
}

export interface SemanticSearchResponse {
  query: string;
  results: SemanticSearchResultItem[];
  total_results: number;
  threshold_used: number;
  messages_searched: number;
}

export interface SemanticCacheStats {
  embedding_count: number;
  size_bytes: number;
  size_mb: number;
}

// Priority Inbox types
export type PriorityLevel = "critical" | "high" | "medium" | "low";

export type PriorityReason =
  | "contains_question"
  | "action_requested"
  | "time_sensitive"
  | "important_contact"
  | "frequent_contact"
  | "awaiting_response"
  | "multiple_messages"
  | "contains_urgency"
  | "normal";

export interface PriorityMessage {
  message_id: number;
  chat_id: string;
  sender: string;
  sender_name: string | null;
  text: string;
  date: string;
  priority_score: number;
  priority_level: PriorityLevel;
  reasons: PriorityReason[];
  needs_response: boolean;
  handled: boolean;
  conversation_name: string | null;
}

export interface PriorityInboxResponse {
  messages: PriorityMessage[];
  total_count: number;
  unhandled_count: number;
  needs_response_count: number;
  critical_count: number;
  high_count: number;
}

export interface MarkHandledRequest {
  chat_id: string;
  message_id: number;
}

export interface MarkHandledResponse {
  success: boolean;
  chat_id: string;
  message_id: number;
  handled: boolean;
}

export interface ImportantContactRequest {
  identifier: string;
  important: boolean;
}

export interface ImportantContactResponse {
  success: boolean;
  identifier: string;
  important: boolean;
}

export interface PriorityStats {
  handled_count: number;
  important_contacts_count: number;
}

// PDF Export types
export interface PDFExportDateRange {
  start?: string;
  end?: string;
}

export interface PDFExportRequest {
  include_attachments?: boolean;
  include_reactions?: boolean;
  date_range?: PDFExportDateRange;
  limit?: number;
}

export interface PDFExportResponse {
  success: boolean;
  filename: string;
  data: string; // Base64-encoded PDF
  message_count: number;
  size_bytes: number;
}

// WebSocket types
export interface WebSocketStatus {
  active_connections: number;
  health_subscribers: number;
  status: "operational" | "degraded" | "offline";
}

export interface GenerationStreamRequest {
  prompt: string;
  context_documents?: string[];
  few_shot_examples?: Array<{ input: string; output: string }>;
  max_tokens?: number;
  temperature?: number;
  stop_sequences?: string[];
}

export interface StreamingGenerationResult {
  generation_id: string;
  text: string;
  tokens_used: number;
  generation_time_ms: number;
  model_name: string;
  used_template: boolean;
  template_name: string | null;
  finish_reason: string;
}

// Insights types
export interface SentimentResult {
  score: number;
  label: "positive" | "negative" | "neutral";
  positive_count: number;
  negative_count: number;
  neutral_count: number;
}

export interface SentimentTrend {
  date: string;
  score: number;
  message_count: number;
}

export interface ResponsePatterns {
  avg_response_time_minutes: number | null;
  median_response_time_minutes: number | null;
  fastest_response_minutes: number | null;
  slowest_response_minutes: number | null;
  response_times_by_hour: Record<number, number>;
  response_times_by_day: Record<string, number>;
  my_avg_response_time_minutes: number | null;
  their_avg_response_time_minutes: number | null;
}

export interface FrequencyTrends {
  daily_counts: Record<string, number>;
  weekly_counts: Record<string, number>;
  monthly_counts: Record<string, number>;
  trend_direction: "increasing" | "decreasing" | "stable";
  trend_percentage: number;
  most_active_day: string | null;
  most_active_hour: number | null;
  messages_per_day_avg: number;
}

export interface RelationshipHealth {
  overall_score: number;
  engagement_score: number;
  sentiment_score: number;
  responsiveness_score: number;
  consistency_score: number;
  health_label: "excellent" | "good" | "fair" | "needs_attention" | "concerning";
  factors: Record<string, string>;
}

export interface ConversationInsights {
  chat_id: string;
  contact_name: string | null;
  time_range: TimeRange;
  sentiment_overall: SentimentResult;
  sentiment_trends: SentimentTrend[];
  response_patterns: ResponsePatterns;
  frequency_trends: FrequencyTrends;
  relationship_health: RelationshipHealth;
  total_messages_analyzed: number;
  first_message_date: string | null;
  last_message_date: string | null;
}

// Calendar types
export interface DetectedEvent {
  title: string;
  start: string;
  end: string | null;
  location: string | null;
  description: string | null;
  all_day: boolean;
  confidence: number;
  source_text: string;
  message_id: number | null;
}

export interface Calendar {
  id: string;
  name: string;
  color: string | null;
  is_editable: boolean;
}

export interface CalendarEvent {
  id: string;
  calendar_id: string;
  calendar_name: string;
  title: string;
  start: string;
  end: string;
  all_day: boolean;
  location: string | null;
  notes: string | null;
  url: string | null;
  attendees: string[];
  status: "confirmed" | "tentative" | "cancelled";
}

export interface CreateEventResponse {
  success: boolean;
  event_id: string | null;
  error: string | null;
}

// Custom Template types
export interface CustomTemplate {
  id: string;
  name: string;
  template_text: string;
  trigger_phrases: string[];
  category: string;
  tags: string[];
  min_group_size: number | null;
  max_group_size: number | null;
  enabled: boolean;
  created_at: string;
  updated_at: string;
  usage_count: number;
}

export interface CustomTemplateCreateRequest {
  name: string;
  template_text: string;
  trigger_phrases: string[];
  category?: string;
  tags?: string[];
  min_group_size?: number | null;
  max_group_size?: number | null;
  enabled?: boolean;
}

export interface CustomTemplateUpdateRequest {
  name?: string;
  template_text?: string;
  trigger_phrases?: string[];
  category?: string;
  tags?: string[];
  min_group_size?: number | null;
  max_group_size?: number | null;
  enabled?: boolean;
}

export interface CustomTemplateListResponse {
  templates: CustomTemplate[];
  total: number;
  categories: string[];
  tags: string[];
}

export interface CustomTemplateUsageStats {
  total_templates: number;
  enabled_templates: number;
  total_usage: number;
  usage_by_category: Record<string, number>;
  top_templates: Array<{ id: string; name: string; usage_count: number }>;
}

export interface CustomTemplateTestRequest {
  trigger_phrases: string[];
  test_inputs: string[];
}

export interface CustomTemplateTestResult {
  input: string;
  matched: boolean;
  best_match: string | null;
  similarity: number;
}

export interface CustomTemplateTestResponse {
  results: CustomTemplateTestResult[];
  match_rate: number;
  threshold: number;
}

export interface CustomTemplateExportResponse {
  version: number;
  export_date: string;
  template_count: number;
  templates: Record<string, unknown>[];
}

export interface CustomTemplateImportRequest {
  data: Record<string, unknown>;
  overwrite?: boolean;
}

export interface CustomTemplateImportResponse {
  imported: number;
  skipped: number;
  errors: number;
  total_templates: number;
}

// Threading types
export interface ThreadResponse {
  thread_id: string;
  messages: number[];
  topic_label: string;
  start_time: string | null;
  end_time: string | null;
  participant_count: number;
  message_count: number;
}

export interface ThreadedMessage extends Message {
  thread_id: string;
  thread_position: number;
  is_thread_start: boolean;
}

export interface ThreadedViewResponse {
  chat_id: string;
  threads: ThreadResponse[];
  messages: ThreadedMessage[];
  total_threads: number;
  total_messages: number;
}

export interface ThreadingConfigRequest {
  time_gap_threshold_minutes?: number;
  semantic_similarity_threshold?: number;
  use_semantic_analysis?: boolean;
}

// Attachment Manager types
export type AttachmentType = "images" | "videos" | "audio" | "documents" | "other" | "all";

export interface ExtendedAttachment {
  filename: string;
  file_path: string | null;
  mime_type: string | null;
  file_size: number | null;
  width: number | null;
  height: number | null;
  duration_seconds: number | null;
  created_date: string | null;
  is_sticker: boolean;
  uti: string | null;
}

export interface AttachmentWithContext {
  attachment: ExtendedAttachment;
  message_id: number;
  message_date: string;
  chat_id: string;
  sender: string;
  sender_name: string | null;
  is_from_me: boolean;
}

export interface AttachmentStats {
  chat_id: string;
  total_count: number;
  total_size_bytes: number;
  total_size_formatted: string;
  by_type: Record<string, number>;
  size_by_type: Record<string, number>;
}

export interface StorageByConversation {
  chat_id: string;
  display_name: string | null;
  attachment_count: number;
  total_size_bytes: number;
  total_size_formatted: string;
}

export interface StorageSummary {
  total_attachments: number;
  total_size_bytes: number;
  total_size_formatted: string;
  by_conversation: StorageByConversation[];
}

// Feedback types
export type FeedbackAction = "sent" | "edited" | "dismissed" | "copied";

export interface EvaluationScores {
  tone_score: number;
  relevance_score: number;
  naturalness_score: number;
  length_score: number;
  overall_score: number;
}

export interface RecordFeedbackRequest {
  action: FeedbackAction;
  suggestion_text: string;
  chat_id: string;
  context_messages: string[];
  edited_text?: string | null;
  include_evaluation?: boolean;
  metadata?: Record<string, unknown>;
}

export interface RecordFeedbackResponse {
  success: boolean;
  suggestion_id: string;
  evaluation: EvaluationScores | null;
}

export interface FeedbackStatsResponse {
  total_feedback: number;
  sent_unchanged: number;
  edited: number;
  dismissed: number;
  copied: number;
  acceptance_rate: number;
  edit_rate: number;
  avg_evaluation_scores: Record<string, number> | null;
}

export interface ImprovementSuggestion {
  type: string;
  suggestion: string;
  detail: string;
  confidence: number;
}

export interface ImprovementsResponse {
  improvements: ImprovementSuggestion[];
  based_on_entries: number;
}

// A/B Testing Experiment types
export type UserAction =
  | "sent_unchanged"
  | "sent_edited"
  | "dismissed"
  | "regenerated";

export interface VariantConfig {
  id: string;
  weight: number;
  config: Record<string, unknown>;
}

export interface Experiment {
  name: string;
  enabled: boolean;
  description: string;
  created_at: string;
  variants: VariantConfig[];
}

export interface VariantResults {
  variant_id: string;
  total_impressions: number;
  sent_unchanged: number;
  sent_edited: number;
  dismissed: number;
  regenerated: number;
  conversion_rate: number;
}

export interface ExperimentResults {
  experiment_name: string;
  variants: VariantResults[];
  total_outcomes: number;
  winner: string | null;
  is_significant: boolean;
  p_value: number | null;
}

export interface ExperimentListResponse {
  experiments: Experiment[];
  total: number;
}

export interface RecordOutcomeRequest {
  variant_id: string;
  contact_id: string;
  action: UserAction;
}

export interface RecordOutcomeResponse {
  success: boolean;
  experiment_name: string;
  variant_id: string;
  action: string;
}

export interface CreateExperimentRequest {
  name: string;
  description?: string;
  enabled?: boolean;
  variants: Array<{
    id: string;
    weight: number;
    config?: Record<string, unknown>;
  }>;
}

// Quality Metrics types
export interface QualitySummary {
  total_responses: number;
  template_responses: number;
  model_responses: number;
  template_hit_rate_percent: number;
  model_fallback_rate_percent: number;
  avg_hhem_score: number | null;
  hhem_score_count: number;
  acceptance_rate_percent: number;
  accepted_unchanged_count: number;
  accepted_modified_count: number;
  rejected_count: number;
  avg_edit_distance: number | null;
  avg_template_latency_ms: number;
  avg_model_latency_ms: number;
  uptime_seconds: number;
  responses_per_second: number;
}

export interface TrendDataPoint {
  date: string;
  template_hit_rate_percent: number;
  model_hit_rate_percent: number;
  avg_hhem_score: number | null;
  acceptance_rate_percent: number;
  avg_edit_distance: number | null;
  avg_template_latency_ms: number;
  avg_model_latency_ms: number;
  total_responses: number;
}

export interface ContactQuality {
  contact_id: string;
  total_responses: number;
  template_responses: number;
  model_responses: number;
  acceptance_rate: number;
  avg_hhem_score: number | null;
  avg_edit_distance: number | null;
  avg_latency_ms: number;
}

export interface TimeOfDayQuality {
  hour: number;
  total_responses: number;
  template_hit_rate: number;
  acceptance_rate: number;
  avg_latency_ms: number;
}

export interface IntentQuality {
  intent: string;
  total_responses: number;
  template_hit_rate: number;
  acceptance_rate: number;
  avg_hhem_score: number | null;
  avg_latency_ms: number;
}

export interface ConversationTypeQuality {
  total_responses: number;
  template_hit_rate_percent: number;
  acceptance_rate_percent: number;
  avg_hhem_score: number | null;
  avg_latency_ms: number;
}

export interface Recommendation {
  category: string;
  priority: string;
  title: string;
  description: string;
  metric_value: number | null;
  target_value: number | null;
}

export interface QualityDashboardData {
  summary: QualitySummary;
  trends: TrendDataPoint[];
  top_contacts: ContactQuality[];
  time_of_day: TimeOfDayQuality[];
  by_intent: IntentQuality[];
  by_conversation_type: Record<string, ConversationTypeQuality>;
  recommendations: Recommendation[];
}

// =============================================================================
// Tag System types
// =============================================================================

// Tag types
export interface Tag {
  id: number;
  name: string;
  color: string;
  icon: string;
  description: string | null;
  parent_id: number | null;
  aliases: string[];
  sort_order: number;
  is_system: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export interface TagWithPath extends Tag {
  path: string;
}

export interface TagCreateRequest {
  name: string;
  color?: string;
  icon?: string;
  description?: string | null;
  parent_id?: number | null;
  aliases?: string[];
}

export interface TagUpdateRequest {
  name?: string;
  color?: string;
  icon?: string;
  description?: string | null;
  parent_id?: number | null;
  aliases?: string[];
  sort_order?: number;
}

export interface TagListResponse {
  tags: Tag[];
  total: number;
}

// Conversation Tag types
export interface ConversationTag {
  chat_id: string;
  tag: Tag;
  added_at: string | null;
  added_by: string;
  confidence: number;
}

export interface ConversationTagsResponse {
  chat_id: string;
  tags: ConversationTag[];
}

export interface ConversationTagRequest {
  tag_id: number;
}

export interface BulkTagRequest {
  chat_ids: string[];
  tag_ids: number[];
}

export interface BulkTagResponse {
  affected_count: number;
  chat_ids: string[];
}

// Smart Folder types
export interface RuleCondition {
  field: string;
  operator: string;
  value: string | number | boolean | string[] | null;
}

export interface SmartFolderRules {
  match: "all" | "any";
  conditions: RuleCondition[];
  sort_by: string;
  sort_order: "asc" | "desc";
  limit: number;
}

export interface SmartFolder {
  id: number;
  name: string;
  icon: string;
  color: string;
  rules: SmartFolderRules;
  sort_order: number;
  is_default: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export interface SmartFolderCreateRequest {
  name: string;
  icon?: string;
  color?: string;
  rules: SmartFolderRules;
}

export interface SmartFolderUpdateRequest {
  name?: string;
  icon?: string;
  color?: string;
  rules?: SmartFolderRules;
  sort_order?: number;
}

export interface SmartFolderListResponse {
  folders: SmartFolder[];
  total: number;
}

export interface SmartFolderPreviewRequest {
  rules: SmartFolderRules;
  limit?: number;
}

export interface SmartFolderPreviewResponse {
  total_matches: number;
  preview: Record<string, unknown>[];
  has_more: boolean;
}

// Tag Rule types (Auto-tagging)
export interface TagRule {
  id: number;
  name: string;
  trigger: string;
  conditions: RuleCondition[];
  tag_ids: number[];
  priority: number;
  is_enabled: boolean;
  created_at: string | null;
  last_triggered_at: string | null;
  trigger_count: number;
}

export interface TagRuleCreateRequest {
  name: string;
  trigger?: string;
  conditions: RuleCondition[];
  tag_ids: number[];
  priority?: number;
  is_enabled?: boolean;
}

export interface TagRuleUpdateRequest {
  name?: string;
  trigger?: string;
  conditions?: RuleCondition[];
  tag_ids?: number[];
  priority?: number;
  is_enabled?: boolean;
}

export interface TagRuleListResponse {
  rules: TagRule[];
  total: number;
}

// Tag Suggestion types
export interface TagSuggestion {
  tag_id: number | null;
  tag_name: string;
  confidence: number;
  reason: string | null;
  source: string;
}

export interface TagSuggestionsRequest {
  chat_id: string;
  limit?: number;
}

export interface TagSuggestionsResponse {
  chat_id: string;
  suggestions: TagSuggestion[];
}

export interface SuggestionFeedbackRequest {
  chat_id: string;
  tag_id: number;
  accepted: boolean;
}

// Tag Statistics types
export interface TagUsageStats {
  id: number;
  name: string;
  count: number;
}

export interface TagStatisticsResponse {
  total_tags: number;
  total_tagged_conversations: number;
  average_tags_per_conversation: number;
  most_used_tags: TagUsageStats[];
}

// Tag color presets
export const TAG_COLORS = {
  RED: "#ef4444",
  ORANGE: "#f97316",
  AMBER: "#f59e0b",
  YELLOW: "#eab308",
  LIME: "#84cc16",
  GREEN: "#22c55e",
  EMERALD: "#10b981",
  TEAL: "#14b8a6",
  CYAN: "#06b6d4",
  SKY: "#0ea5e9",
  BLUE: "#3b82f6",
  INDIGO: "#6366f1",
  VIOLET: "#8b5cf6",
  PURPLE: "#a855f7",
  FUCHSIA: "#d946ef",
  PINK: "#ec4899",
  ROSE: "#f43f5e",
  SLATE: "#64748b",
} as const;

// Tag icon options
export const TAG_ICONS = [
  "star",
  "heart",
  "flag",
  "bookmark",
  "folder",
  "tag",
  "inbox",
  "briefcase",
  "home",
  "users",
  "user",
  "clock",
  "calendar",
  "bell",
  "alert",
  "check",
  "circle",
  "square",
  "arrow-up",
  "arrow-down",
  "message",
  "mail",
  "phone",
  "sparkles",
  "fire",
  "zap",
] as const;

// Rule field options
export const RULE_FIELDS = [
  "chat_id",
  "display_name",
  "last_message_date",
  "message_count",
  "is_group",
  "unread_count",
  "is_flagged",
  "relationship",
  "contact_name",
  "last_message_text",
  "has_attachments",
  "tags",
  "sentiment",
  "priority",
  "needs_response",
] as const;

// Rule operator options
export const RULE_OPERATORS = [
  "equals",
  "not_equals",
  "contains",
  "not_contains",
  "starts_with",
  "ends_with",
  "is_empty",
  "is_not_empty",
  "greater_than",
  "less_than",
  "in_last_days",
  "before",
  "after",
  "has_tag",
  "has_any_tag",
  "has_all_tags",
  "has_no_tags",
] as const;

// =============================================================================
// Analytics Dashboard types
// =============================================================================

export interface AnalyticsOverview {
  total_messages: number;
  sent_messages: number;
  received_messages: number;
  active_conversations: number;
  avg_messages_per_day: number;
  avg_response_time_minutes: number | null;
  sentiment: {
    score: number;
    label: string;
  };
  peak_hour: number | null;
  peak_day: string | null;
  date_range: {
    start: string | null;
    end: string | null;
  };
  period_comparison: {
    total_change_percent: number;
    sent_change_percent: number;
    contacts_change_percent: number;
  };
  time_range: string;
}

export interface TimelineDataPoint {
  date?: string;
  hour?: number;
  total: number;
  sent: number;
  received: number;
}

export interface AnalyticsTimeline {
  granularity: string;
  metric: string;
  time_range: string;
  data: TimelineDataPoint[];
  total_points: number;
}

export interface HeatmapDataPoint {
  date: string;
  count: number;
  level: number; // 0-4 activity level
}

export interface ActivityHeatmap {
  data: HeatmapDataPoint[];
  stats: {
    total_days: number;
    active_days: number;
    max_count: number;
    avg_count: number;
  };
  time_range: string;
}

export interface ContactStats {
  contact_id: string;
  contact_name: string | null;
  total_messages: number;
  sent_count: number;
  received_count: number;
  avg_response_time_minutes: number | null;
  sentiment_score: number;
  engagement_score: number;
  message_trend: string;
  last_message_date: string | null;
  emoji_usage: {
    total: number;
    per_message: number;
    top_emojis: Record<string, number>;
  };
  hourly_distribution: Record<number, number>;
  daily_distribution: Record<string, number>;
  weekly_counts: Record<string, number>;
  time_range: string;
}

export interface LeaderboardContact {
  rank: number;
  contact_id: string;
  contact_name: string | null;
  total_messages: number;
  sent_count: number;
  received_count: number;
  engagement_score: number;
  avg_response_time_minutes: number | null;
  sentiment_score: number;
  trend: string;
}

export interface ContactsLeaderboard {
  contacts: LeaderboardContact[];
  total_contacts: number;
  sort_by: string;
  time_range: string;
}

export interface TrendInfo {
  direction: string;
  percentage_change: number;
  confidence: number;
}

export interface TrendingContact {
  contact_id: string;
  contact_name: string | null;
  trend: string;
  change_percent: number;
  confidence: number;
}

export interface Anomaly {
  date: string;
  type: string;
  value: number;
  expected: number;
  deviation: number;
}

export interface PeakPeriod {
  hour?: number;
  day?: string;
  count: number;
  percentage: number;
}

export interface TrendingPatterns {
  overall_trend: TrendInfo;
  weekly_trend: TrendInfo | null;
  trending_contacts: TrendingContact[];
  anomalies: Anomaly[];
  peak_hours: PeakPeriod[];
  peak_days: PeakPeriod[];
  seasonality: {
    detected: boolean;
    pattern: string | null;
  };
  time_range: string;
}

// =============================================================================
// Scheduler types
// =============================================================================

export type ScheduledPriority = "urgent" | "normal" | "low";

export type ScheduledStatus =
  | "pending"
  | "queued"
  | "sending"
  | "sent"
  | "failed"
  | "cancelled"
  | "expired";

export interface SendResult {
  success: boolean;
  sent_at: string | null;
  error: string | null;
  attempts: number;
}

export interface TimingSuggestion {
  suggested_time: string;
  confidence: number;
  reason: string;
  is_optimal: boolean;
}

export interface ScheduledItem {
  id: string;
  draft_id: string;
  contact_id: number;
  chat_id: string;
  message_text: string;
  send_at: string;
  priority: ScheduledPriority;
  status: ScheduledStatus;
  created_at: string;
  updated_at: string;
  expires_at: string | null;
  timezone: string | null;
  depends_on: string | null;
  retry_count: number;
  max_retries: number;
  result: SendResult | null;
}

export interface ScheduleDraftRequest {
  draft_id: string;
  contact_id: number;
  chat_id: string;
  message_text: string;
  send_at: string;
  priority?: ScheduledPriority;
  timezone?: string;
  depends_on?: string;
  metadata?: Record<string, unknown>;
}

export interface SmartScheduleRequest {
  draft_id: string;
  contact_id: number;
  chat_id: string;
  message_text: string;
  earliest?: string;
  latest?: string;
  priority?: ScheduledPriority;
  metadata?: Record<string, unknown>;
}

export interface SmartScheduleResponse {
  item: ScheduledItem;
  suggestion: TimingSuggestion;
}

export interface ScheduledListResponse {
  items: ScheduledItem[];
  total: number;
  pending: number;
  sent: number;
  failed: number;
}

export interface TimingSuggestionsResponse {
  suggestions: TimingSuggestion[];
  contact_id: number;
}

export interface SchedulerStatsResponse {
  running: boolean;
  total: number;
  pending: number;
  sent: number;
  failed: number;
  pending_in_undo_window: number;
  next_due: string | null;
}

// =============================================================================
// Graph Visualization types
// =============================================================================

export interface GraphNode {
  id: string;
  label: string;
  size: number;
  color: string;
  relationship_type: string;
  message_count: number;
  last_contact: string | null;
  sentiment_score: number;
  response_time_avg: number | null;
  x: number | null;
  y: number | null;
  cluster_id: number | null;
  metadata: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  message_count: number;
  sentiment: number;
  last_interaction: string | null;
  bidirectional: boolean;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: Record<string, unknown>;
}

export interface ClusterResult {
  clusters: Record<string, number>;
  modularity: number;
  num_clusters: number;
  cluster_sizes: Record<number, number>;
  cluster_labels: Record<number, string>;
}

export type LayoutType = "force" | "hierarchical" | "radial";

export interface GraphEvolutionSnapshot {
  timestamp: string;
  graph: GraphData;
  metrics: Record<string, unknown>;
}

export interface GraphEvolutionResponse {
  from_date: string;
  to_date: string;
  interval: string;
  snapshots: GraphEvolutionSnapshot[];
  total_snapshots: number;
}

export interface GraphExportResponse {
  format: string;
  filename: string;
  data: string;
  size_bytes: number;
}

export interface ContactFact {
  category: string;
  subject: string;
  predicate: string;
  value: string | null;
  confidence: number;
}

export interface ContactProfileDetail {
  contact_id: string;
  contact_name: string | null;
  relationship: string;
  formality: string;
  formality_score: number;
  style_guide: string;
  message_count: number;
  avg_message_length: number;
  avg_response_time_minutes: number | null;
  facts: ContactFact[];
}

export interface GraphStats {
  total_contacts: number;
  total_messages: number;
  avg_messages_per_contact: number;
  relationship_distribution: Record<string, number>;
  cluster_count: number;
  most_active_contact: string | null;
  most_active_messages: number;
  generated_at: string;
}

// =============================================================================
// Knowledge Graph types (facts as entity nodes)
// =============================================================================

export interface KnowledgeNode {
  id: string;
  label: string;
  node_type: "contact" | "entity";
  category: string;
  color: string;
  size: number;
  relationship_type?: string;
  message_count?: number;
  metadata: Record<string, unknown>;
}

export interface KnowledgeEdge {
  source: string;
  target: string;
  edge_type: string;
  label: string;
  weight: number;
  category: string;
}

export interface KnowledgeGraphData {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  metadata: {
    total_nodes?: number;
    total_edges?: number;
    contact_count?: number;
    entity_count?: number;
  };
}
