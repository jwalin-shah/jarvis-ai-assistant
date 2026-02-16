/**
 * Mock API data for E2E tests.
 *
 * These mock responses match the types defined in src/lib/api/types.ts
 * and replicate the structure returned by the FastAPI backend.
 */

import type {
  Conversation,
  HealthResponse,
  Message,
  ModelInfo,
  SettingsResponse,
} from "../../src/lib/api/types";

/**
 * Mock health response - healthy system
 */
export const mockHealthResponse: HealthResponse = {
  status: "healthy",
  imessage_access: true,
  memory_available_gb: 6.5,
  memory_used_gb: 9.5,
  memory_mode: "FULL",
  model_loaded: true,
  permissions_ok: true,
  details: null,
  jarvis_rss_mb: 512,
  jarvis_vms_mb: 2048,
};

/**
 * Mock health response - degraded system
 */
export const mockHealthResponseDegraded: HealthResponse = {
  status: "degraded",
  imessage_access: true,
  memory_available_gb: 2.0,
  memory_used_gb: 14.0,
  memory_mode: "LITE",
  model_loaded: false,
  permissions_ok: true,
  details: { memory: "Low memory, running in LITE mode" },
  jarvis_rss_mb: 256,
  jarvis_vms_mb: 1024,
};

/**
 * Mock conversations list
 */
export const mockConversations: Conversation[] = [
  {
    chat_id: "chat-1",
    participants: ["+1234567890"],
    display_name: "John Doe",
    last_message_date: new Date().toISOString(),
    message_count: 42,
    is_group: false,
    last_message_text: "Hey, are you free for lunch tomorrow?",
  },
  {
    chat_id: "chat-2",
    participants: ["+1987654321"],
    display_name: "Jane Smith",
    last_message_date: new Date(Date.now() - 3600000).toISOString(),
    message_count: 128,
    is_group: false,
    last_message_text: "Thanks for the help!",
  },
  {
    chat_id: "chat-3",
    participants: ["+1111111111", "+2222222222", "+3333333333"],
    display_name: "Project Team",
    last_message_date: new Date(Date.now() - 86400000).toISOString(),
    message_count: 256,
    is_group: true,
    last_message_text: "Meeting moved to 3pm",
  },
  {
    chat_id: "chat-4",
    participants: ["+4444444444"],
    display_name: null,
    last_message_date: new Date(Date.now() - 172800000).toISOString(),
    message_count: 15,
    is_group: false,
    last_message_text: "See you next week!",
  },
];

/**
 * Mock messages for chat-1
 */
export const mockMessagesChat1: Message[] = [
  {
    id: 1,
    chat_id: "chat-1",
    sender: "+1234567890",
    sender_name: "John Doe",
    text: "Hey, are you free for lunch tomorrow?",
    date: new Date().toISOString(),
    is_from_me: false,
    attachments: [],
    reply_to_id: null,
    reactions: [],
    date_delivered: null,
    date_read: null,
    is_system_message: false,
  },
  {
    id: 2,
    chat_id: "chat-1",
    sender: "me",
    sender_name: null,
    text: "Sure! What time works for you?",
    date: new Date(Date.now() - 300000).toISOString(),
    is_from_me: true,
    attachments: [],
    reply_to_id: null,
    reactions: [
      {
        type: "love",
        sender: "+1234567890",
        sender_name: "John Doe",
        date: new Date(Date.now() - 290000).toISOString(),
      },
    ],
    date_delivered: new Date(Date.now() - 299000).toISOString(),
    date_read: new Date(Date.now() - 298000).toISOString(),
    is_system_message: false,
  },
  {
    id: 3,
    chat_id: "chat-1",
    sender: "+1234567890",
    sender_name: "John Doe",
    text: "How about noon at the usual place?",
    date: new Date(Date.now() - 600000).toISOString(),
    is_from_me: false,
    attachments: [],
    reply_to_id: null,
    reactions: [],
    date_delivered: null,
    date_read: null,
    is_system_message: false,
  },
];

/**
 * Mock messages for chat-3 (group chat)
 */
export const mockMessagesChat3: Message[] = [
  {
    id: 10,
    chat_id: "chat-3",
    sender: "+1111111111",
    sender_name: "Alice",
    text: "Meeting moved to 3pm",
    date: new Date(Date.now() - 86400000).toISOString(),
    is_from_me: false,
    attachments: [],
    reply_to_id: null,
    reactions: [],
    date_delivered: null,
    date_read: null,
    is_system_message: false,
  },
  {
    id: 11,
    chat_id: "chat-3",
    sender: "+2222222222",
    sender_name: "Bob",
    text: "Works for me!",
    date: new Date(Date.now() - 86500000).toISOString(),
    is_from_me: false,
    attachments: [],
    reply_to_id: 10,
    reactions: [],
    date_delivered: null,
    date_read: null,
    is_system_message: false,
  },
  {
    id: 12,
    chat_id: "chat-3",
    sender: "me",
    sender_name: null,
    text: "I can make it too",
    date: new Date(Date.now() - 86600000).toISOString(),
    is_from_me: true,
    attachments: [],
    reply_to_id: null,
    reactions: [
      {
        type: "like",
        sender: "+1111111111",
        sender_name: "Alice",
        date: new Date(Date.now() - 86590000).toISOString(),
      },
    ],
    date_delivered: new Date(Date.now() - 86599000).toISOString(),
    date_read: new Date(Date.now() - 86598000).toISOString(),
    is_system_message: false,
  },
];

/**
 * Mock available models
 */
export const mockModels: ModelInfo[] = [
  {
    model_id: "qwen2.5-0.5b-instruct-4bit",
    name: "Qwen 2.5 0.5B Instruct (4-bit)",
    size_gb: 0.4,
    quality_tier: "basic",
    ram_requirement_gb: 4,
    is_downloaded: true,
    is_loaded: true,
    is_recommended: true,
    description: "Fast and lightweight model for quick responses",
  },
  {
    model_id: "qwen2.5-1.5b-instruct-4bit",
    name: "Qwen 2.5 1.5B Instruct (4-bit)",
    size_gb: 1.0,
    quality_tier: "good",
    ram_requirement_gb: 6,
    is_downloaded: true,
    is_loaded: false,
    is_recommended: false,
    description: "Balanced model with good quality and speed",
  },
  {
    model_id: "qwen2.5-3b-instruct-4bit",
    name: "Qwen 2.5 3B Instruct (4-bit)",
    size_gb: 2.0,
    quality_tier: "best",
    ram_requirement_gb: 8,
    is_downloaded: false,
    is_loaded: false,
    is_recommended: false,
    description: "Highest quality responses, requires more memory",
  },
];

/**
 * Mock settings response
 */
export const mockSettingsResponse: SettingsResponse = {
  model_id: "qwen2.5-0.5b-instruct-4bit",
  generation: {
    temperature: 0.7,
    max_tokens_reply: 150,
    max_tokens_summary: 500,
  },
  behavior: {
    auto_suggest_replies: true,
    suggestion_count: 3,
    context_messages_reply: 20,
    context_messages_summary: 50,
  },
  system: {
    system_ram_gb: 16,
    current_memory_usage_gb: 9.5,
    model_loaded: true,
    model_memory_usage_gb: 0.5,
    imessage_access: true,
  },
};

/**
 * Mock draft reply response
 */
export const mockDraftReplyResponse = {
  suggestions: [
    { text: "Sounds great! I'll be there.", confidence: 0.92 },
    { text: "Yes, noon works perfectly for me!", confidence: 0.87 },
    { text: "Looking forward to it. See you then!", confidence: 0.81 },
  ],
  context_used: {
    num_messages: 10,
    participants: ["John Doe"],
  },
};

/**
 * Mock API ping response
 */
export const mockPingResponse = {
  status: "ok",
  service: "jarvis-api",
};

/**
 * Mock smart reply suggestions (V2 API)
 */
export const mockSmartReplies = {
  replies: [
    {
      text: "Sounds great! I'll be there.",
      confidence: 0.92,
      tone: "friendly",
    },
    {
      text: "Yes, noon works perfectly for me!",
      confidence: 0.87,
      tone: "enthusiastic",
    },
    {
      text: "Looking forward to it. See you then!",
      confidence: 0.81,
      tone: "casual",
    },
  ],
  generation_time_ms: 245,
  model_used: "qwen2.5-0.5b-instruct-4bit",
};

/**
 * Mock search results
 */
export const mockSearchResults: Message[] = [
  {
    id: 100,
    chat_id: "chat-1",
    sender: "+1234567890",
    sender_name: "John Doe",
    text: "Do you want to grab lunch tomorrow?",
    date: new Date(Date.now() - 86400000).toISOString(),
    is_from_me: false,
    attachments: [],
    reply_to_id: null,
    reactions: [],
    date_delivered: null,
    date_read: null,
    is_system_message: false,
  },
  {
    id: 101,
    chat_id: "chat-2",
    sender: "me",
    sender_name: null,
    text: "I had a great lunch meeting yesterday",
    date: new Date(Date.now() - 172800000).toISOString(),
    is_from_me: true,
    attachments: [],
    reply_to_id: null,
    reactions: [],
    date_delivered: null,
    date_read: null,
    is_system_message: false,
  },
];

/**
 * Mock semantic search results
 */
export const mockSemanticSearchResults = {
  results: [
    {
      message: mockSearchResults[0],
      similarity: 0.85,
    },
    {
      message: mockSearchResults[1],
      similarity: 0.72,
    },
  ],
  query_embedding_time_ms: 45,
  search_time_ms: 12,
  total_messages_searched: 1000,
};

/**
 * Mock summary response
 */
export const mockSummaryResponse = {
  summary:
    "Discussion about lunch plans tomorrow. John suggested meeting at noon at the usual place. Both parties confirmed availability.",
  key_points: [
    "Lunch planned for tomorrow",
    "Meeting at noon",
    "Location: usual place",
  ],
  sentiment: "positive",
  messages_analyzed: 10,
  generation_time_ms: 320,
};

/**
 * Mock templates
 */
export const mockTemplates = [
  {
    id: "template-1",
    name: "Meeting Confirmation",
    content: "I'll be there at {{time}}. Looking forward to it!",
    category: "professional",
    variables: ["time"],
    created_at: new Date(Date.now() - 604800000).toISOString(),
    updated_at: new Date(Date.now() - 604800000).toISOString(),
    usage_count: 15,
  },
  {
    id: "template-2",
    name: "Running Late",
    content: "Sorry, I'm running about {{minutes}} minutes late. Be there soon!",
    category: "casual",
    variables: ["minutes"],
    created_at: new Date(Date.now() - 1209600000).toISOString(),
    updated_at: new Date(Date.now() - 86400000).toISOString(),
    usage_count: 8,
  },
  {
    id: "template-3",
    name: "Thank You",
    content: "Thank you so much for {{reason}}! Really appreciate it.",
    category: "gratitude",
    variables: ["reason"],
    created_at: new Date(Date.now() - 2592000000).toISOString(),
    updated_at: new Date(Date.now() - 2592000000).toISOString(),
    usage_count: 23,
  },
];

/**
 * Mock digest response
 */
export const mockDigestResponse = {
  period: "daily",
  generated_at: new Date().toISOString(),
  summary:
    "You had 15 conversations today. 3 require follow-up. Most active contact: John Doe.",
  highlights: [
    {
      type: "unread",
      count: 5,
      description: "Unread messages",
    },
    {
      type: "mentions",
      count: 2,
      description: "Messages mentioning you",
    },
    {
      type: "action_items",
      count: 3,
      description: "Action items detected",
    },
  ],
  top_contacts: [
    { name: "John Doe", message_count: 12 },
    { name: "Jane Smith", message_count: 8 },
    { name: "Project Team", message_count: 6 },
  ],
};

/**
 * Mock priority inbox items
 */
export const mockPriorityInbox = {
  items: [
    {
      id: "priority-1",
      chat_id: "chat-1",
      message_id: 1,
      reason: "question",
      confidence: 0.95,
      created_at: new Date().toISOString(),
      handled: false,
      message: mockMessagesChat1[0],
    },
    {
      id: "priority-2",
      chat_id: "chat-3",
      message_id: 10,
      reason: "time_sensitive",
      confidence: 0.88,
      created_at: new Date(Date.now() - 3600000).toISOString(),
      handled: false,
      message: mockMessagesChat3[0],
    },
  ],
  total_count: 2,
};

/**
 * Generate a large list of mock conversations for performance testing.
 */
export function generateLargeConversationList(count: number): Conversation[] {
  const conversations: Conversation[] = [];

  for (let i = 0; i < count; i++) {
    const isGroup = i % 5 === 0; // Every 5th is a group
    const hoursAgo = i * 2; // Spread over time

    conversations.push({
      chat_id: `chat-perf-${i}`,
      participants: isGroup
        ? [`+1${String(i).padStart(10, "0")}`, `+2${String(i).padStart(10, "0")}`]
        : [`+1${String(i).padStart(10, "0")}`],
      display_name: isGroup ? `Group ${i}` : `Contact ${i}`,
      last_message_date: new Date(Date.now() - hoursAgo * 3600000).toISOString(),
      message_count: Math.floor(Math.random() * 500) + 10,
      is_group: isGroup,
      last_message_text: `This is the last message for conversation ${i}`,
    });
  }

  return conversations;
}

/**
 * Generate a large list of mock messages for performance testing.
 */
export function generateLargeMessageList(count: number): Message[] {
  const messages: Message[] = [];

  for (let i = 0; i < count; i++) {
    const isFromMe = i % 3 === 0; // Every 3rd is from me
    const minutesAgo = i * 5; // 5 minutes between messages

    messages.push({
      id: 10000 + i,
      chat_id: "chat-perf-0",
      sender: isFromMe ? "me" : "+11234567890",
      sender_name: isFromMe ? null : "Test Contact",
      text: `This is message number ${i}. ${generateRandomText(50)}`,
      date: new Date(Date.now() - minutesAgo * 60000).toISOString(),
      is_from_me: isFromMe,
      attachments: i % 10 === 0 ? [{ type: "image", filename: "photo.jpg" }] : [],
      reply_to_id: i > 0 && i % 7 === 0 ? 10000 + i - 1 : null,
      reactions:
        i % 8 === 0
          ? [
              {
                type: "love",
                sender: "+11234567890",
                sender_name: "Test Contact",
                date: new Date(Date.now() - (minutesAgo - 1) * 60000).toISOString(),
              },
            ]
          : [],
      date_delivered: isFromMe
        ? new Date(Date.now() - (minutesAgo - 0.5) * 60000).toISOString()
        : null,
      date_read: isFromMe
        ? new Date(Date.now() - (minutesAgo - 0.3) * 60000).toISOString()
        : null,
      is_system_message: false,
    });
  }

  return messages;
}

/**
 * Helper to generate random text for messages.
 */
function generateRandomText(wordCount: number): string {
  const words = [
    "hello",
    "world",
    "test",
    "message",
    "great",
    "thanks",
    "sure",
    "okay",
    "sounds",
    "good",
    "meeting",
    "tomorrow",
    "lunch",
    "dinner",
    "project",
    "update",
    "review",
    "completed",
    "pending",
    "urgent",
  ];

  return Array.from({ length: wordCount }, () => words[Math.floor(Math.random() * words.length)])
    .join(" ")
    .trim();
}
