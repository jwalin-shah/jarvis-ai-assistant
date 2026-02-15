<script lang="ts">
  import { tick } from "svelte";
  import { jarvis } from "../socket/client";

  interface ChatMessage {
    role: "user" | "assistant";
    content: string;
  }

  let messages = $state<ChatMessage[]>([]);
  let inputText = $state("");
  let isGenerating = $state(false);
  let streamingContent = $state("");
  let messagesAreaRef = $state<HTMLDivElement | null>(null);
  let inputRef = $state<HTMLTextAreaElement | null>(null);

  async function scrollToBottom(instant = false) {
    await tick();
    if (messagesAreaRef) {
      messagesAreaRef.scrollTo({
        top: messagesAreaRef.scrollHeight,
        behavior: instant ? "instant" : "smooth",
      });
    }
  }

  async function sendMessage() {
    const text = inputText.trim();
    if (!text || isGenerating) return;

    // Add user message
    messages.push({ role: "user", content: text });
    inputText = "";
    isGenerating = true;
    streamingContent = "";
    await scrollToBottom(true);

    // Reset textarea height
    if (inputRef) inputRef.style.height = "auto";

    // Safety timeout to prevent stuck input (30 seconds)
    const safetyTimeout = setTimeout(() => {
      if (isGenerating) {
        console.warn("[Chat] Safety timeout triggered - resetting input state");
        isGenerating = false;
        messages.push({
          role: "assistant",
          content: "Response timed out. Please try again.",
        });
      }
    }, 30000);

    try {
      // Build history from prior messages (exclude current)
      const history = messages.slice(0, -1).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      console.log("[Chat] Starting chatStream with history length:", history.length);
      
      await jarvis.chatStream(text, history, (token) => {
        streamingContent += token;
        // Auto-scroll during streaming
        if (messagesAreaRef) {
          messagesAreaRef.scrollTop = messagesAreaRef.scrollHeight;
        }
      });

      console.log("[Chat] chatStream completed successfully");
      
      // Finalize assistant message
      messages.push({ role: "assistant", content: streamingContent.trim() });
      streamingContent = "";
    } catch (error) {
      console.error("[Chat] Error in chatStream:", error);
      messages.push({
        role: "assistant",
        content: "Sorry, something went wrong. Please try again.",
      });
    } finally {
      clearTimeout(safetyTimeout);
      isGenerating = false;
      await scrollToBottom();
      inputRef?.focus();
    }
  }

  function clearChat() {
    messages = [];
    streamingContent = "";
    isGenerating = false;
    inputRef?.focus();
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function autoResize(event: Event) {
    const target = event.target as HTMLTextAreaElement;
    target.style.height = "auto";
    target.style.height = Math.min(target.scrollHeight, 150) + "px";
  }
</script>

<div class="chat-container">
  <div class="chat-header">
    <div class="header-title">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
      <span>Chat</span>
    </div>
    <button class="new-chat-btn" onclick={clearChat} title="New Chat">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
        <line x1="12" y1="5" x2="12" y2="19"></line>
        <line x1="5" y1="12" x2="19" y2="12"></line>
      </svg>
      New Chat
    </button>
  </div>

  <div class="messages-area" bind:this={messagesAreaRef}>
    {#if messages.length === 0 && !streamingContent}
      <div class="empty-state">
        <div class="empty-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </div>
        <h3>Chat with JARVIS</h3>
        <p>Ask anything. Your conversation is local and ephemeral.</p>
      </div>
    {:else}
      {#each messages as msg}
        <div class="message" class:user={msg.role === "user"} class:assistant={msg.role === "assistant"}>
          <div class="bubble">
            {msg.content}
          </div>
        </div>
      {/each}
      {#if streamingContent}
        <div class="message assistant">
          <div class="bubble streaming">
            {streamingContent}<span class="cursor"></span>
          </div>
        </div>
      {/if}
    {/if}
  </div>

  <div class="input-area">
    <div class="input-wrapper">
      <textarea
        bind:this={inputRef}
        bind:value={inputText}
        onkeydown={handleKeydown}
        oninput={autoResize}
        placeholder="Type a message..."
        rows="1"
        disabled={isGenerating}
        aria-label="Type a message"
      ></textarea>
      <button
        class="send-btn"
        onclick={sendMessage}
        disabled={!inputText.trim() || isGenerating}
        title="Send (Enter)"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      </button>
    </div>
    <span class="input-hint">Shift+Enter for new line</span>
  </div>
</div>

<style>
  .chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }

  .chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-3) var(--space-4);
    border-bottom: 1px solid var(--border-default);
    background: var(--surface-elevated);
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--text-lg);
    font-weight: var(--font-weight-semibold);
    color: var(--text-primary);
  }

  .new-chat-btn {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    padding: var(--space-1) var(--space-3);
    background: var(--surface-base);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    font-size: var(--text-sm);
    cursor: pointer;
    transition: all var(--duration-fast) var(--ease-out);
  }

  .new-chat-btn:hover {
    background: var(--surface-hover);
    color: var(--text-primary);
    border-color: var(--color-primary);
  }

  .messages-area {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }

  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-3);
    color: var(--text-tertiary);
    text-align: center;
    padding: var(--space-8);
  }

  .empty-icon {
    opacity: 0.3;
  }

  .empty-state h3 {
    color: var(--text-secondary);
    font-size: var(--text-lg);
    margin: 0;
  }

  .empty-state p {
    font-size: var(--text-sm);
    margin: 0;
  }

  .message {
    display: flex;
    max-width: 80%;
  }

  .message.user {
    align-self: flex-end;
  }

  .message.assistant {
    align-self: flex-start;
  }

  .bubble {
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-lg);
    font-size: var(--text-sm);
    line-height: var(--line-height-relaxed);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .message.user .bubble {
    background: var(--color-primary);
    color: white;
    border-bottom-right-radius: var(--radius-sm);
  }

  .message.assistant .bubble {
    background: var(--surface-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-default);
    border-bottom-left-radius: var(--radius-sm);
  }

  .bubble.streaming {
    min-height: 1.5em;
  }

  .cursor {
    display: inline-block;
    width: 2px;
    height: 1em;
    background: var(--text-secondary);
    margin-left: 1px;
    vertical-align: text-bottom;
    animation: blink 0.8s step-end infinite;
  }

  @keyframes blink {
    50% { opacity: 0; }
  }

  .input-area {
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--border-default);
    background: var(--surface-elevated);
  }

  .input-wrapper {
    display: flex;
    align-items: flex-end;
    gap: var(--space-2);
    background: var(--surface-base);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    padding: var(--space-2);
    transition: border-color var(--duration-fast) var(--ease-out);
  }

  .input-wrapper:focus-within {
    border-color: var(--color-primary);
  }

  textarea {
    flex: 1;
    border: none;
    outline: none;
    background: transparent;
    color: var(--text-primary);
    font-family: var(--font-family-sans);
    font-size: var(--text-sm);
    line-height: var(--line-height-normal);
    resize: none;
    padding: var(--space-1) var(--space-2);
    max-height: 150px;
  }

  textarea::placeholder {
    color: var(--text-tertiary);
  }

  textarea:disabled {
    opacity: 0.5;
  }

  .send-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border: none;
    border-radius: var(--radius-md);
    background: var(--color-primary);
    color: white;
    cursor: pointer;
    flex-shrink: 0;
    transition: opacity var(--duration-fast) var(--ease-out);
  }

  .send-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .send-btn:not(:disabled):hover {
    opacity: 0.9;
  }

  .input-hint {
    display: block;
    margin-top: var(--space-1);
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    text-align: right;
  }
</style>
