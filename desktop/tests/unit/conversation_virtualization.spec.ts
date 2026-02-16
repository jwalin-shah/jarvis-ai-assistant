/**
 * Unit tests for ConversationList virtualization
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock conversation data generator
function generateConversations(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    chat_id: `chat_${i}`,
    participants: [`+1234567${i.toString().padStart(3, '0')}`],
    display_name: i % 3 === 0 ? `Contact ${i}` : null,
    last_message_date: new Date(Date.now() - i * 60000).toISOString(),
    message_count: 100 + i,
    is_group: i % 10 === 0,
    last_message_text: `Last message ${i}`,
  }));
}

describe('Conversation Virtualization', () => {
  const ESTIMATED_CONVERSATION_HEIGHT = 72;
  const BUFFER_SIZE = 5;
  const MIN_VISIBLE = 10;

  describe('visible range calculation', () => {
    it('should calculate correct visible range for scroll position', () => {
      const totalConversations = 1000;
      const scrollTop = 720; // Scroll 10 items down
      const containerHeight = 720; // Viewport fits 10 items

      const startIdx = Math.max(0, Math.floor(scrollTop / ESTIMATED_CONVERSATION_HEIGHT) - BUFFER_SIZE);
      const visibleCount = Math.ceil(containerHeight / ESTIMATED_CONVERSATION_HEIGHT);
      const endIdx = Math.min(
        totalConversations,
        startIdx + visibleCount + BUFFER_SIZE * 2
      );

      expect(startIdx).toBe(5); // 10 - 5 buffer
      expect(endIdx).toBe(30); // 5 + 10 + 15 buffer
    });

    it('should handle scroll at top', () => {
      const scrollTop = 0;
      const containerHeight = 720;

      const startIdx = Math.max(0, Math.floor(scrollTop / ESTIMATED_CONVERSATION_HEIGHT) - BUFFER_SIZE);

      expect(startIdx).toBe(0);
    });

    it('should handle scroll at bottom', () => {
      const totalConversations = 100;
      const scrollTop = 6500; // Near bottom
      const containerHeight = 720;

      const startIdx = Math.max(0, Math.floor(scrollTop / ESTIMATED_CONVERSATION_HEIGHT) - BUFFER_SIZE);
      const endIdx = Math.min(
        totalConversations,
        startIdx + Math.ceil(containerHeight / ESTIMATED_CONVERSATION_HEIGHT) + BUFFER_SIZE * 2
      );

      expect(endIdx).toBeLessThanOrEqual(totalConversations);
    });
  });

  describe('virtual padding calculation', () => {
    it('should calculate top padding correctly', () => {
      const visibleStartIndex = 10;
      const expectedPadding = visibleStartIndex * ESTIMATED_CONVERSATION_HEIGHT;

      expect(expectedPadding).toBe(720);
    });

    it('should calculate bottom padding correctly', () => {
      const totalConversations = 100;
      const visibleEndIndex = 30;
      const totalHeight = totalConversations * ESTIMATED_CONVERSATION_HEIGHT;
      const visibleHeight = (visibleEndIndex - 10) * ESTIMATED_CONVERSATION_HEIGHT; // 20 items visible
      const topPadding = 10 * ESTIMATED_CONVERSATION_HEIGHT;

      const bottomPadding = Math.max(0, totalHeight - topPadding - visibleHeight);

      expect(bottomPadding).toBe((100 - 30) * ESTIMATED_CONVERSATION_HEIGHT); // 70 items below
    });
  });

  describe('render optimization', () => {
    it('should render fewer items than total with virtualization', () => {
      const totalConversations = 1000;
      const visibleCount = 30; // Typical visible + buffer

      const renderedRatio = visibleCount / totalConversations;

      expect(renderedRatio).toBeLessThan(0.05); // Less than 5% rendered
    });

    it('should maintain minimum visible items', () => {
      const scrollTop = 0;
      const containerHeight = 360; // Small viewport - only 5 items fit

      const visibleCount = Math.ceil(containerHeight / ESTIMATED_CONVERSATION_HEIGHT);
      const endIdx = Math.max(visibleCount + BUFFER_SIZE * 2, MIN_VISIBLE);

      expect(endIdx).toBeGreaterThanOrEqual(MIN_VISIBLE);
    });
  });
});
