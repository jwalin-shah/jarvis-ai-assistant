<script lang="ts">
  import type { Tag, Conversation } from '../../api/types';
  import TagBadge from './TagBadge.svelte';
  import TagPicker from './TagPicker.svelte';
  import { createEventDispatcher } from 'svelte';
  import { formatParticipant } from '../../db';

  export let isOpen: boolean = false;
  export let tags: Tag[] = [];
  export let selectedConversations: Conversation[] = [];

  const dispatch = createEventDispatcher<{
    apply: { chatIds: string[]; tagIds: number[]; action: 'add' | 'remove' };
    cancel: void;
  }>();

  let action: 'add' | 'remove' = 'add';
  let selectedTagIds: number[] = [];
  let isProcessing = false;

  $: selectedTags = tags.filter((t) => selectedTagIds.includes(t.id));
  $: canApply = selectedTagIds.length > 0 && selectedConversations.length > 0;

  function handleApply() {
    if (!canApply) return;

    isProcessing = true;
    const chatIds = selectedConversations.map((c) => c.chat_id);

    dispatch('apply', {
      chatIds,
      tagIds: selectedTagIds,
      action,
    });

    // Reset state
    selectedTagIds = [];
    isProcessing = false;
  }

  function handleTagChange(event: CustomEvent<number[]>) {
    selectedTagIds = event.detail;
  }
</script>

{#if isOpen}
  <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
    <div class="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-lg">
      <!-- Header -->
      <div
        class="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700"
      >
        <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
          Bulk Tag {selectedConversations.length} Conversation{selectedConversations.length !== 1
            ? 's'
            : ''}
        </h2>
        <button
          type="button"
          class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          on:click={() => dispatch('cancel')}
          aria-label="Close bulk tagger"
        >
          <svg
            class="w-5 h-5"
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fill-rule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clip-rule="evenodd"
            />
          </svg>
        </button>
      </div>

      <!-- Content -->
      <div class="p-6 space-y-6">
        <!-- Selected Conversations Preview -->
        <div>
          <p class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Selected Conversations
          </p>
          <div
            class="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto p-2 bg-gray-50 dark:bg-gray-700 rounded-lg"
          >
            {#each selectedConversations.slice(0, 10) as conv (conv.chat_id)}
              <span
                class="px-2 py-0.5 text-xs bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-full"
              >
                {conv.display_name || conv.participants.map((p) => formatParticipant(p)).join(', ')}
              </span>
            {/each}
            {#if selectedConversations.length > 10}
              <span class="px-2 py-0.5 text-xs text-gray-500 dark:text-gray-400">
                +{selectedConversations.length - 10} more
              </span>
            {/if}
          </div>
        </div>

        <!-- Action Type -->
        <div>
          <p class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Action</p>
          <div class="flex gap-4">
            <label class="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                bind:group={action}
                value="add"
                class="text-blue-600 focus:ring-blue-500"
              />
              <span class="text-sm text-gray-700 dark:text-gray-300">Add tags</span>
            </label>
            <label class="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                bind:group={action}
                value="remove"
                class="text-blue-600 focus:ring-blue-500"
              />
              <span class="text-sm text-gray-700 dark:text-gray-300">Remove tags</span>
            </label>
          </div>
        </div>

        <!-- Tag Selection -->
        <div>
          <p class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Tags to {action}
          </p>
          <TagPicker
            {tags}
            {selectedTagIds}
            placeholder="Select tags..."
            on:change={handleTagChange}
          />
        </div>

        <!-- Selected Tags Preview -->
        {#if selectedTags.length > 0}
          <div>
            <p class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              {action === 'add' ? 'Tags to add' : 'Tags to remove'}
            </p>
            <div class="flex flex-wrap gap-1.5">
              {#each selectedTags as tag (tag.id)}
                <TagBadge {tag} size="sm" />
              {/each}
            </div>
          </div>
        {/if}

        <!-- Summary -->
        <div class="p-3 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
          <p class="text-sm text-blue-700 dark:text-blue-300">
            {#if canApply}
              This will {action} <strong>{selectedTagIds.length}</strong>
              tag{selectedTagIds.length !== 1 ? 's' : ''}
              {action === 'add' ? 'to' : 'from'} <strong>{selectedConversations.length}</strong>
              conversation{selectedConversations.length !== 1 ? 's' : ''}.
            {:else if selectedTagIds.length === 0}
              Select at least one tag to continue.
            {:else}
              No conversations selected.
            {/if}
          </p>
        </div>
      </div>

      <!-- Footer -->
      <div
        class="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900"
      >
        <button
          type="button"
          class="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          on:click={() => dispatch('cancel')}
        >
          Cancel
        </button>
        <button
          type="button"
          class="px-4 py-2 text-sm text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
          class:bg-blue-600={action === 'add'}
          class:hover:bg-blue-700={action === 'add'}
          class:bg-red-600={action === 'remove'}
          class:hover:bg-red-700={action === 'remove'}
          disabled={!canApply || isProcessing}
          on:click={handleApply}
        >
          {#if isProcessing}
            <svg
              class="animate-spin -ml-1 mr-2 h-4 w-4 inline-block"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                class="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                stroke-width="4"
              ></circle>
              <path
                class="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
            Processing...
          {:else}
            {action === 'add' ? 'Add Tags' : 'Remove Tags'}
          {/if}
        </button>
      </div>
    </div>
  </div>
{/if}
