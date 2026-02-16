<script lang="ts">
  import type { Tag } from '../../api/types';
  import TagBadge from './TagBadge.svelte';
  import { createEventDispatcher, onMount } from 'svelte';

  export let tags: Tag[] = [];
  export let selectedTagIds: number[] = [];
  export let placeholder: string = 'Search tags...';
  export let maxSelections: number = 0; // 0 = unlimited
  export let allowCreate: boolean = false;
  export let disabled: boolean = false;

  const dispatch = createEventDispatcher<{
    change: number[];
    create: string;
  }>();

  let searchQuery = '';
  let isOpen = false;
  let inputElement: HTMLInputElement;
  let containerElement: HTMLDivElement;

  $: selectedTags = tags.filter((t) => selectedTagIds.includes(t.id));
  $: availableTags = tags.filter((t) => !selectedTagIds.includes(t.id));
  $: filteredTags = searchQuery
    ? availableTags.filter(
        (t) =>
          t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          t.aliases.some((a: string) => a.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    : availableTags;
  $: canSelectMore = maxSelections === 0 || selectedTagIds.length < maxSelections;
  $: showCreateOption =
    allowCreate &&
    searchQuery.trim() &&
    !tags.some((t) => t.name.toLowerCase() === searchQuery.toLowerCase());

  function toggleTag(tagId: number) {
    if (disabled) return;

    if (selectedTagIds.includes(tagId)) {
      selectedTagIds = selectedTagIds.filter((id) => id !== tagId);
    } else if (canSelectMore) {
      selectedTagIds = [...selectedTagIds, tagId];
    }
    dispatch('change', selectedTagIds);
  }

  function removeTag(tag: Tag) {
    if (disabled) return;
    selectedTagIds = selectedTagIds.filter((id) => id !== tag.id);
    dispatch('change', selectedTagIds);
  }

  function handleCreate() {
    if (searchQuery.trim()) {
      dispatch('create', searchQuery.trim());
      searchQuery = '';
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      isOpen = false;
      inputElement?.blur();
    } else if (e.key === 'Enter' && showCreateOption) {
      e.preventDefault();
      handleCreate();
    } else if (e.key === 'Backspace' && !searchQuery && selectedTagIds.length > 0) {
      // Remove last selected tag
      selectedTagIds = selectedTagIds.slice(0, -1);
      dispatch('change', selectedTagIds);
    }
  }

  function handleClickOutside(e: MouseEvent) {
    if (containerElement && !containerElement.contains(e.target as Node)) {
      isOpen = false;
    }
  }

  onMount(() => {
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  });
</script>

<div class="relative" bind:this={containerElement}>
  <div
    class="flex flex-wrap gap-1.5 p-2 border rounded-lg bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 min-h-[42px]"
    class:opacity-50={disabled}
    class:cursor-not-allowed={disabled}
    on:click={() => !disabled && inputElement?.focus()}
    role="presentation"
  >
    {#each selectedTags as tag (tag.id)}
      <TagBadge {tag} size="sm" removable={!disabled} on:remove={() => removeTag(tag)} />
    {/each}

    <input
      bind:this={inputElement}
      type="text"
      bind:value={searchQuery}
      {placeholder}
      {disabled}
      class="flex-1 min-w-[100px] bg-transparent outline-none text-sm text-gray-900 dark:text-white placeholder-gray-400"
      on:focus={() => (isOpen = true)}
      on:keydown={handleKeydown}
    />
  </div>

  {#if isOpen && !disabled}
    <div
      class="absolute z-50 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg max-h-60 overflow-auto"
    >
      {#if filteredTags.length === 0 && !showCreateOption}
        <div class="px-3 py-2 text-sm text-gray-500 dark:text-gray-400">No tags found</div>
      {/if}

      {#if showCreateOption}
        <button
          type="button"
          class="w-full px-3 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2 text-blue-600 dark:text-blue-400"
          on:click={handleCreate}
        >
          <svg
            class="w-4 h-4"
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fill-rule="evenodd"
              d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z"
              clip-rule="evenodd"
            />
          </svg>
          Create "{searchQuery}"
        </button>
      {/if}

      {#each filteredTags as tag (tag.id)}
        <button
          type="button"
          class="w-full px-3 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
          class:opacity-50={!canSelectMore}
          disabled={!canSelectMore}
          on:click={() => toggleTag(tag.id)}
        >
          <span class="w-3 h-3 rounded-full flex-shrink-0" style="background-color: {tag.color};"
          ></span>
          <span class="truncate text-gray-900 dark:text-white">{tag.name}</span>
          {#if tag.description}
            <span class="text-gray-400 dark:text-gray-500 text-xs truncate ml-auto">
              {tag.description}
            </span>
          {/if}
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  /* Hide scrollbar but keep functionality */
  div::-webkit-scrollbar {
    width: 6px;
  }
  div::-webkit-scrollbar-track {
    background: transparent;
  }
  div::-webkit-scrollbar-thumb {
    background-color: rgba(155, 155, 155, 0.5);
    border-radius: 3px;
  }
</style>
