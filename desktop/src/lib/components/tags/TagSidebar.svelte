<script lang="ts">
  import type { Tag, SmartFolder } from '../../api/types';
  import { createEventDispatcher } from 'svelte';

  export let tags: Tag[] = [];
  export let smartFolders: SmartFolder[] = [];
  export let selectedTagId: number | null = null;
  export let selectedFolderId: number | null = null;
  export let tagCounts: Record<number, number> = {};

  const dispatch = createEventDispatcher<{
    selectTag: number | null;
    selectFolder: number | null;
    createTag: void;
    createFolder: void;
    editTag: Tag;
    editFolder: SmartFolder;
  }>();

  let searchQuery = '';
  let showAllTags = false;

  // Organize tags into hierarchy
  $: rootTags = tags.filter((t) => !t.parent_id);
  $: filteredTags = searchQuery
    ? tags.filter(
        (t) =>
          t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          t.aliases.some((a: string) => a.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    : showAllTags
      ? tags
      : rootTags.slice(0, 10);

  $: defaultFolders = smartFolders.filter((f) => f.is_default);
  $: customFolders = smartFolders.filter((f) => !f.is_default);

  function getChildTags(parentId: number): Tag[] {
    return tags.filter((t) => t.parent_id === parentId);
  }

  function selectTag(tagId: number | null) {
    selectedTagId = tagId;
    selectedFolderId = null;
    dispatch('selectTag', tagId);
  }

  function selectFolder(folderId: number | null) {
    selectedFolderId = folderId;
    selectedTagId = null;
    dispatch('selectFolder', folderId);
  }

  function getFolderIcon(icon: string | undefined): string {
    const icons: Record<string, string> = {
      inbox:
        'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10',
      mail: 'M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z',
      flag: 'M3 21v-4m0 0V5a2 2 0 012-2h6.5l1 1H21l-3 6 3 6h-8.5l-1-1H5a2 2 0 00-2 2zm9-13.5V9',
      clock: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z',
      folder: 'M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z',
    };
    return (icon && icons[icon]) || icons.folder!;
  }
</script>

<div
  class="flex flex-col h-full bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 w-64"
>
  <!-- Search -->
  <div class="p-3 border-b border-gray-200 dark:border-gray-700">
    <div class="relative">
      <svg
        class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 20 20"
        fill="currentColor"
      >
        <path
          fill-rule="evenodd"
          d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"
          clip-rule="evenodd"
        />
      </svg>
      <input
        type="text"
        bind:value={searchQuery}
        placeholder="Search tags..."
        class="w-full pl-9 pr-3 py-1.5 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
      />
    </div>
  </div>

  <div class="flex-1 overflow-y-auto p-2 space-y-4">
    <!-- Smart Folders -->
    <div>
      <div class="flex items-center justify-between px-2 py-1">
        <h3 class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
          Smart Folders
        </h3>
        <button
          type="button"
          class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          on:click={() => dispatch('createFolder')}
          title="Create folder"
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
        </button>
      </div>

      <div class="space-y-0.5">
        {#each defaultFolders as folder (folder.id)}
          <button
            type="button"
            class="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm transition-colors"
            class:bg-blue-100={selectedFolderId === folder.id}
            class:dark:bg-blue-900={selectedFolderId === folder.id}
            class:text-blue-700={selectedFolderId === folder.id}
            class:dark:text-blue-300={selectedFolderId === folder.id}
            class:text-gray-700={selectedFolderId !== folder.id}
            class:dark:text-gray-300={selectedFolderId !== folder.id}
            class:hover:bg-gray-100={selectedFolderId !== folder.id}
            class:dark:hover:bg-gray-800={selectedFolderId !== folder.id}
            on:click={() => selectFolder(folder.id)}
          >
            <svg
              class="w-4 h-4 flex-shrink-0"
              style="color: {folder.color};"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="2"
            >
              <path stroke-linecap="round" stroke-linejoin="round" d={getFolderIcon(folder.icon)} />
            </svg>
            <span class="truncate">{folder.name}</span>
          </button>
        {/each}

        {#each customFolders as folder (folder.id)}
          <button
            type="button"
            class="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm transition-colors group"
            class:bg-blue-100={selectedFolderId === folder.id}
            class:dark:bg-blue-900={selectedFolderId === folder.id}
            class:text-blue-700={selectedFolderId === folder.id}
            class:dark:text-blue-300={selectedFolderId === folder.id}
            class:text-gray-700={selectedFolderId !== folder.id}
            class:dark:text-gray-300={selectedFolderId !== folder.id}
            class:hover:bg-gray-100={selectedFolderId !== folder.id}
            class:dark:hover:bg-gray-800={selectedFolderId !== folder.id}
            on:click={() => selectFolder(folder.id)}
          >
            <svg
              class="w-4 h-4 flex-shrink-0"
              style="color: {folder.color};"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="2"
            >
              <path stroke-linecap="round" stroke-linejoin="round" d={getFolderIcon(folder.icon)} />
            </svg>
            <span class="truncate flex-1 text-left">{folder.name}</span>
            <span
              role="button"
              tabindex="0"
              class="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              on:click|stopPropagation={() => dispatch('editFolder', folder)}
              on:keydown|stopPropagation={(e) =>
                (e.key === 'Enter' || e.key === ' ') && dispatch('editFolder', folder)}
            >
              <svg
                class="w-3.5 h-3.5"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z"
                />
              </svg>
            </span>
          </button>
        {/each}
      </div>
    </div>

    <!-- Tags -->
    <div>
      <div class="flex items-center justify-between px-2 py-1">
        <h3 class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
          Tags
        </h3>
        <button
          type="button"
          class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          on:click={() => dispatch('createTag')}
          title="Create tag"
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
        </button>
      </div>

      <!-- All Tags option -->
      <button
        type="button"
        class="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm transition-colors mb-1"
        class:bg-blue-100={selectedTagId === null && selectedFolderId === null}
        class:dark:bg-blue-900={selectedTagId === null && selectedFolderId === null}
        class:text-blue-700={selectedTagId === null && selectedFolderId === null}
        class:dark:text-blue-300={selectedTagId === null && selectedFolderId === null}
        class:text-gray-700={selectedTagId !== null || selectedFolderId !== null}
        class:dark:text-gray-300={selectedTagId !== null || selectedFolderId !== null}
        class:hover:bg-gray-100={selectedTagId !== null || selectedFolderId !== null}
        class:dark:hover:bg-gray-800={selectedTagId !== null || selectedFolderId !== null}
        on:click={() => selectTag(null)}
      >
        <svg
          class="w-4 h-4 text-gray-400"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fill-rule="evenodd"
            d="M17.707 9.293a1 1 0 010 1.414l-7 7a1 1 0 01-1.414 0l-7-7A.997.997 0 012 10V5a3 3 0 013-3h5c.256 0 .512.098.707.293l7 7zM5 6a1 1 0 100-2 1 1 0 000 2z"
            clip-rule="evenodd"
          />
        </svg>
        <span>All Tags</span>
      </button>

      <div class="space-y-0.5">
        {#each filteredTags as tag (tag.id)}
          <button
            type="button"
            class="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm transition-colors group"
            class:bg-blue-100={selectedTagId === tag.id}
            class:dark:bg-blue-900={selectedTagId === tag.id}
            class:text-blue-700={selectedTagId === tag.id}
            class:dark:text-blue-300={selectedTagId === tag.id}
            class:text-gray-700={selectedTagId !== tag.id}
            class:dark:text-gray-300={selectedTagId !== tag.id}
            class:hover:bg-gray-100={selectedTagId !== tag.id}
            class:dark:hover:bg-gray-800={selectedTagId !== tag.id}
            on:click={() => selectTag(tag.id)}
          >
            <span class="w-3 h-3 rounded-full flex-shrink-0" style="background-color: {tag.color};"
            ></span>
            <span class="truncate flex-1 text-left">{tag.name}</span>
            {#if tagCounts[tag.id]}
              <span class="text-xs text-gray-400 dark:text-gray-500">
                {tagCounts[tag.id]}
              </span>
            {/if}
            <span
              role="button"
              tabindex="0"
              class="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              on:click|stopPropagation={() => dispatch('editTag', tag)}
              on:keydown|stopPropagation={(e) =>
                (e.key === 'Enter' || e.key === ' ') && dispatch('editTag', tag)}
            >
              <svg
                class="w-3.5 h-3.5"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z"
                />
              </svg>
            </span>
          </button>

          <!-- Show children if this tag has any -->
          {#each getChildTags(tag.id) as childTag (childTag.id)}
            <button
              type="button"
              class="w-full flex items-center gap-2 px-2 py-1.5 pl-6 rounded-lg text-sm transition-colors group"
              class:bg-blue-100={selectedTagId === childTag.id}
              class:dark:bg-blue-900={selectedTagId === childTag.id}
              class:text-blue-700={selectedTagId === childTag.id}
              class:dark:text-blue-300={selectedTagId === childTag.id}
              class:text-gray-700={selectedTagId !== childTag.id}
              class:dark:text-gray-300={selectedTagId !== childTag.id}
              class:hover:bg-gray-100={selectedTagId !== childTag.id}
              class:dark:hover:bg-gray-800={selectedTagId !== childTag.id}
              on:click={() => selectTag(childTag.id)}
            >
              <span
                class="w-2.5 h-2.5 rounded-full flex-shrink-0"
                style="background-color: {childTag.color};"
              ></span>
              <span class="truncate flex-1 text-left">{childTag.name}</span>
              {#if tagCounts[childTag.id]}
                <span class="text-xs text-gray-400 dark:text-gray-500">
                  {tagCounts[childTag.id]}
                </span>
              {/if}
            </button>
          {/each}
        {/each}
      </div>

      {#if !searchQuery && rootTags.length > 10 && !showAllTags}
        <button
          type="button"
          class="w-full px-2 py-1.5 text-sm text-blue-600 dark:text-blue-400 hover:underline text-left"
          on:click={() => (showAllTags = true)}
        >
          Show all {rootTags.length} tags...
        </button>
      {/if}
    </div>
  </div>

  <!-- Keyboard shortcuts hint -->
  <div
    class="p-2 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-400 dark:text-gray-500"
  >
    <span class="font-mono bg-gray-100 dark:bg-gray-800 px-1 rounded">T</span> Quick tag
  </div>
</div>
