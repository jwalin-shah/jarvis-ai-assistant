<script lang="ts">
  import type { SmartFolder, SmartFolderRules, RuleCondition, Tag } from "../../api/types";
  import { TAG_COLORS, RULE_FIELDS, RULE_OPERATORS } from "../../api/types";
  import { createEventDispatcher } from "svelte";

  export let folder: SmartFolder | null = null;
  export let tags: Tag[] = [];
  export let isOpen: boolean = false;

  const dispatch = createEventDispatcher<{
    save: { folder: SmartFolder; isNew: boolean };
    delete: SmartFolder;
    cancel: void;
    preview: SmartFolderRules;
  }>();

  // Form state
  let name = "";
  let icon = "folder";
  let color: string = TAG_COLORS.SLATE;
  let matchType: "all" | "any" = "all";
  let conditions: RuleCondition[] = [];
  let sortBy = "last_message_date";
  let sortOrder: "asc" | "desc" = "desc";
  let limitResults = 0;

  // Field labels for display
  const fieldLabels: Record<string, string> = {
    chat_id: "Chat ID",
    display_name: "Contact Name",
    last_message_date: "Last Message Date",
    message_count: "Message Count",
    is_group: "Is Group Chat",
    unread_count: "Unread Count",
    is_flagged: "Is Flagged",
    relationship: "Relationship",
    contact_name: "Contact Name",
    last_message_text: "Last Message Text",
    has_attachments: "Has Attachments",
    tags: "Tags",
    sentiment: "Sentiment",
    priority: "Priority",
    needs_response: "Needs Response",
  };

  // Operator labels
  const operatorLabels: Record<string, string> = {
    equals: "equals",
    not_equals: "does not equal",
    contains: "contains",
    not_contains: "does not contain",
    starts_with: "starts with",
    ends_with: "ends with",
    is_empty: "is empty",
    is_not_empty: "is not empty",
    greater_than: "is greater than",
    less_than: "is less than",
    in_last_days: "in last N days",
    before: "is before",
    after: "is after",
    has_tag: "has tag",
    has_any_tag: "has any of tags",
    has_all_tags: "has all tags",
    has_no_tags: "has no tags",
  };

  const colorOptions = Object.entries(TAG_COLORS);
  const iconOptions = [
    { value: "folder", label: "Folder" },
    { value: "inbox", label: "Inbox" },
    { value: "mail", label: "Mail" },
    { value: "flag", label: "Flag" },
    { value: "clock", label: "Clock" },
    { value: "star", label: "Star" },
    { value: "heart", label: "Heart" },
    { value: "users", label: "Users" },
    { value: "briefcase", label: "Briefcase" },
  ];

  $: isEditing = folder !== null;
  $: if (isOpen) {
    if (folder) {
      // Load existing folder data
      name = folder.name;
      icon = folder.icon;
      color = folder.color;
      matchType = folder.rules.match;
      conditions = [...folder.rules.conditions];
      sortBy = folder.rules.sort_by;
      sortOrder = folder.rules.sort_order;
      limitResults = folder.rules.limit;
    } else {
      // Reset to defaults for new folder
      name = "";
      icon = "folder";
      color = TAG_COLORS.SLATE;
      matchType = "all";
      conditions = [];
      sortBy = "last_message_date";
      sortOrder = "desc";
      limitResults = 0;
    }
  }

  function addCondition() {
    conditions = [
      ...conditions,
      { field: "display_name", operator: "contains", value: "" },
    ];
  }

  function removeCondition(index: number) {
    conditions = conditions.filter((_, i) => i !== index);
  }

  function updateCondition(index: number, field: keyof RuleCondition, value: unknown) {
    conditions = conditions.map((c, i) =>
      i === index ? { ...c, [field]: value } : c
    );
  }

  function getValueInputType(operator: string): "text" | "number" | "date" | "boolean" | "tags" | "none" {
    if (["is_empty", "is_not_empty", "has_no_tags"].includes(operator)) return "none";
    if (["has_tag", "has_any_tag", "has_all_tags"].includes(operator)) return "tags";
    if (["in_last_days", "greater_than", "less_than"].includes(operator)) return "number";
    if (["before", "after"].includes(operator)) return "date";
    if (["equals", "not_equals"].includes(operator)) {
      // Could be boolean for certain fields
      return "text";
    }
    return "text";
  }

  function buildRules(): SmartFolderRules {
    return {
      match: matchType,
      conditions,
      sort_by: sortBy,
      sort_order: sortOrder,
      limit: limitResults,
    };
  }

  function handleSave() {
    if (!name.trim()) return;

    const rules = buildRules();
    const savedFolder: SmartFolder = {
      id: folder?.id ?? 0,
      name: name.trim(),
      icon,
      color,
      rules,
      sort_order: folder?.sort_order ?? 0,
      is_default: false,
      created_at: folder?.created_at ?? null,
      updated_at: null,
    };

    dispatch("save", { folder: savedFolder, isNew: !folder });
  }

  function handlePreview() {
    dispatch("preview", buildRules());
  }

  function handleDelete() {
    if (folder && confirm(`Delete smart folder "${folder.name}"?`)) {
      dispatch("delete", folder);
    }
  }
</script>

{#if isOpen}
  <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
    <div class="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
      <!-- Header -->
      <div class="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
          {isEditing ? "Edit Smart Folder" : "Create Smart Folder"}
        </h2>
        <button
          type="button"
          class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          on:click={() => dispatch("cancel")}
        >
          <svg class="w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
          </svg>
        </button>
      </div>

      <!-- Content -->
      <div class="p-6 overflow-y-auto max-h-[calc(90vh-140px)] space-y-6">
        <!-- Basic Info -->
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Name
            </label>
            <input
              type="text"
              bind:value={name}
              placeholder="My Smart Folder"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Icon
            </label>
            <select
              bind:value={icon}
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
            >
              {#each iconOptions as opt}
                <option value={opt.value}>{opt.label}</option>
              {/each}
            </select>
          </div>
        </div>

        <!-- Color -->
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Color
          </label>
          <div class="flex flex-wrap gap-2">
            {#each colorOptions as [colorName, colorValue]}
              <button
                type="button"
                class="w-6 h-6 rounded-full border-2 transition-all"
                class:border-gray-900={color === colorValue}
                class:dark:border-white={color === colorValue}
                class:border-transparent={color !== colorValue}
                class:scale-110={color === colorValue}
                style="background-color: {colorValue};"
                on:click={() => (color = colorValue)}
                title={colorName}
              ></button>
            {/each}
          </div>
        </div>

        <!-- Match Type -->
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Match
          </label>
          <div class="flex gap-4">
            <label class="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                bind:group={matchType}
                value="all"
                class="text-blue-600 focus:ring-blue-500"
              />
              <span class="text-sm text-gray-700 dark:text-gray-300">All conditions</span>
            </label>
            <label class="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                bind:group={matchType}
                value="any"
                class="text-blue-600 focus:ring-blue-500"
              />
              <span class="text-sm text-gray-700 dark:text-gray-300">Any condition</span>
            </label>
          </div>
        </div>

        <!-- Conditions -->
        <div>
          <div class="flex items-center justify-between mb-2">
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Conditions
            </label>
            <button
              type="button"
              class="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400"
              on:click={addCondition}
            >
              + Add condition
            </button>
          </div>

          {#if conditions.length === 0}
            <p class="text-sm text-gray-500 dark:text-gray-400 italic">
              No conditions - folder will include all conversations
            </p>
          {:else}
            <div class="space-y-2">
              {#each conditions as condition, index}
                <div class="flex items-center gap-2 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <select
                    value={condition.field}
                    on:change={(e) => updateCondition(index, "field", e.currentTarget.value)}
                    class="px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                  >
                    {#each RULE_FIELDS as field}
                      <option value={field}>{fieldLabels[field] || field}</option>
                    {/each}
                  </select>

                  <select
                    value={condition.operator}
                    on:change={(e) => updateCondition(index, "operator", e.currentTarget.value)}
                    class="px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                  >
                    {#each RULE_OPERATORS as op}
                      <option value={op}>{operatorLabels[op] || op}</option>
                    {/each}
                  </select>

                  {#if getValueInputType(condition.operator) === "text"}
                    <input
                      type="text"
                      value={condition.value || ""}
                      on:input={(e) => updateCondition(index, "value", e.currentTarget.value)}
                      placeholder="Value"
                      class="flex-1 px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                    />
                  {:else if getValueInputType(condition.operator) === "number"}
                    <input
                      type="number"
                      value={condition.value || 0}
                      on:input={(e) => updateCondition(index, "value", parseInt(e.currentTarget.value))}
                      class="w-24 px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                    />
                  {:else if getValueInputType(condition.operator) === "date"}
                    <input
                      type="date"
                      value={condition.value || ""}
                      on:input={(e) => updateCondition(index, "value", e.currentTarget.value)}
                      class="px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                    />
                  {:else if getValueInputType(condition.operator) === "tags"}
                    <select
                      value={condition.value}
                      on:change={(e) => updateCondition(index, "value", parseInt(e.currentTarget.value))}
                      class="flex-1 px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                    >
                      {#each tags as tag}
                        <option value={tag.id}>{tag.name}</option>
                      {/each}
                    </select>
                  {/if}

                  <button
                    type="button"
                    class="text-gray-400 hover:text-red-500"
                    on:click={() => removeCondition(index)}
                  >
                    <svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
                    </svg>
                  </button>
                </div>
              {/each}
            </div>
          {/if}
        </div>

        <!-- Sort Options -->
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Sort by
            </label>
            <select
              bind:value={sortBy}
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
            >
              <option value="last_message_date">Last Message Date</option>
              <option value="display_name">Contact Name</option>
              <option value="message_count">Message Count</option>
              <option value="unread_count">Unread Count</option>
            </select>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Order
            </label>
            <select
              bind:value={sortOrder}
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
            >
              <option value="desc">Descending</option>
              <option value="asc">Ascending</option>
            </select>
          </div>
        </div>

        <!-- Limit -->
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Limit results (0 = unlimited)
          </label>
          <input
            type="number"
            bind:value={limitResults}
            min="0"
            max="1000"
            class="w-32 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      <!-- Footer -->
      <div class="flex items-center justify-between px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
        <div>
          {#if isEditing && !folder?.is_default}
            <button
              type="button"
              class="text-red-600 hover:text-red-700 dark:text-red-400 text-sm"
              on:click={handleDelete}
            >
              Delete folder
            </button>
          {/if}
        </div>

        <div class="flex items-center gap-3">
          <button
            type="button"
            class="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
            on:click={handlePreview}
          >
            Preview
          </button>
          <button
            type="button"
            class="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
            on:click={() => dispatch("cancel")}
          >
            Cancel
          </button>
          <button
            type="button"
            class="px-4 py-2 text-sm text-white bg-blue-600 hover:bg-blue-700 rounded-lg disabled:opacity-50"
            disabled={!name.trim()}
            on:click={handleSave}
          >
            {isEditing ? "Save Changes" : "Create Folder"}
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}
