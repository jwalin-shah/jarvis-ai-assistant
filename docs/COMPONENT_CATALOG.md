# JARVIS Desktop Component Catalog

A comprehensive guide to all Svelte components in the JARVIS desktop application.

## Table of Contents

1. [Component Overview](#component-overview)
2. [Component Reference](#component-reference)
3. [Component Tree](#component-tree)
4. [Store Dependencies](#store-dependencies)
5. [Style Tokens](#style-tokens)
6. [Missing Components](#missing-components)

---

## Component Overview

The JARVIS desktop app contains **41 Svelte components** organized into the following categories:

| Category | Count | Purpose |
|----------|-------|---------|
| **Core UI** | 8 | Layout, navigation, and fundamental UI elements |
| **Message Views** | 4 | Conversation display and message composition |
| **Graph Visualization** | 5 | Relationship network D3.js visualization |
| **Tag Management** | 5 | Tagging system and smart folders |
| **Icons** | 5 + 1 | SVG icon components and icon library |
| **Utility** | 6 | Skeleton loaders, error boundaries, etc. |
| **Overlays** | 7 | Modals, search, command palette, etc. |

---

## Component Reference

### Core UI Components

#### App.svelte
**File:** `desktop/src/App.svelte`

The root application component that manages global state and view routing.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `currentView` | `$state<'messages' \| 'dashboard' \| 'health' \| 'settings' \| 'templates' \| 'network'>` | Active view state |
| `showSearch` | `$state<boolean>` | Global search modal visibility |
| `showShortcuts` | `$state<boolean>` | Keyboard shortcuts modal visibility |
| `showCommandPalette` | `$state<boolean>` | Command palette visibility |
| `sidebarCollapsed` | `$state<boolean>` | Sidebar collapse state |

**Dependencies:**
- `Sidebar`, `ConversationList`, `MessageView`, `GlobalSearch`, `ErrorBoundary`, `KeyboardShortcuts`, `CommandPalette`, `Toast`

**Store Dependencies:**
- `healthStore` (checkApiConnection)
- `conversationsStore` (clearSelection)
- `theme` (initializeTheme)
- `keyboard` (initAnnouncer)

**Screenshot Description:**
Full-screen layout with a collapsible sidebar on the left, main content area (changes based on view), and floating overlays for search/command palette. Dark theme with glass morphism effects.

---

#### Sidebar.svelte
**File:** `desktop/src/lib/components/Sidebar.svelte`

Navigation sidebar with collapsible state and view switching.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `currentView` | `$bindable<ViewType>` | Two-way bound active view |
| `collapsed` | `$bindable<boolean>` | Two-way bound collapse state |

**Events:** None

**Slots:** None

**Dependencies:** None

**Store Dependencies:** `healthStore` (connection status dot)

**Screenshot Description:**
200px wide (60px collapsed) vertical sidebar with JARVIS logo, 6 navigation buttons with icons (Dashboard, Messages, Templates, Network, Health, Settings), and connection status at bottom. Collapses to icon-only mode.

---

#### Button.svelte (Root)
**File:** `desktop/src/lib/components/Button.svelte`

Reusable button component with multiple variants.

**Props/Exports:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `variant` | `'primary' \| 'secondary' \| 'ghost' \| 'danger'` | `'secondary'` | Visual style |
| `size` | `'sm' \| 'md' \| 'lg'` | `'md'` | Button size |
| `loading` | `boolean` | `false` | Show loading spinner |
| `fullWidth` | `boolean` | `false` | Full width button |
| `children` | `Snippet` | - | Button content |

**Dependencies:** None

**Screenshot Description:**
Rounded rectangle buttons with hover lift effect. Primary=blue, Secondary=elevated dark, Ghost=transparent, Danger=red. Includes loading spinner animation.

---

#### Button.svelte (UI)
**File:** `desktop/src/lib/components/ui/Button.svelte`

Same API as root Button.svelte - unified button component.

---

#### EmptyState.svelte
**File:** `desktop/src/lib/components/ui/EmptyState.svelte`

Empty state placeholder for empty lists/views.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `title` | `string` | Main heading |
| `description` | `string` | Optional subtext |
| `icon` | `Snippet` | Icon to display |
| `action` | `{ label, onClick, variant }` | Optional action button |

**Dependencies:** `Button`

**Screenshot Description:**
Centered layout with 64px circular icon container, title, description text, and optional action button. Used in MessageView when no conversation selected.

---

#### Icon.svelte
**File:** `desktop/src/lib/components/Icon.svelte`

Lucide-style SVG icon component with built-in icons.

**Props/Exports:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `IconName` | - | Icon identifier |
| `size` | `number` | `20` | Icon size in px |
| `strokeWidth` | `number` | `1.5` | Stroke width |

**Icon Names:** sparkles, x-circle, check, alert-circle, send, refresh-cw, copy, clipboard-check, message-circle, settings, bar-chart-2, search, chevron-down, chevron-up, user, users, sun, moon

**Dependencies:** None

---

#### Skeleton.svelte
**File:** `desktop/src/lib/components/Skeleton.svelte`

Loading placeholder with shimmer animation.

**Props/Exports:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `width` | `string` | `'100%'` | Width CSS value |
| `height` | `string` | `'16px'` | Height CSS value |
| `borderRadius` | `string` | `'4px'` | Border radius |
| `animated` | `boolean` | `true` | Shimmer animation |

---

#### ErrorBoundary.svelte
**File:** `desktop/src/lib/components/ErrorBoundary.svelte`

Catches JavaScript errors and displays fallback UI.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `children` | `Snippet` | Child content to wrap |

**Events:** None (handles window errors)

**Store Dependencies:** `toast` (error notifications)

**Dependencies:** `Icon`

**Screenshot Description:**
Full-screen overlay with centered error card containing alert icon, "Something went wrong" heading, error message, expandable technical details, and Reload/Dismiss buttons.

---

### Message View Components

#### MessageView.svelte
**File:** `desktop/src/lib/components/MessageView.svelte`

Main conversation display with message list, compose area, and AI suggestions.

**Props/Exports:** None (uses stores)

**Dependencies:** 
- `SuggestionBar`, `MessageSkeleton`, `MessageItem`, `DateHeader`, `ComposeArea`, `EmptyState`, `MessageIcon`

**Store Dependencies:**
- `conversationsStore` (messages, loading states, optimistic messages)
- `keyboard` (activeZone, messageIndex)
- `selectedConversation` (current conversation)
- `highlightedMessageId`, `scrollToMessageId`

**Screenshot Description:**
Three-panel layout: header with avatar/contact info and AI Draft button, scrollable message list with date headers, floating suggestion bar (when open), and compose area at bottom. Uses virtual scrolling for performance.

---

#### MessageItem.svelte (message-view)
**File:** `desktop/src/lib/components/message-view/MessageItem.svelte`

Individual message bubble with optimistic UI support.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `message` | `Message` | Message data |
| `isGroup` | `boolean` | Group chat flag |
| `isHighlighted` | `boolean` | Search result highlight |
| `isKeyboardFocused` | `boolean` | Keyboard nav focus |
| `isNew` | `boolean` | New message animation |
| `onRetry` | `(id: string) => void` | Retry failed message |
| `onDismiss` | `(id: string) => void` | Dismiss failed message |

**Dependencies:** None

**Screenshot Description:**
Chat bubble with rounded corners, different styling for "from me" (blue gradient, right aligned) vs "from them" (gray, left aligned). Shows sender name in groups, timestamp, attachments, reactions, and retry actions for failed messages.

---

#### ComposeArea.svelte
**File:** `desktop/src/lib/components/message-view/ComposeArea.svelte`

Message input with auto-resize and send functionality.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `value` | `string` | Input value (bindable) |
| `disabled` | `boolean` | Disable input |
| `sending` | `boolean` | Sending state |
| `placeholder` | `string` | Placeholder text |
| `onSend` | `(text: string) => void` | Send callback |
| `onFocus` | `() => void` | Focus callback |

**Methods:**
- `focus()` - Focus the textarea
- `setValue(text)` - Set value and resize

**Dependencies:** `SendIcon`, `Button`

**Screenshot Description:**
Rounded input bar with auto-expanding textarea (max 120px), placeholder "iMessage", and send button. White/elevated background with subtle border.

---

#### DateHeader.svelte
**File:** `desktop/src/lib/components/message-view/DateHeader.svelte`

Date separator in message lists.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `date` | `string` | ISO date string |

**Screenshot Description:**
Centered pill-shaped label showing "Today", "Yesterday", or full date. Gray background, small text.

---

### Conversation Components

#### ConversationList.svelte
**File:** `desktop/src/lib/components/ConversationList.svelte`

Sidebar list of conversations with topic tags and avatars.

**Props/Exports:** None

**Dependencies:** `ConversationSkeleton`

**Store Dependencies:**
- `conversationsStore` (conversation list, loading, selected)
- `keyboard` (activeZone, conversationIndex, announce)

**Screenshot Description:**
300px wide panel with "Messages" header, search input, and scrollable list. Each item shows circular avatar (with lazy loading), contact name, topic tags (colored pills), and message preview. Selected state with accent color border.

---

#### ConversationSkeleton.svelte
**File:** `desktop/src/lib/components/ConversationSkeleton.svelte`

Loading placeholder for conversation list.

**Props/Exports:** None

**Dependencies:** `Skeleton`

**Screenshot Description:**
8 skeleton rows with circular avatar placeholder and two lines of text at varying widths.

---

#### MessageItem.svelte (root)
**File:** `desktop/src/lib/components/MessageItem.svelte`

Legacy message component (deprecated in favor of message-view/MessageItem).

---

#### MessageSkeleton.svelte
**File:** `desktop/src/lib/components/MessageSkeleton.svelte`

Loading placeholder for message list.

**Props/Exports:** None

**Dependencies:** `Skeleton`

**Screenshot Description:**
6 skeleton message bubbles at varying widths, alternating left/right alignment to simulate conversation.

---

### Overlay Components

#### GlobalSearch.svelte
**File:** `desktop/src/lib/components/GlobalSearch.svelte`

Modal search with text and semantic (AI) search modes.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `onClose` | `() => void` | Close callback |

**Dependencies:** None

**Store Dependencies:** `conversationsStore` (for conversation names)

**Screenshot Description:**
Centered modal with search input, mode toggle (keyword/semantic), filter panel (sender, dates, attachments), and grouped results by conversation. Keyboard navigable with up/down/enter.

---

#### CommandPalette.svelte
**File:** `desktop/src/lib/components/CommandPalette.svelte`

Quick command launcher with keyboard navigation.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `onClose` | `() => void` | Close callback |
| `onNavigate` | `(view) => void` | View navigation callback |
| `onOpenSearch` | `() => void` | Open search callback |
| `onOpenShortcuts` | `() => void` | Open shortcuts callback |

**Dependencies:** `Icon`

**Store Dependencies:** `theme` (setTheme for theme commands)

**Screenshot Description:**
Centered glass-morphism modal with search input, categorized command list with icons and keyboard shortcuts, and footer with navigation hints. Commands grouped by Navigation, Actions, Theme.

---

#### KeyboardShortcuts.svelte
**File:** `desktop/src/lib/components/KeyboardShortcuts.svelte`

Keyboard shortcuts reference modal.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `onClose` | `() => void` | Close callback |

**Dependencies:** `Icon`

**Screenshot Description:**
Centered modal with 2-column grid of shortcut categories (Navigation, List Navigation, Messages, General). Each shortcut shows key combination pills (e.g., ⌘K) and description.

---

#### SuggestionBar.svelte
**File:** `desktop/src/lib/components/SuggestionBar.svelte`

AI reply suggestions with streaming text support.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `chatId` | `string` | Current conversation ID |
| `onSelect` | `(text: string) => void` | Suggestion selection callback |
| `onClose` | `() => void` | Close callback |
| `initialSuggestions` | `DraftSuggestion[]` | Prefetched suggestions |

**Dependencies:** `Icon`

**Store Dependencies:** `jarvis` socket client (streaming)

**Screenshot Description:**
Horizontal bar with sparkles icon, streaming text (when loading), or chip buttons for each suggestion. Close and regenerate buttons on right. Slides up from bottom of compose area.

---

#### Toast.svelte
**File:** `desktop/src/lib/components/Toast.svelte`

Notification toast container.

**Props/Exports:** None

**Dependencies:** `Icon`

**Store Dependencies:** `toasts` (all toasts), `dismissToast`

**Screenshot Description:**
Fixed position stack in bottom-right corner. Each toast has icon (check/x/alert), message, optional description, action button, and dismiss button. Colored by type (success=green, error=red, warning=orange, info=blue).

---

### View Components

#### Dashboard.svelte
**File:** `desktop/src/lib/components/Dashboard.svelte`

System overview with metrics and analytics.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `onNavigate` | `(view: View) => void` | Optional navigation callback |

**Dependencies:** None

**Store Dependencies:** `healthStore`, `conversationsStore`, `metricsStore`

**Screenshot Description:**
4-column grid of stat cards (Conversations, System Health, AI Model, iMessage). Each card has icon, stat value, and subtitle. Below is Routing Metrics section with summary cards, decision distribution chips, and request log table with latency breakdown bars.

---

#### HealthStatus.svelte
**File:** `desktop/src/lib/components/HealthStatus.svelte`

Detailed system health monitoring.

**Props/Exports:** None

**Dependencies:** None

**Store Dependencies:** `healthStore`, `templateAnalyticsStore`

**Screenshot Description:**
Status banner with icon (changes based on health), 4 metric cards (Memory, Process, AI Model, iMessage) with progress bars. Template Analytics section with pie chart coverage visualization, statistics grid, bar chart for top templates, and category similarity bars.

---

#### Settings.svelte
**File:** `desktop/src/lib/components/Settings.svelte`

Application settings with model management.

**Props/Exports:** None

**Dependencies:** None

**Store Dependencies:** `theme` (themeMode, accentColor, setTheme, setAccentColor, setReducedMotion)

**Screenshot Description:**
Vertical sections: Model selection (cards with download/activate buttons), Generation parameters (sliders for temperature, max tokens), Suggestions toggle and settings, Appearance (theme radio buttons, accent color picker), System info (read-only stats).

---

#### TemplateBuilder.svelte
**File:** `desktop/src/lib/components/TemplateBuilder.svelte`

Custom response template management.

**Props/Exports:** None

**Dependencies:** None

**Screenshot Description:**
Two-tab interface (Templates/Stats). Template list with filter dropdowns, cards showing template info with enable/disable/edit/delete actions. Modal editor for creating/editing with form fields and template tester.

---

#### AttachmentGallery.svelte
**File:** `desktop/src/lib/components/AttachmentGallery.svelte`

Media and file browser for conversations.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `chatId` | `string \| null` | Optional conversation filter |

**Dependencies:** None

**Screenshot Description:**
Tabbed interface (Gallery/Storage) with type filters, date range pickers. Grid view with thumbnails for images/videos, list view for files. Storage tab shows total usage card and conversation breakdown with progress bars.

---

### Graph Visualization Components

#### RelationshipGraph.svelte
**File:** `desktop/src/lib/components/graph/RelationshipGraph.svelte`

D3.js force-directed relationship network.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `width` | `number` | Canvas width |
| `height` | `number` | Canvas height |
| `onNodeClick` | `(node) => void` | Node click handler |
| `onNodeDoubleClick` | `(node) => void` | Node double-click handler |

**Dependencies:** `GraphControls`, `NodeTooltip`, `ClusterLegend`

**Screenshot Description:**
Interactive SVG network with draggable nodes, zoom/pan controls, colored by relationship type. Nodes sized by message count, links by connection strength. Hover shows tooltip, click highlights connected nodes.

---

#### GraphControls.svelte
**File:** `desktop/src/lib/components/graph/GraphControls.svelte`

Toolbar for graph interaction.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `showLabels` | `boolean` | Labels visible |
| `currentLayout` | `LayoutType` | Current layout algorithm |

**Events:**
- `resetZoom`, `toggleLabels`, `reheat`, `changeLayout`, `search`, `filterRelationships`, `export`

**Screenshot Description:**
Horizontal toolbar with search input, layout dropdown, relationship filter chips, and icon buttons for zoom reset, labels toggle, reheat simulation, and export.

---

#### NodeTooltip.svelte
**File:** `desktop/src/lib/components/graph/NodeTooltip.svelte`

Hover tooltip for graph nodes.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `node` | `GraphNode` | Node data |
| `x` | `number` | X position |
| `y` | `number` | Y position |

**Screenshot Description:**
Floating card following cursor showing contact name, relationship type, message count, sentiment score with color, avg response time, and last contact date.

---

#### ClusterLegend.svelte
**File:** `desktop/src/lib/components/graph/ClusterLegend.svelte`

Legend and filter for relationship types.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `nodes` | `GraphNode[]` | All nodes |
| `colors` | `Record<string, string>` | Type color mapping |
| `onfilter` | `(types: string[]) => void` | Filter callback |

**Screenshot Description:**
Collapsible panel at bottom-left showing colored dots for each relationship type with counts. Click to filter, dimming non-matching nodes in graph.

---

#### TimeSlider.svelte
**File:** `desktop/src/lib/components/graph/TimeSlider.svelte`

Temporal navigation for graph evolution.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `fromDate` | `Date` | Start date |
| `toDate` | `Date` | End date |
| `currentDate` | `Date` | Selected date |
| `isPlaying` | `boolean` | Animation playing |

**Events:** `dateChange`, `play`, `pause`, `reset`

**Screenshot Description:**
Card with date range labels, progress slider with fill bar, playback controls (step back, play/pause, step forward, reset), and current date display.

---

### Tag Management Components

#### TagSidebar.svelte
**File:** `desktop/src/lib/components/tags/TagSidebar.svelte`

Navigation sidebar for tags and smart folders.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `tags` | `Tag[]` | All tags |
| `smartFolders` | `SmartFolder[]` | All folders |
| `selectedTagId` | `number \| null` | Selected tag |
| `selectedFolderId` | `number \| null` | Selected folder |
| `tagCounts` | `Record<number, number>` | Usage counts |

**Events:** `selectTag`, `selectFolder`, `createTag`, `createFolder`, `editTag`, `editFolder`

**Dependencies:** `TagBadge`

**Screenshot Description:**
264px wide sidebar with search input, Smart Folders section with folder icons, Tags section with color dots and counts, hierarchical display for child tags. "All Tags" option at top.

---

#### TagBadge.svelte
**File:** `desktop/src/lib/components/tags/TagBadge.svelte`

Colored tag pill component.

**Props/Exports:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tag` | `Tag` | - | Tag data |
| `size` | `'sm' \| 'md' \| 'lg'` | `'md'` | Badge size |
| `removable` | `boolean` | `false` | Show remove button |
| `clickable` | `boolean` | `false` | Enable click |
| `showIcon` | `boolean` | `true` | Show tag icon |

**Events:** `click`, `remove`

**Screenshot Description:**
Rounded pill with icon (star, heart, flag, etc.), tag name, and optional X button. Background color from tag, text color auto-calculated for contrast.

---

#### TagPicker.svelte
**File:** `desktop/src/lib/components/tags/TagPicker.svelte`

Multi-select dropdown for tags.

**Props/Exports:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tags` | `Tag[]` | `[]` | Available tags |
| `selectedTagIds` | `number[]` | `[]` | Selected IDs |
| `placeholder` | `string` | `'Search tags...'` | Input placeholder |
| `maxSelections` | `number` | `0` | Max selections (0=unlimited) |
| `allowCreate` | `boolean` | `false` | Allow new tag creation |
| `disabled` | `boolean` | `false` | Disable picker |

**Events:** `change`, `create`

**Dependencies:** `TagBadge`

**Screenshot Description:**
Input field with selected tag badges inline, dropdown showing available tags with color dots, "Create" option when allowCreate enabled and search has no matches.

---

#### BulkTagger.svelte
**File:** `desktop/src/lib/components/tags/BulkTagger.svelte`

Modal for bulk tag operations on conversations.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `isOpen` | `boolean` | Modal visibility |
| `tags` | `Tag[]` | Available tags |
| `selectedConversations` | `Conversation[]` | Target conversations |

**Events:** `apply`, `cancel`

**Dependencies:** `TagBadge`, `TagPicker`

**Screenshot Description:**
Modal with selected conversation preview chips, radio buttons for Add/Remove action, tag picker, and confirmation summary showing operation scope.

---

#### SmartFolderEditor.svelte
**File:** `desktop/src/lib/components/tags/SmartFolderEditor.svelte`

Modal for creating/editing smart folders with rules.

**Props/Exports:**
| Name | Type | Description |
|------|------|-------------|
| `folder` | `SmartFolder \| null` | Folder to edit (null=new) |
| `tags` | `Tag[]` | Available tags |
| `isOpen` | `boolean` | Modal visibility |

**Events:** `save`, `delete`, `cancel`, `preview`

**Screenshot Description:**
Complex modal with name/icon inputs, color picker, match type radio (all/any), dynamic condition builder with field/operator/value dropdowns, sort options, and preview button.

---

### Icon Components

All icon components in `desktop/src/lib/components/icons/`:

| Component | Props | Description |
|-----------|-------|-------------|
| `HealthIcon` | `size`, `class`, `strokeWidth` | Activity/health icon |
| `MessageIcon` | `size`, `class`, `strokeWidth` | Speech bubble icon |
| `SearchIcon` | `size`, `class`, `strokeWidth` | Magnifying glass |
| `SendIcon` | `size`, `class` | Paper airplane |
| `SettingsIcon` | `size`, `class`, `strokeWidth` | Gear/cog icon |

**IconProps interface:**
```typescript
interface IconProps {
  size?: number;
  class?: string;
  strokeWidth?: number;
}
```

---

## Component Tree

```
App
├── Sidebar (persistent)
│   └── healthStore (status dot)
├── ErrorBoundary
│   └── Current View:
│       ├── dashboard: Dashboard
│       │   ├── healthStore
│       │   ├── conversationsStore
│       │   └── metricsStore
│       ├── messages: MessagesContainer
│       │   ├── ConversationList
│       │   │   └── ConversationSkeleton (loading)
│       │   └── MessageView
│       │       ├── EmptyState (no selection)
│       │       ├── Header (avatar, info, AI Draft button)
│       │       ├── MessagesContainer
│       │       │   ├── MessageSkeleton (loading)
│       │       │   ├── LoadEarlierSection
│       │       │   ├── VirtualScrollSpacer
│       │       │   ├── DateHeader[]
│       │       │   └── MessageItem[]
│       │       │       └── reactions, attachments, optimistic UI
│       │       ├── NewMessagesButton (floating)
│       │       ├── SuggestionBar (conditional)
│       │       │   ├── Icon (sparkles)
│       │       │   └── streaming/chips UI
│       │       └── ComposeArea
│       │           ├── Textarea
│       │           └── Button (send)
│       ├── health: HealthStatus
│       ├── settings: Settings
│       ├── templates: TemplateBuilder
│       │   └── Editor Modal
│       └── network: RelationshipGraph
│           ├── GraphControls
│           ├── SVG (D3 visualization)
│           ├── ClusterLegend
│           └── NodeTooltip (conditional)
├── GlobalSearch (modal, conditional)
├── CommandPalette (modal, conditional)
├── KeyboardShortcuts (modal, conditional)
└── Toast (fixed, global)
```

---

## Store Dependencies

### Store to Component Mapping

| Store | Components | Data Flow |
|-------|------------|-----------|
| **conversationsStore** | `App`, `ConversationList`, `MessageView`, `GlobalSearch`, `Dashboard` | Selected conversation, message lists, loading states, optimistic messages |
| **healthStore** | `App`, `Sidebar`, `Dashboard`, `HealthStatus`, `Settings` | Connection status, system health metrics |
| **theme** | `App`, `Settings`, `CommandPalette` | Theme mode, accent color, reduced motion |
| **keyboard** | `ConversationList`, `MessageView` | Focus zones, navigation indices, announcements |
| **metricsStore** | `Dashboard` | Routing metrics, latency data |
| **templateAnalyticsStore** | `HealthStatus` | Template usage statistics |
| **toast** | `Toast`, `ErrorBoundary` | Notification queue |
| **websocket** | (background service) | Real-time message updates |

### Store Interface Summary

```typescript
// conversationsStore
- selectedChatId, selectedConversation
- conversations[], messages[]
- loading, loadingMessages, loadingMore, hasMore
- optimisticMessages[], prefetchedDraft
- Functions: selectConversation, loadMoreMessages, pollMessages, etc.

// healthStore
- data: HealthResponse | null
- loading, error, connected
- Functions: fetchHealth, checkApiConnection

// theme
- themeMode: 'dark' | 'light' | 'system'
- accentColor: string
- reducedMotion: boolean
- Functions: setTheme, setAccentColor, setReducedMotion

// keyboard
- activeZone: 'conversations' | 'messages' | 'compose' | null
- conversationIndex, messageIndex
- Functions: setActiveZone, setConversationIndex, announce

// metricsStore
- summary: MetricsSummary | null
- recentRequests: MetricsRequest[]
- Functions: startMetricsPolling, stopMetricsPolling

// toast
- toasts: Toast[]
- Functions: toast.success/error/warning/info, dismissToast
```

---

## Style Tokens

All design tokens are defined in `desktop/src/lib/styles/tokens.css`.

### Color Tokens

```css
/* Primitive Colors */
--color-blue: #007AFF
--color-green: #34C759
--color-orange: #FF9500
--color-red: #FF3B30
--color-purple: #5856D6
--color-pink: #FF2D55
--color-yellow: #FFCC00
--color-teal: #5AC8FA
--color-indigo: #5E5CE6

/* Semantic Colors */
--color-primary: var(--color-blue)
--color-success: var(--color-green)
--color-warning: var(--color-orange)
--color-error: var(--color-red)
--color-info: var(--color-teal)

/* Surface Colors (Dark Default) */
--surface-base: #0A0A0A
--surface-elevated: #141414
--surface-surface: #1C1C1E
--surface-hover: #2C2C2E
--surface-active: #3A3A3C

/* Text Colors */
--text-primary: #FFFFFF
--text-secondary: rgba(255, 255, 255, 0.6)
--text-tertiary: rgba(255, 255, 255, 0.4)
--text-disabled: rgba(255, 255, 255, 0.3)

/* Border Colors */
--border-subtle: rgba(255, 255, 255, 0.08)
--border-default: rgba(255, 255, 255, 0.12)
--border-focus: rgba(0, 122, 255, 0.5)

/* Message Bubbles */
--bubble-me: var(--color-primary)
--bubble-me-gradient: linear-gradient(135deg, var(--color-primary) 0%, #0056b3 100%)
--bubble-other: var(--surface-surface)

/* Group Accent */
--group-color: var(--color-purple)
```

### Spacing Scale

```css
--space-0: 0
--space-1: 4px
--space-2: 8px
--space-3: 12px
--space-4: 16px
--space-5: 20px
--space-6: 24px
--space-7: 28px
--space-8: 32px
--space-9: 36px
--space-10: 40px
--space-12: 48px
--space-16: 64px
```

### Typography

```css
--font-family-sans: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif
--font-family-mono: 'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace

/* Font Sizes */
--text-xs: 11px
--text-sm: 13px
--text-base: 15px
--text-lg: 17px
--text-xl: 20px
--text-2xl: 24px
--text-3xl: 32px

/* Font Weights */
--font-weight-normal: 400
--font-weight-medium: 500
--font-weight-semibold: 600
--font-weight-bold: 700

/* Line Heights */
--line-height-tight: 1.25
--line-height-normal: 1.5
--line-height-relaxed: 1.75

/* Letter Spacing */
--letter-spacing-tight: -0.02em
--letter-spacing-normal: 0
--letter-spacing-wide: 0.02em
```

### Border Radius

```css
--radius-sm: 6px
--radius-md: 8px
--radius-lg: 12px
--radius-xl: 16px
--radius-2xl: 20px
--radius-full: 9999px
```

### Animation

```css
/* Durations */
--duration-instant: 0ms
--duration-fast: 150ms
--duration-normal: 200ms
--duration-slow: 300ms
--duration-slower: 500ms

/* Easings */
--ease-linear: linear
--ease-out: cubic-bezier(0.33, 1, 0.68, 1)
--ease-in-out: cubic-bezier(0.65, 0, 0.35, 1)
--ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1)
```

### Shadows

```css
--shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.1)
--shadow-md: 0 4px 12px rgba(0, 0, 0, 0.15)
--shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.2)
--shadow-focus: 0 0 0 3px var(--border-focus)
```

### Z-Index Scale

```css
--z-base: 0
--z-dropdown: 100
--z-sticky: 200
--z-modal-backdrop: 300
--z-modal: 400
--z-popover: 500
--z-tooltip: 600
--z-toast: 700
```

### Utility Classes

```css
.focus-ring:focus-visible  /* Focus outline */
.sr-only                   /* Screen reader only */
.glass                     /* Backdrop blur effect */
```

---

## Missing Components

Based on the current app structure, the following components could be extracted for better reusability:

### 1. **Modal/Dialog Component**
**Current State:** Each modal (CommandPalette, KeyboardShortcuts, GlobalSearch, BulkTagger, SmartFolderEditor, TemplateBuilder modals) implements its own backdrop, positioning, and animation.

**Suggested API:**
```svelte
<Modal bind:open title="Modal Title" on:close>
  <ModalBody>Content</ModalBody>
  <ModalFooter>
    <Button>Cancel</Button>
    <Button variant="primary">Save</Button>
  </ModalFooter>
</Modal>
```

### 2. **Input Component**
**Current State:** Raw `<input>` and `<textarea>` elements styled individually across components.

**Suggested API:**
```svelte
<Input label="Name" bind:value error={errorMessage} />
<Textarea label="Description" bind:value rows={4} />
<Select label="Category" options={categories} bind:value />
```

### 3. **Card Component**
**Current State:** Dashboard cards, metric cards, and template cards each have their own styles.

**Suggested API:**
```svelte
<Card title="Title" icon={IconComponent}>
  <CardContent>Value</CardContent>
  <CardFooter>Subtitle</CardFooter>
</Card>
```

### 4. **Badge Component**
**Current State:** Topic tags, decision chips, and status indicators are similar but separate.

**Suggested API:**
```svelte
<Badge variant="primary|success|warning|error|info">Label</Badge>
<Badge dot color="green">Status</Badge>
```

### 5. **Tooltip Component**
**Current State:** Only NodeTooltip exists; other hover tooltips could use a generic version.

**Suggested API:**
```svelte
<Tooltip text="Helpful hint">
  <Button>Hover me</Button>
</Tooltip>
```

### 6. **Dropdown/Menu Component**
**Current State:** CommandPalette and filter dropdowns could share a common menu base.

**Suggested API:**
```svelte
<Dropdown bind:open>
  <DropdownTrigger>Open Menu</DropdownTrigger>
  <DropdownMenu>
    <DropdownItem on:click={action}>Item 1</DropdownItem>
    <DropdownItem disabled>Item 2</DropdownItem>
    <DropdownDivider />
    <DropdownItem>Item 3</DropdownItem>
  </DropdownMenu>
</Dropdown>
```

### 7. **Avatar Component**
**Current State:** Avatar rendering logic is duplicated in ConversationList, MessageView, and ContactDetailPanel.

**Suggested API:**
```svelte
<Avatar src={url} fallback="JD" size="sm|md|lg" />
<AvatarGroup avatars={users} max={3} />
```

### 8. **Tabs Component**
**Current State:** TemplateBuilder and Settings implement tabs inline.

**Suggested API:**
```svelte
<Tabs bind:activeTab>
  <Tab id="list" title="List">Content</Tab>
  <Tab id="stats" title="Stats">Content</Tab>
</Tabs>
```

### 9. **Toggle/Switch Component**
**Current State:** Custom toggle implementation in Settings.

**Suggested API:**
```svelte
<Toggle bind:checked label="Enable feature" />
```

### 10. **Slider/Range Component**
**Current State:** Range inputs styled individually in Settings.

**Suggested API:**
```svelte
<Slider bind:value min={0} max={100} label="Temperature" showValue />
```

---

*Document generated from source analysis of JARVIS desktop components.*
