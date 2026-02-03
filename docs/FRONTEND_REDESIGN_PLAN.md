# JARVIS Frontend Redesign & Performance Optimization Plan

**Created:** 2026-02-02
**Status:** Planned

## Design Direction

**Style:** Modern Minimal
- Ultra-clean with generous whitespace
- Subtle animations and micro-interactions
- Glassmorphism effects (frosted glass, blur)
- Light/Dark mode support

**Priority:** Performance first, then visual polish

---

## Phase 1: Performance - Skeleton Loading & Optimistic Sending

### 1.1 Skeleton Components

**New files:**

`desktop/src/lib/components/Skeleton.svelte`
```svelte
<!-- Base shimmer loading component -->
- Configurable width, height, borderRadius
- Shimmer animation (gradient slide)
```

`desktop/src/lib/components/MessageSkeleton.svelte`
```svelte
<!-- Message bubble placeholders -->
- Variable bubble widths (randomized for natural look)
- Optional avatar skeleton for group chats
- 5-6 skeleton messages per load
```

`desktop/src/lib/components/ConversationSkeleton.svelte`
```svelte
<!-- Conversation list placeholders -->
- Avatar circle + 2 text lines
- 8 skeleton items
```

**Integration points:**
- `ConversationList.svelte` - Replace loading spinner with skeleton
- `MessageView.svelte` - Replace "Loading..." with skeleton

### 1.2 Optimistic Message Sending

**File:** `desktop/src/lib/stores/conversations.ts`

Add optimistic message handling:
```typescript
interface OptimisticMessage {
  id: string;           // 'optimistic-{timestamp}'
  text: string;
  status: 'sending' | 'sent' | 'failed';
  error?: string;
}

// Functions
addOptimisticMessage(text) → id
updateOptimisticMessage(id, status, error?)
removeOptimisticMessage(id)
messagesWithOptimistic  // derived store
```

**File:** `desktop/src/lib/components/MessageView.svelte`

Update send flow:
1. Clear input immediately (instant feedback)
2. Add optimistic message to UI
3. Scroll to bottom
4. Send to server async
5. Update status on response
6. Show retry on failure

### 1.3 Animation Speedup

Reduce all animation durations:
- `0.3s` → `0.15s`
- `2s` highlight → `1s`
- Remove `bounceIn`, use simple `fadeIn`

---

## Phase 2: Design System Foundation

### 2.1 Design Tokens

**File:** `desktop/src/App.svelte`

```css
:root {
  /* === Theme-aware colors === */
  --color-primary: #007AFF;
  --color-primary-hover: #0066D6;
  --color-success: #30D158;
  --color-warning: #FF9F0A;
  --color-error: #FF453A;

  /* === Spacing (4px base) === */
  --space-1: 4px;  --space-2: 8px;  --space-3: 12px;
  --space-4: 16px; --space-5: 20px; --space-6: 24px;
  --space-8: 32px; --space-10: 40px;

  /* === Typography === */
  --text-xs: 11px; --text-sm: 13px; --text-base: 15px;
  --text-lg: 17px; --text-xl: 20px; --text-2xl: 24px;

  /* === Radius === */
  --radius-sm: 6px; --radius-md: 10px; --radius-lg: 14px;
  --radius-xl: 20px; --radius-full: 9999px;

  /* === Animation === */
  --duration-fast: 150ms;
  --duration-normal: 200ms;
  --duration-slow: 300ms;
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);

  /* === Shadows === */
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.15);
  --shadow-lg: 0 12px 40px rgba(0,0,0,0.2);
}

/* === DARK THEME (default) === */
:root {
  --bg-base: #0A0A0A;
  --bg-elevated: #141414;
  --bg-surface: #1C1C1E;
  --bg-overlay: rgba(255,255,255,0.05);
  --bg-glass: rgba(28,28,30,0.8);

  --text-primary: #FFFFFF;
  --text-secondary: rgba(255,255,255,0.6);
  --text-tertiary: rgba(255,255,255,0.4);

  --border-subtle: rgba(255,255,255,0.08);
  --border-default: rgba(255,255,255,0.12);

  --bubble-me: var(--color-primary);
  --bubble-other: var(--bg-surface);
}

/* === LIGHT THEME === */
:root.light {
  --bg-base: #FFFFFF;
  --bg-elevated: #F5F5F7;
  --bg-surface: #FFFFFF;
  --bg-overlay: rgba(0,0,0,0.03);
  --bg-glass: rgba(255,255,255,0.8);

  --text-primary: #1D1D1F;
  --text-secondary: rgba(0,0,0,0.55);
  --text-tertiary: rgba(0,0,0,0.35);

  --border-subtle: rgba(0,0,0,0.06);
  --border-default: rgba(0,0,0,0.1);

  --bubble-me: var(--color-primary);
  --bubble-other: var(--bg-elevated);
}
```

### 2.2 Theme Toggle

**New file:** `desktop/src/lib/stores/theme.ts`
```typescript
import { writable } from 'svelte/store';

type Theme = 'dark' | 'light' | 'system';

function createThemeStore() {
  const stored = localStorage.getItem('theme') as Theme | null;
  const { subscribe, set } = writable<Theme>(stored || 'dark');

  return {
    subscribe,
    set: (value: Theme) => {
      localStorage.setItem('theme', value);
      applyTheme(value);
      set(value);
    },
    toggle: () => {
      // Toggle between dark and light
    }
  };
}

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === 'system') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.classList.toggle('light', !prefersDark);
  } else {
    root.classList.toggle('light', theme === 'light');
  }
}

export const theme = createThemeStore();
```

**Add to Settings.svelte:**
- Theme toggle switch (Dark / Light / System)

---

## Phase 3: Icon System

**New file:** `desktop/src/lib/components/Icon.svelte`

Reusable SVG icon component with Lucide-style icons:

```svelte
<script lang="ts">
  export let name: IconName;
  export let size: number = 20;
  export let strokeWidth: number = 1.5;

  // Icon paths defined inline for bundle efficiency
</script>

<svg
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width={strokeWidth}
  stroke-linecap="round"
  stroke-linejoin="round"
>
  {@html icons[name]}
</svg>
```

**Icons needed:**
- `sparkles`, `x-circle`, `check`, `alert-circle`
- `send`, `refresh-cw`, `copy`, `clipboard-check`
- `message-circle`, `settings`, `bar-chart-2`
- `search`, `chevron-down`, `chevron-up`
- `user`, `users`, `sun`, `moon`

**Updates:**
- Replace ✨ in `AIDraftPanel.svelte` with `<Icon name="sparkles" />`
- Replace ❌ with `<Icon name="x-circle" />`
- Replace ! with `<Icon name="alert-circle" />`

---

## Phase 4: Visual Polish - Modern Minimal

### 4.1 Glassmorphism Effects

Add frosted glass to elevated surfaces:
```css
.glass {
  background: var(--bg-glass);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-subtle);
}
```

Apply to:
- Sidebar
- Modals/panels
- Header bars
- Smart reply chips container

### 4.2 Generous Whitespace

Increase padding throughout:
- Section padding: `--space-6` (24px)
- Card padding: `--space-5` (20px)
- Component gaps: `--space-4` (16px)
- Message list gaps: `--space-3` (12px)

### 4.3 Refined Typography

```css
body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', sans-serif;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

h1, h2, h3 {
  font-weight: 600;
  letter-spacing: -0.02em;
}

.label {
  font-size: var(--text-xs);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-tertiary);
}
```

### 4.4 Micro-interactions

```css
.button {
  transition: all var(--duration-fast) var(--ease-out);
}

.button:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.button:active {
  transform: translateY(0) scale(0.98);
}

.card:hover {
  transform: scale(1.01);
  box-shadow: var(--shadow-lg);
}
```

### 4.5 Focus States

```css
*:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.3);
  border-radius: var(--radius-md);
}

/* High contrast for accessibility */
@media (prefers-contrast: high) {
  *:focus-visible {
    box-shadow: 0 0 0 3px var(--color-primary);
  }
}
```

### 4.6 Message Bubbles Refinement

```css
.message-bubble {
  border-radius: var(--radius-xl);
  padding: var(--space-3) var(--space-4);
  max-width: 75%;
  box-shadow: var(--shadow-sm);
}

.message-bubble.from-me {
  background: linear-gradient(135deg, #007AFF 0%, #0066D6 100%);
  border-bottom-right-radius: var(--radius-sm);
}

.message-bubble.from-other {
  background: var(--bubble-other);
  border-bottom-left-radius: var(--radius-sm);
}
```

---

## Files Summary

### New Files
| File | Purpose |
|------|---------|
| `components/Skeleton.svelte` | Base shimmer component |
| `components/MessageSkeleton.svelte` | Message placeholders |
| `components/ConversationSkeleton.svelte` | List placeholders |
| `components/Icon.svelte` | SVG icon system |
| `stores/theme.ts` | Light/dark mode state |

### Modified Files
| File | Changes |
|------|---------|
| `App.svelte` | Design tokens, theme classes, global styles |
| `stores/conversations.ts` | Optimistic message handling |
| `MessageView.svelte` | Skeleton, optimistic send, animations |
| `ConversationList.svelte` | Skeleton loading, glassmorphism |
| `AIDraftPanel.svelte` | Icons, glassmorphism |
| `SmartReplyChipsV2.svelte` | Icons, animations |
| `Sidebar.svelte` | Glassmorphism, polish |
| `Settings.svelte` | Theme toggle UI |

---

## Implementation Order

1. **Phase 1** (Performance) - Skeleton + Optimistic + Animations
2. **Phase 2** (Foundation) - Design tokens + Theme system
3. **Phase 3** (Icons) - Icon component + replacements
4. **Phase 4** (Polish) - Glass effects, whitespace, micro-interactions

---

## Verification Checklist

- [ ] Run app: `cd desktop && pnpm dev`
- [ ] Toggle theme - Both modes render correctly
- [ ] Slow network test - Skeletons appear during loading
- [ ] Send message - Appears instantly (optimistic)
- [ ] Fail send - Retry button works
- [ ] Tab navigation - Focus rings visible
- [ ] Check accessibility - Sufficient contrast in both themes
- [ ] Visual consistency - Spacing, colors, radius all uniform
- [ ] Glassmorphism - Blur effects render on macOS
- [ ] Animations - Feel snappy, no jank

---

## Design Inspiration

- Apple Messages (macOS Sequoia)
- Linear App
- Raycast
- Arc Browser
- Notion

Key principles from these apps:
- Restrained color palette
- Generous padding
- Subtle depth through shadows/glass
- Smooth, purposeful animations
- Clear visual hierarchy
