<script lang="ts">
  interface Props {
    url: string;
  }

  let { url }: Props = $props();

  function getDomain(urlStr: string): string {
    try {
      return new URL(urlStr).hostname.replace('www.', '');
    } catch {
      return urlStr;
    }
  }

  function getFaviconUrl(urlStr: string): string {
    try {
      const domain = new URL(urlStr).origin;
      return `${domain}/favicon.ico`;
    } catch {
      return '';
    }
  }

  function handleClick() {
    window.open(url, '_blank', 'noopener,noreferrer');
  }

  let domain = $derived(getDomain(url));
  let faviconUrl = $derived(getFaviconUrl(url));
  let faviconError = $state(false);
</script>

<button class="link-preview" onclick={handleClick} title={url}>
  <div class="link-favicon">
    {#if faviconUrl && !faviconError}
      <img src={faviconUrl} alt="" width="14" height="14" onerror={() => (faviconError = true)} />
    {:else}
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        width="14"
        height="14"
      >
        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
        <polyline points="15 3 21 3 21 9"></polyline>
        <line x1="10" y1="14" x2="21" y2="3"></line>
      </svg>
    {/if}
  </div>
  <span class="link-domain">{domain}</span>
  <svg
    class="link-external"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    width="12"
    height="12"
  >
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
    <polyline points="15 3 21 3 21 9"></polyline>
    <line x1="10" y1="14" x2="21" y2="3"></line>
  </svg>
</button>

<style>
  .link-preview {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: var(--surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all var(--duration-fast) var(--ease-out);
    max-width: 100%;
    overflow: hidden;
  }

  .link-preview:hover {
    background: var(--surface-hover);
    border-color: var(--color-primary);
  }

  .link-favicon {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: var(--text-tertiary);
  }

  .link-favicon img {
    border-radius: 2px;
  }

  .link-domain {
    font-size: var(--text-xs);
    color: var(--color-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .link-external {
    flex-shrink: 0;
    color: var(--text-tertiary);
    opacity: 0;
    transition: opacity var(--duration-fast) var(--ease-out);
  }

  .link-preview:hover .link-external {
    opacity: 1;
  }
</style>
