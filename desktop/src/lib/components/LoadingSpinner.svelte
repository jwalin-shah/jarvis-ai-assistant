<script lang="ts">
  export let size: "small" | "medium" | "large" = "medium";
  export let color: string = "var(--accent-color)";

  const sizes = {
    small: { width: "16px", height: "16px", border: "2px" },
    medium: { width: "32px", height: "32px", border: "3px" },
    large: { width: "48px", height: "48px", border: "4px" },
  };

  $: style = `
    width: ${sizes[size].width};
    height: ${sizes[size].height};
    border-width: ${sizes[size].border};
    border-top-color: ${color};
  `;
</script>

<div class="spinner" {style}></div>

<style>
  .spinner {
    border-style: solid;
    border-color: var(--bg-secondary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
