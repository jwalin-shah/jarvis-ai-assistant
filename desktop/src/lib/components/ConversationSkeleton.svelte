<script lang="ts">
  import Skeleton from './Skeleton.svelte';

  const items = 8;
</script>

<div class="conversation-skeleton-container">
  {#each Array(items) as _, i}
    <div class="conversation-skeleton-item">
      <div class="avatar-skeleton">
        <Skeleton width="44px" height="44px" borderRadius="50%" />
      </div>
      <div class="info-skeleton">
        <div class="name-row-skeleton">
          <Skeleton width={i % 2 === 0 ? '60%' : '45%'} height="14px" borderRadius="4px" />
          <Skeleton width="40px" height="12px" borderRadius="4px" />
        </div>
        <Skeleton width={i % 3 === 0 ? '90%' : '75%'} height="12px" borderRadius="4px" />
      </div>
    </div>
  {/each}
</div>

<style>
  .conversation-skeleton-container {
    display: flex;
    flex-direction: column;
  }

  .conversation-skeleton-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
  }

  .avatar-skeleton {
    flex-shrink: 0;
  }

  .info-skeleton {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .name-row-skeleton {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
</style>
