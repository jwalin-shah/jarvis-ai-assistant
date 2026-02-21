import { writable } from "svelte/store";

const MAX_LATENCY_SAMPLES = 200;

export interface ReliabilityState {
  send_attempts: number;
  send_successes: number;
  send_failures: number;
  draft_attempts: number;
  draft_successes: number;
  draft_failures: number;
  draft_latencies_ms: number[];
  last_failure_reason: string | null;
  last_failure_at: number | null;
}

const initialState: ReliabilityState = {
  send_attempts: 0,
  send_successes: 0,
  send_failures: 0,
  draft_attempts: 0,
  draft_successes: 0,
  draft_failures: 0,
  draft_latencies_ms: [],
  last_failure_reason: null,
  last_failure_at: null,
};

export const reliabilityStore = writable<ReliabilityState>(initialState);

function setFailure(reason: string) {
  reliabilityStore.update((s) => ({
    ...s,
    last_failure_reason: reason,
    last_failure_at: Date.now(),
  }));
}

export function recordSendAttempt(): void {
  reliabilityStore.update((s) => ({ ...s, send_attempts: s.send_attempts + 1 }));
}

export function recordSendSuccess(): void {
  reliabilityStore.update((s) => ({ ...s, send_successes: s.send_successes + 1 }));
}

export function recordSendFailure(reason: string): void {
  reliabilityStore.update((s) => ({ ...s, send_failures: s.send_failures + 1 }));
  setFailure(reason);
}

export function recordDraftAttempt(): void {
  reliabilityStore.update((s) => ({ ...s, draft_attempts: s.draft_attempts + 1 }));
}

export function recordDraftSuccess(latencyMs: number): void {
  reliabilityStore.update((s) => ({
    ...s,
    draft_successes: s.draft_successes + 1,
    draft_latencies_ms: [...s.draft_latencies_ms, Math.max(0, latencyMs)].slice(
      -MAX_LATENCY_SAMPLES
    ),
  }));
}

export function recordDraftFailure(reason: string): void {
  reliabilityStore.update((s) => ({ ...s, draft_failures: s.draft_failures + 1 }));
  setFailure(reason);
}
