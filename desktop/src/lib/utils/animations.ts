/**
 * Advanced animation utilities with spring physics and choreography
 */

export interface SpringConfig {
  stiffness: number;
  damping: number;
  mass: number;
}

// Preset spring configurations
export const springs = {
  // Snappy, responsive feel
  snappy: { stiffness: 400, damping: 30, mass: 1 },
  // Gentle, smooth transitions
  gentle: { stiffness: 120, damping: 14, mass: 1 },
  // Bouncy, playful feel
  bouncy: { stiffness: 300, damping: 10, mass: 1 },
  // Slow, deliberate motion
  slow: { stiffness: 100, damping: 20, mass: 1 },
  // Quick micro-interactions
  micro: { stiffness: 500, damping: 35, mass: 0.5 },
} as const;

// Easing functions for CSS
export const easings = {
  // iOS-like spring approximation
  spring: "cubic-bezier(0.175, 0.885, 0.32, 1.275)",
  // Smooth deceleration
  easeOut: "cubic-bezier(0.33, 1, 0.68, 1)",
  // Quick start, smooth end
  easeOutExpo: "cubic-bezier(0.16, 1, 0.3, 1)",
  // Elastic feel
  elastic: "cubic-bezier(0.68, -0.55, 0.265, 1.55)",
  // Smooth both ways
  easeInOut: "cubic-bezier(0.65, 0, 0.35, 1)",
} as const;

// Duration presets
export const durations = {
  instant: 100,
  fast: 150,
  normal: 250,
  slow: 400,
  slower: 600,
} as const;

/**
 * Calculate staggered delay for list animations
 */
export function staggerDelay(index: number, baseDelay = 50, maxDelay = 300): number {
  return Math.min(index * baseDelay, maxDelay);
}

/**
 * Generate CSS keyframes for a spring animation
 */
export function springKeyframes(
  from: number,
  to: number,
  config: SpringConfig = springs.snappy
): string {
  const { stiffness, damping, mass } = config;
  const frames: string[] = [];
  const steps = 60;

  let velocity = 0;
  let position = from;
  const target = to;

  for (let i = 0; i <= steps; i++) {
    const progress = i / steps;
    const springForce = -stiffness * (position - target);
    const dampingForce = -damping * velocity;
    const acceleration = (springForce + dampingForce) / mass;

    velocity += acceleration * 0.016;
    position += velocity * 0.016;

    frames.push(`${Math.round(progress * 100)}% { transform: translateY(${position}px); }`);
  }

  return frames.join("\n");
}

/**
 * CSS custom property based animation system
 */
export const cssAnimations = {
  fadeIn: `
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
  `,
  fadeOut: `
    @keyframes fadeOut {
      from { opacity: 1; }
      to { opacity: 0; }
    }
  `,
  slideUp: `
    @keyframes slideUp {
      from {
        opacity: 0;
        transform: translateY(16px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
  `,
  slideDown: `
    @keyframes slideDown {
      from {
        opacity: 0;
        transform: translateY(-16px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
  `,
  scaleIn: `
    @keyframes scaleIn {
      from {
        opacity: 0;
        transform: scale(0.9);
      }
      to {
        opacity: 1;
        transform: scale(1);
      }
    }
  `,
  scaleOut: `
    @keyframes scaleOut {
      from {
        opacity: 1;
        transform: scale(1);
      }
      to {
        opacity: 0;
        transform: scale(0.9);
      }
    }
  `,
  shake: `
    @keyframes shake {
      0%, 100% { transform: translateX(0); }
      10%, 30%, 50%, 70%, 90% { transform: translateX(-4px); }
      20%, 40%, 60%, 80% { transform: translateX(4px); }
    }
  `,
  pulse: `
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
  `,
  spin: `
    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
  `,
  shimmer: `
    @keyframes shimmer {
      0% { background-position: -200% 0; }
      100% { background-position: 200% 0; }
    }
  `,
  bounce: `
    @keyframes bounce {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-8px); }
    }
  `,
  wiggle: `
    @keyframes wiggle {
      0%, 100% { transform: rotate(0deg); }
      25% { transform: rotate(-3deg); }
      75% { transform: rotate(3deg); }
    }
  `,
};

/**
 * Transition presets for common use cases
 */
export const transitions = {
  default: `all ${durations.normal}ms ${easings.easeOut}`,
  fast: `all ${durations.fast}ms ${easings.easeOut}`,
  slow: `all ${durations.slow}ms ${easings.easeOut}`,
  spring: `all ${durations.normal}ms ${easings.spring}`,
  color: `color ${durations.fast}ms ${easings.easeOut}, background-color ${durations.fast}ms ${easings.easeOut}`,
  transform: `transform ${durations.normal}ms ${easings.spring}`,
  opacity: `opacity ${durations.normal}ms ${easings.easeOut}`,
};
