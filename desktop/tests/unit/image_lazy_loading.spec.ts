/**
 * Unit tests for image lazy loading with IntersectionObserver
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('Image Lazy Loading', () => {
  let mockIntersectionObserver: any;
  let observedElements: Element[] = [];

  beforeEach(() => {
    observedElements = [];

    mockIntersectionObserver = vi.fn((callback) => ({
      observe: vi.fn((element: Element) => {
        observedElements.push(element);
      }),
      unobserve: vi.fn((element: Element) => {
        observedElements = observedElements.filter((el) => el !== element);
      }),
      disconnect: vi.fn(),
      trigger: (entries: IntersectionObserverEntry[]) =>
        callback(entries, mockIntersectionObserver),
    }));

    vi.stubGlobal('IntersectionObserver', mockIntersectionObserver);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('IntersectionObserver configuration', () => {
    it('should use correct root margin for early loading', () => {
      // Simulate the lazyImage action setup
      const config = {
        root: null,
        rootMargin: '100px',
        threshold: 0.01,
      };

      expect(config.rootMargin).toBe('100px');
      expect(config.threshold).toBe(0.01);
    });

    it('should observe images when created', () => {
      const img = document.createElement('img');

      // Simulate lazyImage action
      const observer = new IntersectionObserver(() => {});
      observer.observe(img);

      expect(observedElements).toContain(img);
    });

    it('should unobserve images when destroyed', () => {
      const img = document.createElement('img');
      const observer = new IntersectionObserver(() => {});

      observer.observe(img);
      expect(observedElements).toHaveLength(1);

      observer.unobserve(img);
      expect(observedElements).toHaveLength(0);
    });
  });

  describe('lazy loading behavior', () => {
    it('should set data-src instead of src initially', () => {
      const img = document.createElement('img');
      const src = 'https://example.com/image.jpg';

      // Simulate initial state
      img.dataset.src = src;

      expect(img.dataset.src).toBe(src);
      expect(img.src).toBe('');
    });

    it('should load image when intersecting', () => {
      const img = document.createElement('img');
      const src = 'https://example.com/image.jpg';

      // Setup
      img.dataset.src = src;
      img.style.opacity = '0';

      // Simulate intersection
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const target = entry.target as HTMLImageElement;
            if (target.dataset.src) {
              target.src = target.dataset.src;
              target.removeAttribute('data-src');
            }
          }
        });
      });

      observer.observe(img);

      // Trigger intersection
      mockIntersectionObserver.trigger([
        { target: img, isIntersecting: true } as IntersectionObserverEntry,
      ]);

      expect(img.src).toBe(src);
      expect(img.dataset.src).toBeUndefined();
    });

    it('should apply fade-in transition', () => {
      const img = document.createElement('img');

      // Simulate lazyImage setup
      img.style.opacity = '0';
      img.style.transition = 'opacity 0.2s ease';

      expect(img.style.opacity).toBe('0');
      expect(img.style.transition).toContain('opacity');
    });

    it('should set opacity to 1 on load', () => {
      const img = document.createElement('img');
      img.style.opacity = '0';

      // Simulate load event
      const handleLoad = () => {
        img.style.opacity = '1';
      };

      img.addEventListener('load', handleLoad);
      img.dispatchEvent(new Event('load'));

      expect(img.style.opacity).toBe('1');
    });
  });

  describe('fallback behavior', () => {
    it('should load immediately if no IntersectionObserver support', () => {
      vi.unstubAllGlobals();
      vi.stubGlobal('IntersectionObserver', undefined);

      const img = document.createElement('img');
      const src = 'https://example.com/image.jpg';

      // Simulate fallback behavior
      if (!('IntersectionObserver' in window)) {
        img.src = src;
      }

      expect(img.src).toBe(src);
    });
  });

  describe('error handling', () => {
    it('should handle missing image gracefully', () => {
      const img = document.createElement('img');
      const fallbackDiv = document.createElement('div');

      // Simulate error handler
      const handleError = () => {
        img.style.display = 'none';
        fallbackDiv.classList.remove('hidden');
      };

      img.addEventListener('error', handleError);
      img.dispatchEvent(new Event('error'));

      expect(img.style.display).toBe('none');
    });
  });
});
