/**
 * Simple logger wrapper to control log levels in production.
 * Automatically adds a prefix to log messages.
 * Only logs debug/log/info to console if in development mode.
 */

export class Logger {
  private prefix: string;

  constructor(prefix: string) {
    this.prefix = prefix;
  }

  private getArgs(args: unknown[]): unknown[] {
    if (args.length > 0 && typeof args[0] === 'string') {
      const message = args[0];
      return [`[${this.prefix}] ${message}`, ...args.slice(1)];
    }
    return [`[${this.prefix}]`, ...args];
  }

  /**
   * Log with console.debug (only in DEV)
   */
  debug(...args: unknown[]) {
    if (import.meta.env.DEV) {
      console.debug(...this.getArgs(args));
    }
  }

  /**
   * Log with console.log (only in DEV)
   */
  log(...args: unknown[]) {
    if (import.meta.env.DEV) {
      console.log(...this.getArgs(args));
    }
  }

  /**
   * Log with console.info (only in DEV)
   */
  info(...args: unknown[]) {
    if (import.meta.env.DEV) {
      console.info(...this.getArgs(args));
    }
  }

  /**
   * Log with console.warn (always logs)
   */
  warn(...args: unknown[]) {
    console.warn(...this.getArgs(args));
  }

  /**
   * Log with console.error (always logs)
   */
  error(...args: unknown[]) {
    console.error(...this.getArgs(args));
  }
}
