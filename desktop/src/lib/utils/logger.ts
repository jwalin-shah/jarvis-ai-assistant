/**
 * Simple logger utility for consistent logging with scopes and levels.
 */

export class Logger {
  private scope: string;

  constructor(scope: string) {
    this.scope = scope;
  }

  private format(message: string): string {
    return `[${this.scope}] ${message}`;
  }

  debug(message: string, ...args: unknown[]): void {
    if (import.meta.env.DEV) {
      console.log(this.format(message), ...args);
    }
  }

  info(message: string, ...args: unknown[]): void {
    console.info(this.format(message), ...args);
  }

  warn(message: string, ...args: unknown[]): void {
    console.warn(this.format(message), ...args);
  }

  error(message: string, ...args: unknown[]): void {
    console.error(this.format(message), ...args);
  }
}
