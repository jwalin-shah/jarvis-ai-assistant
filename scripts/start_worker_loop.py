import logging
import signal
import sys
import time

from jarvis.tasks.worker import start_worker, stop_worker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("jarvis.worker")

def main():
    logger.info("Starting JARVIS background task worker...")

    # Start the worker thread
    start_worker()

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Interrupt received, stopping worker...")
        stop_worker()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_worker()

if __name__ == "__main__":
    main()
