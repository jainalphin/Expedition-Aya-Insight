"""
Progress monitoring utilities for tracking parallel processing.
"""
import time
import threading
import logging
from typing import Dict, List, Any, Set

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Tracks progress of parallel document processing tasks.
    """

    def __init__(self, total_documents, update_interval=5):
        """
        Initialize the progress tracker.

        Args:
            total_documents: Total number of documents to process
            update_interval: How often to log updates (in seconds)
        """
        self.total = total_documents
        self.completed = 0
        self.failed = 0
        self.in_progress = 0
        self.processed_files = set()
        self.update_interval = update_interval
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()

    def mark_started(self, filename):
        """Mark a document as being processed."""
        with self.lock:
            self.in_progress += 1
            logger.info(f"Started processing: {filename}")

    def mark_completed(self, filename, success=True):
        """Mark a document as completed."""
        with self.lock:
            self.in_progress -= 1
            if filename not in self.processed_files:
                self.processed_files.add(filename)
                if success:
                    self.completed += 1
                else:
                    self.failed += 1

    def get_stats(self):
        """Get current processing statistics."""
        with self.lock:
            elapsed = time.time() - self.start_time
            remaining = self.total - (self.completed + self.failed)

            # Calculate estimated time remaining
            if self.completed > 0:
                avg_time_per_doc = elapsed / self.completed
                est_remaining = avg_time_per_doc * remaining
            else:
                est_remaining = None

            return {
                'total': self.total,
                'completed': self.completed,
                'failed': self.failed,
                'in_progress': self.in_progress,
                'remaining': remaining,
                'elapsed_seconds': elapsed,
                'estimated_remaining_seconds': est_remaining
            }

    def _format_time(self, seconds):
        """Format seconds as HH:MM:SS."""
        if seconds is None:
            return "unknown"

        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _monitor_progress(self):
        """Monitor and log progress periodically."""
        while not self.stop_monitoring.is_set():
            stats = self.get_stats()

            logger.info(
                f"Progress: {stats['completed']}/{stats['total']} completed, "
                f"{stats['failed']} failed, {stats['in_progress']} in progress | "
                f"Elapsed: {self._format_time(stats['elapsed_seconds'])} | "
                f"Est. remaining: {self._format_time(stats['estimated_remaining_seconds'])}"
            )

            # Check if we're done
            if stats['completed'] + stats['failed'] >= stats['total']:
                logger.info("All documents processed!")
                break

            # Wait for next update
            self.stop_monitoring.wait(self.update_interval)

    def start_monitoring(self):
        """Start background monitoring thread."""
        self.monitor_thread = threading.Thread(target=self._monitor_progress)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self):
        """Stop the monitoring thread and report final results."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_monitoring.set()
            self.monitor_thread.join(timeout=2.0)

        # Log final statistics
        stats = self.get_stats()
        logger.info(
            f"Final results: {stats['completed']}/{stats['total']} completed, "
            f"{stats['failed']} failed | "
            f"Total time: {self._format_time(stats['elapsed_seconds'])}"
        )

        success_rate = (stats['completed'] / stats['total']) * 100 if stats['total'] > 0 else 0
        logger.info(f"Success rate: {success_rate:.2f}%")

        return stats