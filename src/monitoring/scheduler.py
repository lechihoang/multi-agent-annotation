

import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional
from .metrics import MetricsCollector


class WeightUpdateScheduler:

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        update_interval_hours: int = 24,
        update_hour: int = 0,
    ):
        """Initialize scheduler.

        Args:
            metrics_collector: MetricsCollector instance
            update_interval_hours: Hours between updates (default: 24)
            update_hour: Hour of day for update (default: 0 = midnight)
        """
        self.metrics = metrics_collector
        self.update_interval = update_interval_hours * 3600
        self.update_hour = update_hour

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_update: Optional[datetime] = None
        self._on_update_callbacks: list[Callable[[dict], None]] = []

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(
            f"✓ WeightUpdateScheduler started (interval: {self.update_interval // 3600}h)"
        )

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        print("✓ WeightUpdateScheduler stopped")

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._wait_for_next_update()

                if self._stop_event.is_set():
                    break

                self._perform_update()

            except Exception as e:
                print(f"Warning: Weight update failed: {e}")

    def _wait_for_next_update(self):
        now = datetime.now()
        next_update = datetime(
            now.year, now.month, now.day, self.update_hour
        ) + timedelta(days=1)

        if self._last_update is not None:
            next_update = self._last_update + timedelta(seconds=self.update_interval)
            if next_update <= now:
                next_update = now + timedelta(seconds=60)

        wait_seconds = (next_update - now).total_seconds()
        if wait_seconds > 0:
            self._stop_event.wait(timeout=wait_seconds)

    def _perform_update(self):
        print(f"\n{'=' * 50}")
        print(f"Weight Update - {datetime.now().isoformat()}")
        print("=" * 50)

        stats = self.metrics.get_overall_stats()
        print(f"System Accuracy: {stats['system_accuracy']:.2%}")
        print(f"Total Annotations: {stats['total_annotations']}")

        new_weights = self.metrics.update_weights()

        print(f"\nNew Agent Weights:")
        for agent, weight in sorted(new_weights.items()):
            agent_perf = stats["agent_performance"].get(agent, {})
            accuracy = agent_perf.get("accuracy", "N/A")
            print(f"  {agent}: {weight:.4f} (accuracy: {accuracy})")

        self._last_update = datetime.now()

        for callback in self._on_update_callbacks:
            try:
                callback(new_weights)
            except Exception as e:
                print(f"Warning: Update callback failed: {e}")

    def trigger_update(self) -> dict:
        return self._perform_update() or {}

    def on_update(self, callback: Callable[[dict], None]):
        self._on_update_callbacks.append(callback)

    def get_status(self) -> dict:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "update_interval_hours": self.update_interval // 3600,
            "callbacks_registered": len(self._on_update_callbacks),
        }


def create_weight_updater(
    storage_path: str = "data/metrics.json",
) -> tuple[MetricsCollector, WeightUpdateScheduler]:
    metrics = MetricsCollector(storage_path=storage_path)
    scheduler = WeightUpdateScheduler(metrics)
    return metrics, scheduler
