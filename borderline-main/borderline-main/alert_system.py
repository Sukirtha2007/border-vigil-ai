"""
Alert System — manages alert queue with deduplication and severity levels.

Key design decisions:
  - Alerts are DEDUPLICATED by (event_type, track_id) key — same alert for same
    person doesn't stack. Instead it refreshes the timestamp.
  - MAX 4 visible alerts at a time, prioritized by severity then recency.
  - Shorter display duration (3s) to prevent screen clutter.
  - Audio only on CRITICAL, with a cooldown to prevent beep spam.
"""

import time
import sys
import threading


# Severity levels (BGR colors)
SEVERITY_COLORS = {
    "INFO": (200, 200, 0),       # cyan-ish
    "WARNING": (0, 180, 255),    # orange
    "CRITICAL": (0, 0, 255),     # red
}

SEVERITY_PRIORITY = {
    "INFO": 0,
    "WARNING": 1,
    "CRITICAL": 2,
}

MAX_VISIBLE_ALERTS = 4


class AlertSystem:

    def __init__(self, enable_audio: bool = True, alert_display_duration: float = 3.0):
        self.enable_audio = enable_audio
        self.alert_display_duration = alert_display_duration

        # Deduplicated alerts: key -> (message, severity, timestamp)
        # Key = "event_type:track_id" (e.g., "LOITERING:5")
        self._alerts: dict[str, tuple[str, str, float]] = {}
        self._lock = threading.Lock()

        # Flash state
        self._flash_until: float = 0.0

        # Audio cooldown — don't beep more than once per 3 seconds
        self._last_audio_time: float = 0.0
        self._audio_cooldown: float = 3.0

    def trigger(self, message: str, severity: str = "WARNING",
                event_type: str = "", track_id: int = -1):
        """
        Add or refresh an alert. Deduplicates by (event_type, track_id).
        If the same key already exists, it refreshes the timestamp (no stacking).
        """
        now = time.time()
        key = f"{event_type}:{track_id}" if event_type and track_id >= 0 else f"msg:{hash(message)}"

        with self._lock:
            self._alerts[key] = (message, severity, now)

            if severity == "CRITICAL":
                self._flash_until = now + 0.3

        # Audio — with cooldown
        if self.enable_audio and severity == "CRITICAL":
            if (now - self._last_audio_time) > self._audio_cooldown:
                self._last_audio_time = now
                self._play_alert_sound()

    def _play_alert_sound(self):
        def _beep():
            try:
                if sys.platform == "win32":
                    import winsound
                    winsound.Beep(1000, 200)
                    winsound.Beep(1500, 150)
            except Exception:
                pass
        t = threading.Thread(target=_beep, daemon=True)
        t.start()

    def get_active_alerts(self) -> list[tuple[str, str]]:
        """
        Get currently active alerts (not expired), sorted by severity (highest first),
        limited to MAX_VISIBLE_ALERTS.
        """
        now = time.time()
        with self._lock:
            # Prune expired
            self._alerts = {
                k: (msg, sev, ts) for k, (msg, sev, ts) in self._alerts.items()
                if (now - ts) < self.alert_display_duration
            }

            # Sort: CRITICAL first, then WARNING, then INFO. Within same severity, newest first.
            sorted_alerts = sorted(
                self._alerts.values(),
                key=lambda x: (SEVERITY_PRIORITY.get(x[1], 0), x[2]),
                reverse=True
            )

            return [(msg, sev) for msg, sev, _ in sorted_alerts[:MAX_VISIBLE_ALERTS]]

    def is_flashing(self) -> bool:
        return time.time() < self._flash_until

    def get_highest_severity(self) -> str | None:
        alerts = self.get_active_alerts()
        if not alerts:
            return None

        max_sev = "INFO"
        for _, sev in alerts:
            if SEVERITY_PRIORITY.get(sev, 0) > SEVERITY_PRIORITY.get(max_sev, 0):
                max_sev = sev
        return max_sev
