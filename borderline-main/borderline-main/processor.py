"""
Frame Processor — the central processing pipeline.

For each frame:
  1. Night-mode preprocessing (if dark)
  2. YOLOv8 detection + ByteTrack tracking
  3. Boundary crossing check (line or zone)
  4. Suspicious activity analysis
  5. Event logging + snapshot saving
  6. Alert triggering (deduplicated)
  7. Drawing all overlays
"""

import time
import numpy as np
from detector import Detector, Detection
from crossing_tracker import CrossingTracker
from activity_analyzer import ActivityAnalyzer
from night_processor import NightProcessor
from event_logger import EventLogger
from alert_system import AlertSystem, SEVERITY_PRIORITY
from geometry import Line
import display


class FrameProcessor:

    def __init__(self,
                 detector: Detector,
                 crossing_tracker: CrossingTracker,
                 activity_analyzer: ActivityAnalyzer,
                 night_processor: NightProcessor,
                 event_logger: EventLogger,
                 alert_system: AlertSystem):
        self.detector = detector
        self.crossing_tracker = crossing_tracker
        self.activity_analyzer = activity_analyzer
        self.night_processor = night_processor
        self.event_logger = event_logger
        self.alert_system = alert_system

        # FPS tracking
        self._frame_times: list[float] = []
        self._fps: float = 0.0

        # Per-track severity cache
        self._track_severities: dict[int, tuple[str, float]] = {}  # track_id -> (severity, timestamp)
        self._severity_decay = 5.0  # seconds before severity resets

        # Per-track velocity
        self._prev_positions: dict[int, tuple[float, float, float]] = {}
        self._velocities: dict[int, tuple[float, float]] = {}

    def _update_fps(self):
        now = time.time()
        self._frame_times.append(now)
        self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._fps = (len(self._frame_times) - 1) / elapsed

    def _compute_velocities(self, detections: list[Detection]):
        now = time.time()
        new_positions = {}

        for det in detections:
            if det.track_id < 0:
                continue
            bx, by = det.bottom_center
            prev = self._prev_positions.get(det.track_id)
            if prev is not None:
                px, py, pt = prev
                dt = now - pt
                if dt > 0.01:
                    self._velocities[det.track_id] = ((bx - px) / dt, (by - py) / dt)
            new_positions[det.track_id] = (bx, by, now)

        self._prev_positions = new_positions

    def _get_track_severity(self, track_id: int) -> str:
        """Get severity for a track, decaying to INFO after _severity_decay seconds."""
        entry = self._track_severities.get(track_id)
        if entry is None:
            return "INFO"
        sev, ts = entry
        if (time.time() - ts) > self._severity_decay:
            self._track_severities.pop(track_id, None)
            return "INFO"
        return sev

    def _set_track_severity(self, track_id: int, severity: str):
        """Set severity, keeping the highest if already set."""
        current = self._track_severities.get(track_id)
        now = time.time()
        if current is None or SEVERITY_PRIORITY.get(severity, 0) >= SEVERITY_PRIORITY.get(current[0], 0):
            self._track_severities[track_id] = (severity, now)

    def process_frame(self, frame: np.ndarray,
                      line: Line | None = None,
                      zone_points: np.ndarray | None = None) -> np.ndarray:
        self._update_fps()

        # ── Step 1: Night-mode preprocessing ──
        processed_frame = self.night_processor.process(frame)
        display_frame = frame.copy()

        # ── Step 2: Detection + Tracking ──
        detections = self.detector.detect_and_track(processed_frame)

        # ── Step 3: Compute velocities ──
        self._compute_velocities(detections)

        # ── Step 4: Boundary crossing check ──
        crossing_events = []
        if zone_points is not None:
            crossing_events = self.crossing_tracker.check_zone_entry(zone_points, detections)
        elif line is not None:
            crossing_events = self.crossing_tracker.check_crossings(line, detections)

        for event in crossing_events:
            self.event_logger.log_crossing(event, frame=frame)
            self.alert_system.trigger(
                f"{event.direction}: {event.class_name} #{event.track_id}",
                severity="CRITICAL",
                event_type="BOUNDARY",
                track_id=event.track_id
            )
            self._set_track_severity(event.track_id, "CRITICAL")

        # ── Step 5: Suspicious activity analysis ──
        suspicious_events = self.activity_analyzer.analyze(detections, zone_points)

        for event in suspicious_events:
            self.event_logger.log_suspicious(event, frame=frame)
            # Compact alert message — no full details string
            short_msg = f"{event.event_type}: #{event.track_id}"
            if event.duration > 0:
                short_msg += f" ({event.duration:.0f}s)"
            self.alert_system.trigger(
                short_msg,
                severity=event.severity,
                event_type=event.event_type,
                track_id=event.track_id
            )
            self._set_track_severity(event.track_id, event.severity)

        # ── Step 6: Draw overlays ──

        # Boundary
        if zone_points is not None:
            display.draw_zone(display_frame, zone_points)
        elif line is not None:
            display.draw_line(display_frame, line,
                              restricted_side=self.crossing_tracker.restricted_side)

        # Detections
        for det in detections:
            severity = self._get_track_severity(det.track_id)
            velocity = self._velocities.get(det.track_id)
            display.draw_detection(display_frame, det, severity=severity, velocity=velocity)

        # Loitering indicators
        for key, alert in self.activity_analyzer.active_alerts.items():
            if alert.event_type in ("LOITERING", "ZONE_LINGERING"):
                display.draw_loitering_indicator(
                    display_frame,
                    alert.location[0], alert.location[1],
                    radius=self.activity_analyzer.loiter_radius,
                    duration=alert.duration
                )

        # Alert panel (top-right, compact)
        display.draw_alert_panel(display_frame, self.alert_system)

        # Frame border
        highest = self.alert_system.get_highest_severity()
        display.draw_frame_border(display_frame, highest)

        # HUD
        stats = self.crossing_tracker.get_stats()
        display.draw_hud(
            display_frame, stats,
            fps=self._fps,
            night_mode=self.night_processor.is_night_mode
        )

        # Cleanup stale track severities
        active_ids = {d.track_id for d in detections}
        stale = [tid for tid in self._track_severities if tid not in active_ids]
        for tid in stale:
            self._track_severities.pop(tid, None)

        return display_frame
