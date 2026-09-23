"""
Activity Analyzer — behavioral analytics engine for suspicious activity detection.

Maintains per-track history (positions, timestamps) and runs anomaly detectors:
  1. Loitering — person stays within a small radius for too long
  2. Speed anomaly — person running (fast) or crawling (slow + low aspect ratio)
  3. Erratic movement — frequent direction changes (zigzag pattern)
  4. Group formation — N+ people clustering near a boundary

All detectors are stateless functions that operate on the track history.
The ActivityAnalyzer class manages the history and orchestrates the detectors.
"""

from dataclasses import dataclass, field
from collections import deque
from detector import Detection
from geometry import Point, distance, angle_between_vectors
import numpy as np
import time
import math


@dataclass
class SuspiciousEvent:
    """A detected suspicious activity."""
    event_type: str          # LOITERING, RUNNING, CRAWLING, ERRATIC, GROUP, ZONE_LINGERING
    severity: str            # INFO, WARNING, CRITICAL
    track_id: int
    class_name: str
    location: tuple[float, float]
    timestamp: float
    details: str             # human-readable description
    duration: float = 0.0    # how long the activity has been ongoing (seconds)


@dataclass
class TrackHistory:
    """Per-track movement history."""
    track_id: int
    class_name: str
    positions: deque = field(default_factory=lambda: deque(maxlen=300))  # (x, y, timestamp)
    last_alert_time: dict = field(default_factory=dict)  # event_type -> timestamp

    @property
    def latest_position(self) -> tuple[float, float] | None:
        if not self.positions:
            return None
        return (self.positions[-1][0], self.positions[-1][1])

    @property
    def latest_timestamp(self) -> float:
        if not self.positions:
            return 0.0
        return self.positions[-1][2]

    def displacement_over(self, seconds: float) -> float:
        """Total displacement of the track over the last N seconds."""
        if len(self.positions) < 2:
            return 0.0
        now = self.positions[-1][2]
        cutoff = now - seconds

        total = 0.0
        prev = None
        for x, y, t in self.positions:
            if t < cutoff:
                continue
            if prev is not None:
                total += math.sqrt((x - prev[0]) ** 2 + (y - prev[1]) ** 2)
            prev = (x, y)
        return total

    def net_displacement_over(self, seconds: float) -> float:
        """Straight-line displacement from position N seconds ago to now."""
        if len(self.positions) < 2:
            return 0.0
        now = self.positions[-1][2]
        cutoff = now - seconds

        start_pos = None
        for x, y, t in self.positions:
            if t >= cutoff:
                start_pos = (x, y)
                break

        if start_pos is None:
            return 0.0

        end_pos = (self.positions[-1][0], self.positions[-1][1])
        return math.sqrt((end_pos[0] - start_pos[0]) ** 2 +
                         (end_pos[1] - start_pos[1]) ** 2)

    def speed_pixels_per_second(self, over_seconds: float = 1.0) -> float:
        """Average speed in pixels/second over the last N seconds."""
        if len(self.positions) < 2:
            return 0.0
        now = self.positions[-1][2]
        cutoff = now - over_seconds

        total_dist = 0.0
        prev = None
        start_t = None
        end_t = None

        for x, y, t in self.positions:
            if t < cutoff:
                continue
            if start_t is None:
                start_t = t
            end_t = t
            if prev is not None:
                total_dist += math.sqrt((x - prev[0]) ** 2 + (y - prev[1]) ** 2)
            prev = (x, y)

        if start_t is None or end_t is None or (end_t - start_t) < 0.1:
            return 0.0
        return total_dist / (end_t - start_t)

    def direction_changes(self, over_seconds: float = 5.0, angle_threshold: float = 90.0) -> int:
        """Count how many times the direction changed by > angle_threshold degrees."""
        if len(self.positions) < 3:
            return 0

        now = self.positions[-1][2]
        cutoff = now - over_seconds

        # Collect recent positions
        recent = [(x, y) for x, y, t in self.positions if t >= cutoff]
        if len(recent) < 3:
            return 0

        # Sample every few positions to avoid noise from micro-movements
        step = max(1, len(recent) // 20)
        sampled = recent[::step]
        if len(sampled) < 3:
            return 0

        changes = 0
        for i in range(1, len(sampled) - 1):
            v1 = np.array([sampled[i][0] - sampled[i - 1][0],
                           sampled[i][1] - sampled[i - 1][1]])
            v2 = np.array([sampled[i + 1][0] - sampled[i][0],
                           sampled[i + 1][1] - sampled[i][1]])

            # Skip near-zero vectors (stationary or jitter noise)
            if np.linalg.norm(v1) < 8.0 or np.linalg.norm(v2) < 8.0:
                continue

            angle = angle_between_vectors(v1, v2)
            if angle > angle_threshold:
                changes += 1

        return changes


class ActivityAnalyzer:
    """
    Behavioral analytics engine. Call `analyze()` every frame with the current
    detections to get a list of suspicious events.
    """

    def __init__(self,
                 loiter_radius: float = 60.0,
                 loiter_time: float = 15.0,
                 run_speed_threshold: float = 200.0,
                 crawl_speed_threshold: float = 30.0,
                 crawl_aspect_ratio: float = 1.2,
                 erratic_direction_changes: int = 6,
                 group_radius: float = 120.0,
                 group_min_size: int = 3,
                 alert_cooldown: float = 15.0,
                 zone_linger_time: float = 10.0):
        """
        Args:
            loiter_radius: max displacement (px) to consider someone stationary
            loiter_time: seconds of being stationary before loiter alert
            run_speed_threshold: px/sec above which = running
            crawl_speed_threshold: px/sec below which + high aspect ratio = crawling
            crawl_aspect_ratio: min W/H ratio to flag as crawling posture
            erratic_direction_changes: min direction changes in 5s to flag as erratic
            group_radius: max distance between people to consider them clustered
            group_min_size: minimum cluster size to trigger group alert
            alert_cooldown: seconds between repeated alerts of same type for same track
            zone_linger_time: seconds of lingering near zone boundary before alert
        """
        self.loiter_radius = loiter_radius
        self.loiter_time = loiter_time
        self.run_speed_threshold = run_speed_threshold
        self.crawl_speed_threshold = crawl_speed_threshold
        self.crawl_aspect_ratio = crawl_aspect_ratio
        self.erratic_direction_changes = erratic_direction_changes
        self.group_radius = group_radius
        self.group_min_size = group_min_size
        self.alert_cooldown = alert_cooldown
        self.zone_linger_time = zone_linger_time

        # Per-track history
        self._tracks: dict[int, TrackHistory] = {}

        # Track cleanup
        self._stale_timeout = 15.0

        # Active alerts (for display) — event_type:track_id -> SuspiciousEvent
        self.active_alerts: dict[str, SuspiciousEvent] = {}

    def _get_or_create_track(self, det: Detection) -> TrackHistory:
        if det.track_id not in self._tracks:
            self._tracks[det.track_id] = TrackHistory(
                track_id=det.track_id,
                class_name=det.class_name
            )
        return self._tracks[det.track_id]

    def _can_alert(self, track: TrackHistory, event_type: str) -> bool:
        """Check if enough time has passed since last alert of this type for this track."""
        last = track.last_alert_time.get(event_type)
        if last is None:
            return True
        return (time.time() - last) >= self.alert_cooldown

    def _record_alert(self, track: TrackHistory, event_type: str):
        track.last_alert_time[event_type] = time.time()

    def _cleanup_stale(self):
        now = time.time()
        stale = [tid for tid, t in self._tracks.items()
                 if (now - t.latest_timestamp) > self._stale_timeout]
        for tid in stale:
            self._tracks.pop(tid, None)

        # Clean old active alerts
        stale_alerts = [k for k, v in self.active_alerts.items()
                        if (now - v.timestamp) > self.alert_cooldown]
        for k in stale_alerts:
            self.active_alerts.pop(k, None)

    def analyze(self, detections: list[Detection],
                zone_points: np.ndarray | None = None) -> list[SuspiciousEvent]:
        """
        Analyze current frame's detections for suspicious behavior.
        Call this every frame.

        Returns list of new SuspiciousEvent objects (may be empty).
        """
        now = time.time()
        events = []

        # Update track histories
        persons = []
        for det in detections:
            if det.track_id < 0:
                continue

            track = self._get_or_create_track(det)
            bx, by = det.bottom_center
            track.positions.append((bx, by, now))

            if det.class_name == "person":
                persons.append((det, track))

        # Run anomaly detectors on person tracks
        for det, track in persons:
            # --- Loitering Detection ---
            loiter_event = self._check_loitering(det, track, now)
            if loiter_event:
                events.append(loiter_event)

            # --- Speed Anomaly (Running) ---
            run_event = self._check_running(det, track, now)
            if run_event:
                events.append(run_event)

            # --- Crawling Detection ---
            crawl_event = self._check_crawling(det, track, now)
            if crawl_event:
                events.append(crawl_event)

            # --- Erratic Movement ---
            erratic_event = self._check_erratic(det, track, now)
            if erratic_event:
                events.append(erratic_event)

        # --- Group Formation Detection ---
        group_events = self._check_groups(persons, now)
        events.extend(group_events)

        # --- Zone Lingering ---
        if zone_points is not None:
            for det, track in persons:
                linger_event = self._check_zone_lingering(det, track, zone_points, now)
                if linger_event:
                    events.append(linger_event)

        # Update active alerts
        for ev in events:
            key = f"{ev.event_type}:{ev.track_id}"
            self.active_alerts[key] = ev

        # Cleanup
        self._cleanup_stale()

        return events

    def _check_loitering(self, det: Detection, track: TrackHistory,
                         now: float) -> SuspiciousEvent | None:
        """Person stays within loiter_radius for > loiter_time seconds."""
        if len(track.positions) < 10:
            return None

        net_disp = track.net_displacement_over(self.loiter_time)
        total_disp = track.displacement_over(self.loiter_time)

        # Check we have enough history
        oldest_in_window = None
        cutoff = now - self.loiter_time
        for x, y, t in track.positions:
            if t >= cutoff:
                oldest_in_window = t
                break

        if oldest_in_window is None or (now - oldest_in_window) < self.loiter_time * 0.8:
            return None  # not enough history yet

        # Loitering = stayed within a small area
        if net_disp < self.loiter_radius and total_disp < self.loiter_radius * 3:
            if self._can_alert(track, "LOITERING"):
                self._record_alert(track, "LOITERING")
                bx, by = det.bottom_center
                duration = now - oldest_in_window
                event = SuspiciousEvent(
                    event_type="LOITERING",
                    severity="WARNING",
                    track_id=det.track_id,
                    class_name=det.class_name,
                    location=(bx, by),
                    timestamp=now,
                    details=f"Person #{det.track_id} loitering for {duration:.0f}s "
                            f"(displacement: {net_disp:.0f}px)",
                    duration=duration
                )
                print(f"[SUSPICIOUS] {event.details}")
                return event
        return None

    def _check_running(self, det: Detection, track: TrackHistory,
                       now: float) -> SuspiciousEvent | None:
        """Person moving unusually fast."""
        speed = track.speed_pixels_per_second(over_seconds=1.0)

        if speed > self.run_speed_threshold:
            if self._can_alert(track, "RUNNING"):
                self._record_alert(track, "RUNNING")
                bx, by = det.bottom_center
                event = SuspiciousEvent(
                    event_type="RUNNING",
                    severity="CRITICAL",
                    track_id=det.track_id,
                    class_name=det.class_name,
                    location=(bx, by),
                    timestamp=now,
                    details=f"Person #{det.track_id} RUNNING at {speed:.0f} px/s"
                )
                print(f"[SUSPICIOUS] {event.details}")
                return event
        return None

    def _check_crawling(self, det: Detection, track: TrackHistory,
                        now: float) -> SuspiciousEvent | None:
        """Slow movement + low/wide aspect ratio = person crawling on ground."""
        if len(track.positions) < 5:
            return None

        speed = track.speed_pixels_per_second(over_seconds=2.0)
        ar = det.aspect_ratio

        # Crawling: slow speed AND wide aspect ratio (person is horizontal)
        if speed < self.crawl_speed_threshold and speed > 2.0 and ar > self.crawl_aspect_ratio:
            if self._can_alert(track, "CRAWLING"):
                self._record_alert(track, "CRAWLING")
                bx, by = det.bottom_center
                event = SuspiciousEvent(
                    event_type="CRAWLING",
                    severity="CRITICAL",
                    track_id=det.track_id,
                    class_name=det.class_name,
                    location=(bx, by),
                    timestamp=now,
                    details=f"Person #{det.track_id} appears to be CRAWLING "
                            f"(speed: {speed:.0f} px/s, aspect ratio: {ar:.2f})"
                )
                print(f"[SUSPICIOUS] {event.details}")
                return event
        return None

    def _check_erratic(self, det: Detection, track: TrackHistory,
                       now: float) -> SuspiciousEvent | None:
        """Frequent direction changes in short time = erratic/evasive movement."""
        if len(track.positions) < 15:
            return None

        changes = track.direction_changes(over_seconds=5.0, angle_threshold=90.0)

        if changes >= self.erratic_direction_changes:
            if self._can_alert(track, "ERRATIC"):
                self._record_alert(track, "ERRATIC")
                bx, by = det.bottom_center
                event = SuspiciousEvent(
                    event_type="ERRATIC",
                    severity="WARNING",
                    track_id=det.track_id,
                    class_name=det.class_name,
                    location=(bx, by),
                    timestamp=now,
                    details=f"Person #{det.track_id} moving ERRATICALLY "
                            f"({changes} sharp direction changes in 5s)"
                )
                print(f"[SUSPICIOUS] {event.details}")
                return event
        return None

    def _check_groups(self, persons: list[tuple[Detection, TrackHistory]],
                      now: float) -> list[SuspiciousEvent]:
        """N+ people clustering within a small area."""
        events = []
        if len(persons) < self.group_min_size:
            return events

        # Build list of (track_id, position)
        positions = []
        for det, track in persons:
            bx, by = det.bottom_center
            positions.append((det.track_id, bx, by, det))

        # Simple clustering: for each person, count neighbors within radius
        already_alerted = set()
        for i, (tid_i, xi, yi, det_i) in enumerate(positions):
            if tid_i in already_alerted:
                continue

            neighbors = [tid_i]
            for j, (tid_j, xj, yj, _) in enumerate(positions):
                if i == j:
                    continue
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < self.group_radius:
                    neighbors.append(tid_j)

            if len(neighbors) >= self.group_min_size:
                track_i = self._tracks.get(tid_i)
                if track_i and self._can_alert(track_i, "GROUP"):
                    self._record_alert(track_i, "GROUP")

                    # Mark all in the group so we don't duplicate
                    already_alerted.update(neighbors)

                    event = SuspiciousEvent(
                        event_type="GROUP",
                        severity="WARNING",
                        track_id=tid_i,
                        class_name="person",
                        location=(xi, yi),
                        timestamp=now,
                        details=f"GROUP of {len(neighbors)} people detected near "
                                f"({xi:.0f}, {yi:.0f}) — IDs: {neighbors}"
                    )
                    print(f"[SUSPICIOUS] {event.details}")
                    events.append(event)

        return events

    def _check_zone_lingering(self, det: Detection, track: TrackHistory,
                              zone_points: np.ndarray,
                              now: float) -> SuspiciousEvent | None:
        """Person lingering near (but not inside) a restricted zone boundary."""
        from geometry import point_in_polygon

        bx, by = det.bottom_center
        anchor = Point(bx, by)

        # Check if person is near the zone boundary (within a buffer) but not inside
        inside = point_in_polygon(anchor, zone_points)
        if inside:
            return None  # already handled by crossing tracker

        # Check distance to polygon edges
        min_edge_dist = self._point_to_polygon_distance(bx, by, zone_points)
        if min_edge_dist > self.loiter_radius:
            return None  # too far from zone

        # Person is near the zone boundary — check if they've been here a while
        net_disp = track.net_displacement_over(self.zone_linger_time)

        cutoff = now - self.zone_linger_time
        oldest_in_window = None
        for x, y, t in track.positions:
            if t >= cutoff:
                oldest_in_window = t
                break

        if oldest_in_window is None or (now - oldest_in_window) < self.zone_linger_time * 0.8:
            return None

        if net_disp < self.loiter_radius:
            if self._can_alert(track, "ZONE_LINGERING"):
                self._record_alert(track, "ZONE_LINGERING")
                event = SuspiciousEvent(
                    event_type="ZONE_LINGERING",
                    severity="WARNING",
                    track_id=det.track_id,
                    class_name=det.class_name,
                    location=(bx, by),
                    timestamp=now,
                    details=f"Person #{det.track_id} LINGERING near restricted zone "
                            f"(distance: {min_edge_dist:.0f}px, duration: {now - oldest_in_window:.0f}s)",
                    duration=now - oldest_in_window
                )
                print(f"[SUSPICIOUS] {event.details}")
                return event
        return None

    @staticmethod
    def _point_to_polygon_distance(px: float, py: float, polygon: np.ndarray) -> float:
        """Minimum distance from a point to any edge of a polygon."""
        pts = polygon.reshape(-1, 2)
        min_dist = float('inf')

        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            ax, ay = float(pts[i][0]), float(pts[i][1])
            bx, by = float(pts[j][0]), float(pts[j][1])

            # Point-to-segment distance
            abx, aby = bx - ax, by - ay
            apx, apy = px - ax, py - ay
            ab_sq = abx * abx + aby * aby

            if ab_sq < 1e-6:
                dist = math.sqrt(apx * apx + apy * apy)
            else:
                t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_sq))
                proj_x = ax + t * abx
                proj_y = ay + t * aby
                dist = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

            min_dist = min(min_dist, dist)

        return min_dist
