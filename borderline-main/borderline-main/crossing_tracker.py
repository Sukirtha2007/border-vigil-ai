"""
Crossing Tracker — consumes ByteTrack-tracked detections and checks for
boundary crossings (virtual line or polygon zone).

All tracking/association is handled upstream by ByteTrack in detector.py.
This module only checks geometry: did a tracked object cross the line or enter the zone?

Direction is determined by a calibrated restricted_side value set during setup,
so ENTERING/LEAVING are never ambiguous.
"""

from dataclasses import dataclass
from geometry import Point, Line, side_of_point, point_in_polygon
from detector import Detection
import numpy as np
import time


@dataclass
class CrossingEvent:
    track_id: int
    class_name: str
    from_side: int          # +1 or -1 for line mode; 0/1 for zone mode
    to_side: int
    direction: str          # "INTRUDING" or "RETREATING" or "CROSSED"
    timestamp: float
    location: tuple[float, float]


class CrossingTracker:
    """
    Stateful tracker that maintains per-track side history and detects
    line crossings or zone entries/exits.
    """

    def __init__(self, cooldown_seconds: float = 1.0, restricted_side: int = -1):
        """
        Args:
            cooldown_seconds: minimum time between two crossing events for the same track.
            restricted_side: which side of the line is restricted (+1 or -1).
                             Set via RestrictedSideSelector during setup.
                             Movement TOWARD this side = INTRUDING.
                             Movement AWAY from this side = RETREATING.
        """
        self.cooldown_seconds = cooldown_seconds
        self.restricted_side = restricted_side

        # Per-track state
        self._prev_sides: dict[int, int] = {}
        self._prev_zone_states: dict[int, bool] = {}
        self._last_event_time: dict[int, float] = {}

        # Counters
        self.crossing_count = 0
        self.intrusion_count = 0
        self.retreat_count = 0
        self.entry_count = 0
        self.exit_count = 0

        # Stale track cleanup
        self._last_seen: dict[int, float] = {}
        self._stale_timeout = 10.0

    def _is_on_cooldown(self, track_id: int) -> bool:
        last = self._last_event_time.get(track_id)
        if last is None:
            return False
        return (time.time() - last) < self.cooldown_seconds

    def _record_event(self, track_id: int):
        self._last_event_time[track_id] = time.time()

    def _cleanup_stale_tracks(self):
        """Remove state for tracks that haven't been seen in a while."""
        now = time.time()
        stale_ids = [tid for tid, t in self._last_seen.items()
                     if (now - t) > self._stale_timeout]
        for tid in stale_ids:
            self._prev_sides.pop(tid, None)
            self._prev_zone_states.pop(tid, None)
            self._last_event_time.pop(tid, None)
            self._last_seen.pop(tid, None)

    def check_crossings(self, line: Line, detections: list[Detection]) -> list[CrossingEvent]:
        """
        Check if any tracked detection has crossed the virtual line since last frame.
        Uses bottom-center of bbox as the anchor point.
        """
        events = []
        now = time.time()

        for det in detections:
            if det.track_id < 0:
                continue

            self._last_seen[det.track_id] = now

            bx, by = det.bottom_center
            anchor = Point(bx, by)
            current_side = side_of_point(line, anchor)

            if current_side == 0:
                continue

            prev_side = self._prev_sides.get(det.track_id)
            self._prev_sides[det.track_id] = current_side

            if prev_side is None:
                continue

            if current_side == prev_side:
                continue

            if self._is_on_cooldown(det.track_id):
                continue

            # Direction based on calibrated restricted_side
            if current_side == self.restricted_side:
                direction = "INTRUDING"
                self.intrusion_count += 1
            else:
                direction = "RETREATING"
                self.retreat_count += 1

            self.crossing_count += 1
            self._record_event(det.track_id)

            event = CrossingEvent(
                track_id=det.track_id,
                class_name=det.class_name,
                from_side=prev_side,
                to_side=current_side,
                direction=direction,
                timestamp=now,
                location=(bx, by)
            )
            events.append(event)
            print(f"[ALERT] {direction} — {det.class_name} #{det.track_id} | "
                  f"Total: {self.crossing_count} (intrusions: {self.intrusion_count})")

        self._cleanup_stale_tracks()
        return events

    def check_zone_entry(self, zone_points: np.ndarray, detections: list[Detection]) -> list[CrossingEvent]:
        """
        Check if any tracked detection has entered or exited the defined polygon zone.
        """
        events = []
        now = time.time()

        for det in detections:
            if det.track_id < 0:
                continue

            self._last_seen[det.track_id] = now

            bx, by = det.bottom_center
            anchor = Point(bx, by)
            inside = point_in_polygon(anchor, zone_points)

            prev_inside = self._prev_zone_states.get(det.track_id)
            self._prev_zone_states[det.track_id] = inside

            if prev_inside is None:
                continue

            if inside == prev_inside:
                continue

            if self._is_on_cooldown(det.track_id):
                continue

            if inside:
                direction = "ZONE_INTRUDED"
                self.entry_count += 1
            else:
                direction = "ZONE_EXITED"
                self.exit_count += 1

            self.crossing_count += 1
            self._record_event(det.track_id)

            event = CrossingEvent(
                track_id=det.track_id,
                class_name=det.class_name,
                from_side=0 if prev_inside else 1,
                to_side=1 if inside else 0,
                direction=direction,
                timestamp=now,
                location=(bx, by)
            )
            events.append(event)
            print(f"[ALERT] {direction} — {det.class_name} #{det.track_id} | "
                  f"Entries: {self.entry_count}, Exits: {self.exit_count}")

        self._cleanup_stale_tracks()
        return events

    def get_stats(self) -> dict:
        return {
            "total_crossings": self.crossing_count,
            "intrusions": self.intrusion_count,
            "retreats": self.retreat_count,
            "zone_entries": self.entry_count,
            "zone_exits": self.exit_count,
            "active_tracks": len(self._last_seen),
        }
