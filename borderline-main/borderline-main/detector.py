from dataclasses import dataclass, field
from ultralytics import YOLO
import numpy as np


@dataclass
class Detection:
    """A single tracked detection with all metadata."""
    track_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    class_name: str
    confidence: float

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) / 2

    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) / 2

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2, self.ymax)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def aspect_ratio(self) -> float:
        """W/H ratio. Standing person ~0.4, crawling > 1.0."""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    @property
    def area(self) -> float:
        return self.width * self.height


# COCO class IDs we care about
PERSON_CLASS = 0
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
ALL_TARGET_CLASSES = {PERSON_CLASS: "person", **VEHICLE_CLASSES}


class StaticObjectFilter:
    """
    Filters out static objects (mannequins, dolls, posters) that YOLO
    misclassifies as persons.

    Logic: If a tracked "person" hasn't moved more than `move_thresh` pixels
    over `settle_frames` consecutive frames, it's flagged as static and
    suppressed from downstream processing.

    Once flagged, the track stays suppressed until it moves significantly,
    which handles the case where a real person stands still for a while then
    moves — they'll un-suppress.
    """

    def __init__(self, settle_frames: int = 90, move_thresh: float = 15.0,
                 wake_thresh: float = 40.0):
        """
        Args:
            settle_frames: Number of consecutive near-stationary frames before
                           flagging as static. At 30fps, 90 = 3 seconds.
            move_thresh: Max pixel displacement per frame to consider "not moving".
            wake_thresh: Displacement needed to un-suppress a static object.
        """
        self.settle_frames = settle_frames
        self.move_thresh = move_thresh
        self.wake_thresh = wake_thresh

        # track_id -> {"pos": (cx, cy), "still_count": int, "is_static": bool}
        self._state: dict[int, dict] = {}

    def update_and_filter(self, detections: list['Detection']) -> list['Detection']:
        """
        Update internal state and return only non-static detections.
        """
        active_ids = set()
        filtered = []

        for det in detections:
            if det.track_id < 0:
                filtered.append(det)
                continue

            active_ids.add(det.track_id)
            cx, cy = det.cx, det.cy

            state = self._state.get(det.track_id)
            if state is None:
                # New track — initialize
                self._state[det.track_id] = {
                    "pos": (cx, cy),
                    "still_count": 0,
                    "is_static": False,
                    "initial_pos": (cx, cy)
                }
                filtered.append(det)
                continue

            # Compute displacement from last frame
            px, py = state["pos"]
            disp = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            state["pos"] = (cx, cy)

            if state["is_static"]:
                # Check if it woke up (started moving significantly)
                ix, iy = state["initial_pos"]
                total_disp = ((cx - ix) ** 2 + (cy - iy) ** 2) ** 0.5
                if total_disp > self.wake_thresh:
                    state["is_static"] = False
                    state["still_count"] = 0
                    state["initial_pos"] = (cx, cy)
                    filtered.append(det)
                # else: still static, suppress it
                continue

            # Not yet flagged — count still frames
            if disp < self.move_thresh:
                state["still_count"] += 1
            else:
                state["still_count"] = 0
                state["initial_pos"] = (cx, cy)

            if state["still_count"] >= self.settle_frames:
                # Flag as static — this is a doll/mannequin/poster
                state["is_static"] = True
                print(f"[STATIC FILTER] Track #{det.track_id} flagged as static object "
                      f"(stationary for {state['still_count']} frames)")
                continue

            filtered.append(det)

        # Cleanup tracks that disappeared
        stale = [tid for tid in self._state if tid not in active_ids]
        for tid in stale:
            self._state.pop(tid, None)

        return filtered

    @property
    def static_ids(self) -> set[int]:
        """Currently suppressed track IDs."""
        return {tid for tid, s in self._state.items() if s["is_static"]}


class Detector:
    """YOLOv8 detector with built-in BoT-SORT tracking and static object filtering."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.45,
                 detect_vehicles: bool = True,
                 tracker_config: str = "botsort_custom.yaml",
                 imgsz: int = 480,
                 skip_frames: int = 2):
        """
        Args:
            model_name: YOLO model file.
            confidence: Detection confidence threshold.
            detect_vehicles: Whether to detect cars/trucks/etc.
            tracker_config: Path to tracker YAML config.
            imgsz: Inference resolution. Lower = faster. 480 is a good CPU tradeoff.
                   640 is YOLO default. 320 is fastest but less accurate.
            skip_frames: Process every Nth frame. 1 = every frame (slowest),
                         2 = every other frame, 3 = every 3rd frame.
                         Skipped frames reuse the previous detections.
        """
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.detect_vehicles = detect_vehicles
        self._target_classes = set(ALL_TARGET_CLASSES.keys()) if detect_vehicles else {PERSON_CLASS}
        self._imgsz = imgsz
        self._skip_frames = max(1, skip_frames)
        self._frame_count = 0
        self._cached_detections: list[Detection] = []

        # Static object filter — suppresses mannequins, dolls, posters
        self.static_filter = StaticObjectFilter(
            settle_frames=90,   # 3 seconds at 30fps to confirm static
            move_thresh=12.0,   # px/frame jitter tolerance
            wake_thresh=40.0    # px total displacement to un-suppress
        )

        # Resolve tracker config path relative to this file's directory
        import os
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), tracker_config)
        if os.path.exists(config_path):
            self._tracker_config = config_path
            print(f"[TRACKER] Using BoT-SORT with ReID: {config_path}")
        else:
            self._tracker_config = "botsort.yaml"
            print(f"[TRACKER] WARNING: Custom config not found, using default botsort.yaml")

        print(f"[DETECTOR] Inference resolution: {imgsz}px | "
              f"Frame skip: process 1/{skip_frames} | "
              f"Static filter: ON (settle={90} frames)")

    def detect_and_track(self, frame: np.ndarray) -> list[Detection]:
        """
        Run YOLOv8 detection + BoT-SORT tracking.

        Frame skipping: On skipped frames, returns cached detections from the
        last processed frame. The tracker still gets called every Nth frame
        to maintain track state, but the expensive CNN inference is avoided
        on intermediate frames.
        """
        self._frame_count += 1

        # Frame skipping — reuse cached detections on non-process frames
        if self._frame_count % self._skip_frames != 0 and self._cached_detections:
            return self._cached_detections

        results = self.model.track(
            frame,
            persist=True,
            tracker=self._tracker_config,
            conf=self.confidence,
            imgsz=self._imgsz,
            verbose=False
        )[0]

        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            self._cached_detections = detections
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])

            if cls_id not in self._target_classes:
                continue

            conf = float(box.conf[0])

            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = ALL_TARGET_CLASSES.get(cls_id, "unknown")

            detections.append(Detection(
                track_id=track_id,
                xmin=x1, ymin=y1, xmax=x2, ymax=y2,
                class_name=class_name,
                confidence=conf
            ))

        # Apply static object filter — removes dolls, mannequins, etc.
        detections = self.static_filter.update_and_filter(detections)

        self._cached_detections = detections
        return detections

    def detect_only(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection WITHOUT tracking. Useful for single-frame analysis.
        """
        results = self.model(frame, conf=self.confidence, imgsz=self._imgsz, verbose=False)[0]
        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            if cls_id not in self._target_classes:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = ALL_TARGET_CLASSES.get(cls_id, "unknown")

            detections.append(Detection(
                track_id=i,
                xmin=x1, ymin=y1, xmax=x2, ymax=y2,
                class_name=class_name,
                confidence=conf
            ))

        return detections
