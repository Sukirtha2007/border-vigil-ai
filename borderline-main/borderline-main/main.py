"""
IBVAP — Intelligent Border Video Analytics Platform
Main entry point for local video file analysis.

Usage:
    python main.py test1.mp4                         # line mode (default)
    python main.py test1.mp4 --mode zone             # zone mode
    python main.py test1.mp4 --night-mode on         # force night enhancement
    python main.py test1.mp4 --confidence 0.5        # higher confidence threshold
    python main.py test1.mp4 --no-audio              # disable alert sounds
"""

import argparse
import cv2
import numpy as np
from video_loader import VideoLoader
from line_selector import LineSelector, ZoneSelector, RestrictedSideSelector
from geometry import Line, Point
from detector import Detector
from crossing_tracker import CrossingTracker
from activity_analyzer import ActivityAnalyzer
from night_processor import NightProcessor
from event_logger import EventLogger
from alert_system import AlertSystem
from processor import FrameProcessor


MAX_DISPLAY_WIDTH = 1280


def resize_for_display(frame: np.ndarray, max_width: int = MAX_DISPLAY_WIDTH):
    h, w = frame.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return cv2.resize(frame, (new_w, new_h)), ratio
    return frame, 1.0


def main():
    parser = argparse.ArgumentParser(
        description="IBVAP — Intelligent Border Video Analytics Platform"
    )
    parser.add_argument("video_path", type=str, help="Path to .mp4 video file")
    parser.add_argument("--mode", choices=["line", "zone"], default="line",
                        help="Detection mode: line (virtual fence) or zone (restricted area)")
    parser.add_argument("--confidence", type=float, default=0.45,
                        help="Detection confidence threshold (default: 0.45)")
    parser.add_argument("--night-mode", choices=["auto", "on", "off"], default="auto",
                        help="Night-mode enhancement: auto (default), on, off")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio alerts")
    parser.add_argument("--no-vehicles", action="store_true",
                        help="Disable vehicle detection (person-only mode)")
    parser.add_argument("--loiter-time", type=float, default=15.0,
                        help="Seconds of stationary presence before loiter alert (default: 15)")
    parser.add_argument("--cooldown", type=float, default=1.0,
                        help="Seconds between crossing alerts for same track (default: 1.0)")
    parser.add_argument("--skip-frames", type=int, default=2,
                        help="Process every Nth frame (default: 2). Higher = faster but less responsive. 1 = no skip.")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="YOLO inference resolution in px (default: 480). Lower = faster. Try 320 for max speed, 640 for max accuracy.")
    args = parser.parse_args()

    print("=" * 60)
    print("  IBVAP — Intelligent Border Video Analytics Platform")
    print("=" * 60)
    print(f"  Video: {args.video_path}")
    print(f"  Mode: {args.mode}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Night mode: {args.night_mode}")
    print(f"  Vehicles: {'disabled' if args.no_vehicles else 'enabled'}")
    print("=" * 60)

    # Load video
    loader = VideoLoader(args.video_path)
    first_frame = loader.read_first_frame()

    # Resize for boundary selection UI
    display_frame, ratio = resize_for_display(first_frame)
    print(f"\nResolution: {first_frame.shape[1]}x{first_frame.shape[0]}"
          f" (display: {display_frame.shape[1]}x{display_frame.shape[0]})")

    # ── Select boundary ──
    line: Line | None = None
    zone_points: np.ndarray | None = None
    restricted_side: int = -1  # default

    if args.mode == "zone":
        print("\n[SETUP] Select a restricted zone by clicking points on the frame.")
        print("        Left-click to add points. Right-click or Enter to close.")
        selector = ZoneSelector()
        zone_scaled = selector.select_zone(display_frame)
        zone_points = (zone_scaled / ratio).astype(np.int32)
        print(f"  Zone defined with {len(zone_points)} points.\n")
    else:
        print("\n[SETUP] Step 1: Select a virtual fence by clicking 2 points.")
        selector = LineSelector()
        line_scaled = selector.select_line(display_frame)
        line = Line(
            p1=Point(line_scaled.p1.x / ratio, line_scaled.p1.y / ratio),
            p2=Point(line_scaled.p2.x / ratio, line_scaled.p2.y / ratio)
        )
        print(f"  Fence: ({line.p1.x:.0f},{line.p1.y:.0f}) → "
              f"({line.p2.x:.0f},{line.p2.y:.0f})")

        # Step 2: Calibrate restricted side
        print("[SETUP] Step 2: Click the RESTRICTED side of the fence.")
        side_selector = RestrictedSideSelector()
        restricted_side = side_selector.select_restricted_side(display_frame, line_scaled)
        print(f"  Restricted side: {'A' if restricted_side > 0 else 'B'} (value={restricted_side})\n")

    # ── Initialize all components (after boundary selection so we have restricted_side) ──
    detector = Detector(
        confidence=args.confidence,
        detect_vehicles=not args.no_vehicles,
        imgsz=args.imgsz,
        skip_frames=args.skip_frames
    )
    crossing_tracker = CrossingTracker(
        cooldown_seconds=args.cooldown,
        restricted_side=restricted_side
    )
    activity_analyzer = ActivityAnalyzer(loiter_time=args.loiter_time)
    night_processor = NightProcessor(mode=args.night_mode)
    event_logger = EventLogger()
    alert_system = AlertSystem(enable_audio=not args.no_audio)

    processor = FrameProcessor(
        detector=detector,
        crossing_tracker=crossing_tracker,
        activity_analyzer=activity_analyzer,
        night_processor=night_processor,
        event_logger=event_logger,
        alert_system=alert_system
    )

    # ── Main loop ──
    print("[RUNNING] Processing video... Press Q to quit.\n")
    window_name = "IBVAP — Border Surveillance"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for i, frame in enumerate(loader):
        # Process frame through the full pipeline
        processed = processor.process_frame(frame, line, zone_points)

        # Resize for display
        proc_display, _ = resize_for_display(processed)
        cv2.imshow(window_name, proc_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ──
    loader.release()
    cv2.destroyAllWindows()

    # Print summary
    stats = event_logger.get_stats()
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Total events logged: {stats['total_events']}")
    if stats['by_type']:
        for etype, count in stats['by_type'].items():
            print(f"    {etype}: {count}")
    if stats['by_severity']:
        for sev, count in stats['by_severity'].items():
            print(f"    [{sev}]: {count}")
    print(f"  Event database: {event_logger.db_path}")
    print(f"  Alert snapshots: {event_logger.snapshot_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
