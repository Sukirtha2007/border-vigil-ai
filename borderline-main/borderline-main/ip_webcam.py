"""
IBVAP — IP Webcam / CCTV Stream Entry Point.

Usage:
    python ip_webcam.py                                              # default stream URL
    python ip_webcam.py --source rtsp://192.168.1.100:554/stream    # RTSP camera
    python ip_webcam.py --source http://172.16.124.27:8080/video     # IP webcam app
    python ip_webcam.py --mode zone --night-mode auto                # zone + night mode
"""

import argparse
import cv2
import numpy as np
from geometry import Line, Point
from line_selector import LineSelector, ZoneSelector, RestrictedSideSelector
from detector import Detector
from crossing_tracker import CrossingTracker
from activity_analyzer import ActivityAnalyzer
from night_processor import NightProcessor
from event_logger import EventLogger
from alert_system import AlertSystem
from processor import FrameProcessor


MAX_DISPLAY_WIDTH = 1024


def resize_for_display(frame, max_width=MAX_DISPLAY_WIDTH):
    h, w = frame.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return cv2.resize(frame, (new_w, new_h)), ratio
    return frame, 1.0


def main():
    parser = argparse.ArgumentParser(
        description="IBVAP — IP Webcam / CCTV Stream Surveillance"
    )
    parser.add_argument("--source", type=str, default="http://172.16.124.27:8080/video",
                        help="IP webcam / RTSP stream URL")
    parser.add_argument("--mode", choices=["line", "zone"], default="line",
                        help="Detection mode: line or zone")
    parser.add_argument("--confidence", type=float, default=0.45,
                        help="Detection confidence threshold (default: 0.45)")
    parser.add_argument("--night-mode", choices=["auto", "on", "off"], default="auto",
                        help="Night-mode enhancement")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio alerts")
    parser.add_argument("--no-vehicles", action="store_true",
                        help="Disable vehicle detection")
    parser.add_argument("--loiter-time", type=float, default=15.0,
                        help="Loiter alert threshold in seconds")
    parser.add_argument("--cooldown", type=float, default=1.0,
                        help="Crossing alert cooldown in seconds")
    parser.add_argument("--skip-frames", type=int, default=2,
                        help="Process every Nth frame (default: 2). Higher = faster.")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="YOLO inference resolution in px (default: 480). Lower = faster.")
    args = parser.parse_args()

    print("=" * 60)
    print("  IBVAP — IP Webcam / CCTV Stream Surveillance")
    print("=" * 60)
    print(f"  Source: {args.source}")
    print(f"  Mode: {args.mode}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Night mode: {args.night_mode}")
    print("=" * 60)

    # Connect to stream
    print(f"\nConnecting to: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: Could not connect to {args.source}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Could not read first frame from stream")
        cap.release()
        return

    display_frame, ratio = resize_for_display(first_frame)
    print(f"Resolution: {first_frame.shape[1]}x{first_frame.shape[0]}"
          f" (display: {display_frame.shape[1]}x{display_frame.shape[0]})")

    # ── Select boundary ──
    line: Line | None = None
    zone_points: np.ndarray | None = None
    restricted_side: int = -1

    if args.mode == "zone":
        print("\n[SETUP] Click to define restricted zone. Right-click or Enter to close.")
        zs = ZoneSelector()
        zone_scaled = zs.select_zone(display_frame)
        zone_points = (zone_scaled / ratio).astype(np.int32)
        print(f"  Zone: {len(zone_points)} points (scaled to original resolution)")
    else:
        print("\n[SETUP] Step 1: Click 2 points to define the virtual fence.")
        ls = LineSelector()
        line_scaled = ls.select_line(display_frame)
        line = Line(
            p1=Point(line_scaled.p1.x / ratio, line_scaled.p1.y / ratio),
            p2=Point(line_scaled.p2.x / ratio, line_scaled.p2.y / ratio)
        )
        print(f"  Fence: ({line.p1.x:.0f},{line.p1.y:.0f}) → ({line.p2.x:.0f},{line.p2.y:.0f})")

        # Step 2: Calibrate restricted side
        print("[SETUP] Step 2: Click the RESTRICTED side of the fence.")
        side_selector = RestrictedSideSelector()
        restricted_side = side_selector.select_restricted_side(display_frame, line_scaled)
        print(f"  Restricted side: {'A' if restricted_side > 0 else 'B'} (value={restricted_side})")

    # ── Initialize components (after boundary selection) ──
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
    print("\n[RUNNING] Live surveillance active... Press Q to quit.\n")
    window_name = "IBVAP — Live Surveillance"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or lost connection. Attempting reconnect...")
            cap.release()
            cap = cv2.VideoCapture(args.source)
            if not cap.isOpened():
                print("Reconnection failed. Exiting.")
                break
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame after reconnect. Exiting.")
                break

        processed = processor.process_frame(frame, line, zone_points)

        proc_display, _ = resize_for_display(processed)
        cv2.imshow(window_name, proc_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Print summary
    stats = event_logger.get_stats()
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Frames processed: {frame_count}")
    print(f"  Total events: {stats['total_events']}")
    if stats['by_type']:
        for etype, count in stats['by_type'].items():
            print(f"    {etype}: {count}")
    print(f"  Database: {event_logger.db_path}")
    print(f"  Snapshots: {event_logger.snapshot_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
