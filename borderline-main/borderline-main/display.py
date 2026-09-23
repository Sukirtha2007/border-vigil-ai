"""
Display — tactical surveillance overlays.

Design principles:
  - Corner brackets instead of full rectangles (military/surveillance aesthetic)
  - Compact alert panel in top-right, max 4 alerts, doesn't flood the screen
  - Velocity arrows only for fast movement (>50 px/s)
  - Loitering shown as dashed circle, not text-heavy
  - HUD bar is thin and stays out of the way
  - No redundant labels — track ID shown once, small font
"""

import cv2
import numpy as np
from geometry import Point, Line
from detector import Detection
from alert_system import AlertSystem, SEVERITY_COLORS
import time
import math


# ── Colors (BGR) ──
C_NORMAL = (0, 220, 0)         # green
C_WARNING = (0, 180, 255)      # orange
C_CRITICAL = (0, 40, 255)      # red
C_VEHICLE = (220, 160, 0)      # teal-blue
C_FENCE = (0, 255, 255)        # yellow
C_ZONE_BORDER = (80, 80, 255)  # soft red
C_ZONE_FILL = (40, 40, 140)    # dark red
C_HUD_BG = (20, 20, 20)       # near-black
C_HUD_TEXT = (180, 220, 255)   # warm white
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)

SEVERITY_TO_COLOR = {
    "INFO": C_NORMAL,
    "WARNING": C_WARNING,
    "CRITICAL": C_CRITICAL,
}

# Corner bracket length as fraction of bbox dimension
BRACKET_FRAC = 0.2
BRACKET_THICKNESS = 2


def _draw_corner_brackets(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                           color: tuple, thickness: int = BRACKET_THICKNESS):
    """
    Draw tactical corner brackets instead of a full rectangle.
    Looks like:  ┌─    ─┐
                 │      │  (but only corners)
                 └─    ─┘
    """
    w = x2 - x1
    h = y2 - y1
    bl = max(int(min(w, h) * BRACKET_FRAC), 8)  # bracket arm length, min 8px

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + bl, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + bl), color, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - bl, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + bl), color, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + bl, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - bl), color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - bl, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - bl), color, thickness)


def _put_text_with_bg(frame: np.ndarray, text: str, pos: tuple,
                       font_scale: float = 0.4, color: tuple = C_WHITE,
                       bg_color: tuple = C_BLACK, thickness: int = 1,
                       padding: int = 2):
    """Draw text with a semi-transparent background pill."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos

    # Background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, det: Detection,
                   severity: str = "INFO",
                   velocity: tuple[float, float] | None = None):
    """Draw a single detection with tactical corner brackets and minimal label."""
    color = SEVERITY_TO_COLOR.get(severity, C_NORMAL)
    if det.class_name != "person":
        color = C_VEHICLE

    x1, y1, x2, y2 = int(det.xmin), int(det.ymin), int(det.xmax), int(det.ymax)

    # Corner brackets
    _draw_corner_brackets(frame, x1, y1, x2, y2, color)

    # Compact label: just ID and class initial
    cls_short = det.class_name[0].upper()  # P for person, C for car, etc.
    label = f"#{det.track_id} {cls_short}"
    _put_text_with_bg(frame, label, (x1 + 2, y1 - 4),
                       font_scale=0.35, color=color, padding=1)

    # Velocity arrow — only for meaningful speed (>50 px/s)
    if velocity is not None:
        vx, vy = velocity
        speed = (vx ** 2 + vy ** 2) ** 0.5
        if speed > 50.0:
            cx, cy = int(det.cx), int(det.cy)
            scale = min(speed / 80.0, 2.5)
            end_x = int(cx + vx * scale)
            end_y = int(cy + vy * scale)
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 1, tipLength=0.3)


def draw_line(frame: np.ndarray, line: Line, restricted_side: int = -1):
    """Draw the virtual fence line with restricted side indicator."""
    # Draw the line itself — dashed effect
    p1 = line.p1.as_tuple
    p2 = line.p2.as_tuple
    cv2.line(frame, p1, p2, C_FENCE, 2, cv2.LINE_AA)

    # Small endpoint dots
    cv2.circle(frame, p1, 4, C_FENCE, -1)
    cv2.circle(frame, p2, 4, C_FENCE, -1)

    # "FENCE" label at midpoint — small, non-intrusive
    mid_x = int((line.p1.x + line.p2.x) / 2)
    mid_y = int((line.p1.y + line.p2.y) / 2)
    _put_text_with_bg(frame, "FENCE", (mid_x - 18, mid_y - 8),
                       font_scale=0.35, color=C_FENCE, padding=2)

    # Draw a small arrow pointing toward the restricted side
    dx = line.p2.x - line.p1.x
    dy = line.p2.y - line.p1.y
    length = max((dx ** 2 + dy ** 2) ** 0.5, 1.0)
    perp_x = -dy / length
    perp_y = dx / length

    # Arrow points toward restricted side
    sign = 1.0 if restricted_side >= 0 else -1.0
    arrow_start = (mid_x, mid_y)
    arrow_end = (int(mid_x + perp_x * sign * 25), int(mid_y + perp_y * sign * 25))
    cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 0, 255), 2, tipLength=0.5)
    _put_text_with_bg(frame, "RESTRICTED", (arrow_end[0] - 30, arrow_end[1] - 5),
                       font_scale=0.3, color=(0, 100, 255), padding=1)


def draw_zone(frame: np.ndarray, zone_points: np.ndarray, alpha: float = 0.15):
    """Draw the restricted zone as a subtle semi-transparent polygon."""
    overlay = frame.copy()
    pts = zone_points.reshape((-1, 1, 2))

    cv2.fillPoly(overlay, [pts], C_ZONE_FILL)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Border — slightly thicker
    cv2.polylines(frame, [pts], isClosed=True, color=C_ZONE_BORDER, thickness=2)

    # Small centered label
    cx = int(np.mean(zone_points[:, 0]))
    cy = int(np.mean(zone_points[:, 1]))
    _put_text_with_bg(frame, "RESTRICTED ZONE", (cx - 50, cy),
                       font_scale=0.4, color=C_ZONE_BORDER, padding=3)


def draw_alert_panel(frame: np.ndarray, alert_system: AlertSystem):
    """
    Draw a compact alert panel in the top-right corner.
    Max 4 alerts, severity-sorted, non-intrusive.
    """
    alerts = alert_system.get_active_alerts()
    if not alerts:
        return

    h, w = frame.shape[:2]
    panel_w = min(320, w // 3)
    line_h = 22
    panel_h = len(alerts) * line_h + 8
    panel_x = w - panel_w - 10
    panel_y = 10

    # Semi-transparent panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  C_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Thin border
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (60, 60, 60), 1)

    # Draw each alert line
    for i, (message, severity) in enumerate(alerts):
        color = SEVERITY_TO_COLOR.get(severity, C_WHITE)
        y = panel_y + 16 + i * line_h

        # Severity dot
        cv2.circle(frame, (panel_x + 10, y - 4), 4, color, -1)

        # Truncate message to fit panel
        max_chars = panel_w // 7
        display_msg = message[:max_chars] + "..." if len(message) > max_chars else message

        cv2.putText(frame, display_msg, (panel_x + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Screen flash — subtle, short
    if alert_system.is_flashing():
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)


def draw_loitering_indicator(frame: np.ndarray, x: float, y: float,
                             radius: float = 60.0, duration: float = 0.0):
    """Draw a dashed pulsing circle — subtle loitering indicator."""
    cx, cy = int(x), int(y)
    pulse = int(radius + 6 * math.sin(time.time() * 2.5))

    # Dashed circle effect via drawing arc segments
    num_dashes = 12
    for i in range(num_dashes):
        start_angle = i * (360 / num_dashes)
        end_angle = start_angle + (360 / num_dashes) * 0.6
        cv2.ellipse(frame, (cx, cy), (pulse, pulse),
                    0, start_angle, end_angle, C_WARNING, 1, cv2.LINE_AA)

    # Duration text — small, below the circle
    if duration > 0:
        _put_text_with_bg(frame, f"{duration:.0f}s", (cx - 10, cy + pulse + 12),
                           font_scale=0.3, color=C_WARNING, padding=1)


def draw_hud(frame: np.ndarray, stats: dict, fps: float = 0.0,
             night_mode: bool = False):
    """Thin, minimal HUD bar at the bottom."""
    h, w = frame.shape[:2]
    bar_h = 28

    # Semi-transparent bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), C_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Build compact stat string
    parts = []
    crossings = stats.get('total_crossings', 0)
    intrusions = stats.get('intrusions', 0)
    tracks = stats.get('active_tracks', 0)

    parts.append(f"CROSS:{crossings}")
    if intrusions > 0:
        parts.append(f"INTRUDE:{intrusions}")
    parts.append(f"TRACKS:{tracks}")
    parts.append(f"FPS:{fps:.0f}")

    if night_mode:
        parts.append("NIGHT")

    stat_text = "  |  ".join(parts)

    y = h - 9
    cv2.putText(frame, stat_text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_HUD_TEXT, 1, cv2.LINE_AA)

    # Timestamp on the right
    ts = time.strftime("%H:%M:%S")
    ts_w, _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
    cv2.putText(frame, ts, (w - ts_w - 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_HUD_TEXT, 1, cv2.LINE_AA)


def draw_frame_border(frame: np.ndarray, severity: str | None):
    """Subtle colored border on active alerts."""
    if severity is None:
        return
    color = SEVERITY_TO_COLOR.get(severity)
    if color is None:
        return

    thickness = 3 if severity == "CRITICAL" else 1
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)
