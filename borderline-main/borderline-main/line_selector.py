import cv2
import numpy as np
from geometry import Point, Line, side_of_point


class LineSelector:
    def __init__(self):
        self.points: list[Point] = []
        self.finished = False

    def _callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.finished:
            self.points.append(Point(x, y))
            print(f"  Point {len(self.points)}: ({x}, {y})")
            if len(self.points) >= 2:
                self.finished = True

    def select_line(self, frame, window_name: str = "Select Line - Click 2 points") -> Line:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._callback)

        while not self.finished:
            display = frame.copy()
            for pt in self.points:
                cv2.circle(display, pt.as_tuple, 5, (0, 255, 0), -1)
            if len(self.points) == 2:
                cv2.line(display, self.points[0].as_tuple, self.points[1].as_tuple, (0, 255, 0), 2)
            cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow(window_name)
        return Line(self.points[0], self.points[1])


class RestrictedSideSelector:
    """
    After a virtual fence line is drawn, this lets the operator click which
    side of the line is the RESTRICTED side. This removes the ambiguity of
    ENTERING vs LEAVING — we know which side is "in" and which is "out".
    """

    def __init__(self):
        self.clicked_point: Point | None = None
        self.finished = False

    def _callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.finished:
            self.clicked_point = Point(x, y)
            self.finished = True

    def select_restricted_side(self, frame: np.ndarray, line: Line,
                                window_name: str = "Click the RESTRICTED side") -> int:
        """
        Show the line on the frame with both sides labeled A and B.
        User clicks on the restricted side.
        Returns the side value (+1 or -1) that represents the restricted side.
        """
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._callback)

        # Compute label positions for each side
        mid_x = (line.p1.x + line.p2.x) / 2
        mid_y = (line.p1.y + line.p2.y) / 2

        # Direction perpendicular to the line
        dx = line.p2.x - line.p1.x
        dy = line.p2.y - line.p1.y
        length = max((dx ** 2 + dy ** 2) ** 0.5, 1.0)
        # Perpendicular unit vector
        perp_x = -dy / length
        perp_y = dx / length

        offset = 60  # pixels offset from line center
        side_a_pos = (int(mid_x + perp_x * offset), int(mid_y + perp_y * offset))
        side_b_pos = (int(mid_x - perp_x * offset), int(mid_y - perp_y * offset))

        while not self.finished:
            display = frame.copy()

            # Draw the line
            cv2.line(display, line.p1.as_tuple, line.p2.as_tuple, (0, 255, 255), 2)
            cv2.circle(display, line.p1.as_tuple, 6, (0, 255, 255), -1)
            cv2.circle(display, line.p2.as_tuple, 6, (0, 255, 255), -1)

            # Label both sides
            cv2.putText(display, "SIDE A", side_a_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(display, "SIDE B", side_b_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Instructions
            cv2.putText(display, "Click the RESTRICTED side (the side you want to protect)",
                        (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

            cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow(window_name)

        if self.clicked_point is None:
            return -1  # default

        # Determine which side the user clicked
        return side_of_point(line, self.clicked_point)


class ZoneSelector:
    """
    N-point polygon zone selector.
    - Left-click to add points.
    - Right-click or press Enter to close the polygon (minimum 3 points).
    """
    def __init__(self):
        self.points: list[Point] = []
        self.finished = False

    def _callback(self, event, x, y, flags, param):
        if self.finished:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append(Point(x, y))
            print(f"  Point {len(self.points)}: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click to close polygon
            if len(self.points) >= 3:
                self.finished = True
                print(f"  Zone closed with {len(self.points)} points.")

    def select_zone(self, frame, window_name: str = "Select Zone - LClick add, RClick close") -> np.ndarray:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._callback)

        while not self.finished:
            display = frame.copy()

            # Draw placed points
            for i, pt in enumerate(self.points):
                cv2.circle(display, pt.as_tuple, 5, (0, 0, 255), -1)
                # Label point number
                cv2.putText(display, str(i + 1), (pt.as_tuple[0] + 8, pt.as_tuple[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw edges
            if len(self.points) >= 2:
                pts = np.array([p.as_tuple for p in self.points], dtype=np.int32)
                cv2.polylines(display, [pts.reshape((-1, 1, 2))], isClosed=False,
                              color=(0, 0, 255), thickness=2)

            # Preview closing edge if 3+ points
            if len(self.points) >= 3:
                pts = np.array([p.as_tuple for p in self.points], dtype=np.int32)
                cv2.polylines(display, [pts.reshape((-1, 1, 2))], isClosed=True,
                              color=(0, 0, 255), thickness=1)

            # Instructions
            cv2.putText(display, "L-Click: add point | R-Click: close polygon | Q: quit",
                        (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 13:  # Enter key
                if len(self.points) >= 3:
                    self.finished = True
                    print(f"  Zone closed with {len(self.points)} points.")

        cv2.destroyWindow(window_name)
        return np.array([p.as_tuple for p in self.points], dtype=np.int32)
