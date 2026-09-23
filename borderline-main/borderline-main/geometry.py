from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass
class Point:
    x: float
    y: float

    @property
    def as_tuple(self) -> tuple:
        return (int(self.x), int(self.y))

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class Line:
    p1: Point
    p2: Point


def side_of_point(line: Line, point: Point) -> int:
    """Returns +1, -1, or 0 depending on which side of the line the point lies."""
    cross = (line.p2.x - line.p1.x) * (point.y - line.p1.y) - \
            (line.p2.y - line.p1.y) * (point.x - line.p1.x)
    if cross > 0:
        return 1
    elif cross < 0:
        return -1
    return 0


def bottom_center(xmin: float, ymin: float, xmax: float, ymax: float) -> Point:
    """Bottom-center of a bounding box — best anchor point for ground-plane crossing checks."""
    cx = (xmin + xmax) / 2
    return Point(cx, ymax)


def centroid_from_bbox(xmin: float, ymin: float, xmax: float, ymax: float) -> Point:
    """Center of a bounding box."""
    return Point((xmin + xmax) / 2, (ymin + ymax) / 2)


def distance(p1: Point, p2: Point) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def bbox_aspect_ratio(xmin: float, ymin: float, xmax: float, ymax: float) -> float:
    """Width / Height ratio.  A standing person is ~0.4, crawling person is > 1.0."""
    w = xmax - xmin
    h = ymax - ymin
    if h == 0:
        return 0.0
    return w / h


def bbox_area(xmin: float, ymin: float, xmax: float, ymax: float) -> float:
    return (xmax - xmin) * (ymax - ymin)


def point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
    pts = polygon.reshape((-1, 1, 2)).astype(np.float32)
    return cv2.pointPolygonTest(pts, (point.x, point.y), False) >= 0


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns angle in degrees between two 2D vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))
