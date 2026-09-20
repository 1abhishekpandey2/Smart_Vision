import cv2
import numpy as np

Point = tuple[int, int]


class PolygonZone:
    def __init__(self, points: list[Point] | tuple[Point, ...]):
        if len(points) < 3:
            raise ValueError("A polygon zone requires at least 3 points")

        self._points = tuple((int(x), int(y)) for x, y in points)

        self._contour = np.array(
            self._points,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

    @property
    def points(self) -> tuple[Point, ...]:
        return self._points

    def contains(self, point: Point) -> bool:
        result = cv2.pointPolygonTest(
            self._contour,
            (float(point[0]), float(point[1])),
            False,
        )

        return result >= 0

    def boundary_intersection(
        self,
        start: Point,
        end: Point,
    ) -> Point | None:
        """
        Find the first point where the movement segment from start to end
        intersects the polygon boundary.
        """

        px, py = start
        ex, ey = end

        rx = ex - px
        ry = ey - py

        intersections = []

        for index in range(len(self._points)):
            qx, qy = self._points[index]

            next_index = (index + 1) % len(self._points)
            sx_end, sy_end = self._points[next_index]

            sx = sx_end - qx
            sy = sy_end - qy

            denominator = (rx * sy) - (ry * sx)

            if abs(denominator) < 1e-9:
                continue

            qpx = qx - px
            qpy = qy - py

            t = ((qpx * sy) - (qpy * sx)) / denominator
            u = ((qpx * ry) - (qpy * rx)) / denominator

            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                intersection_x = px + (t * rx)
                intersection_y = py + (t * ry)

                intersections.append(
                    (
                        t,
                        (
                            round(intersection_x),
                            round(intersection_y),
                        ),
                    )
                )

        if not intersections:
            return None

        intersections.sort(key=lambda item: item[0])

        return intersections[0][1]
