from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bounding_box: tuple[int, int, int, int]

    @property
    def ground_point(self) -> tuple[int, int]:
        x1, _, x2, y2 = self.bounding_box

        ground_x = (x1 + x2) // 2
        ground_y = y2

        return ground_x, ground_y
