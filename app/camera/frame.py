from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Frame:
    image: Any
    timestamp: datetime
    camera_id: str
