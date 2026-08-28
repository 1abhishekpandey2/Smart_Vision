from enum import Enum, auto


class CameraStatus(Enum):
    """
    Represents the current state of a camera.
    """

    STOPPED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTED = auto()
    RECONNECTING = auto()
