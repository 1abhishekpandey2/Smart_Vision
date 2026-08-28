import numpy as np


class FakeVideoCapture:
    def __init__(
        self,
        device_index: int,
        read_results: list[bool] | None = None,
    ):
        self.device_index = device_index
        self.read_results = read_results or []
        self.read_count = 0
        self.is_open = True

    def isOpened(self):
        return self.is_open

    def read(self):
        self.read_count += 1

        if self.read_results:
            result_index = self.read_count - 1

            if result_index < len(self.read_results):
                success = self.read_results[result_index]
            else:
                success = self.read_results[-1]
        else:
            success = True

        if not success:
            return False, None

        image = np.zeros((480, 640, 3), dtype=np.uint8)

        return True, image

    def release(self):
        self.is_open = False
