import cv2
import numpy as np
import win32.win32gui as wind32
from mss import mss
from .constants import GAME_WINDOW_NAME


def getWindowGeometry(name: str) -> tuple:
    """
    Get the geometry of a window.
    """
    hwnd = wind32.FindWindow(None, name)
    left, top, right, bottom = wind32.GetWindowRect(hwnd)

    return left + 10, top + 40, right - 10, bottom - 10


class GameViewer:
    def __init__(self) -> None:
        self.window_name = GAME_WINDOW_NAME
        self.sct = mss()

    @property
    def bounding_box(self):
        return getWindowGeometry(self.window_name)


    def get_obs(self):
        processed_img = self.get_frame()
        return processed_img

    def get_frame(
        self,
        size=(256, 280),
    ) -> np.ndarray:
        """
        Pulls a frame from the game and processes it

        Args:
            size (tuple, optional): size to resize the screenshot to.
            Defaults to (256, 256).

        Returns:
            np.ndarray: processed frame
        """
        img = cv2.resize(
            self.get_raw_frame(),
            size,
        )
        print(img.shape)
        # cut the bottom part of the screen
        return img[:256, :, :]

    def get_raw_frame(self):
        """
        Returns the raw frame
        """
        return np.array(self.sct.grab(self.bounding_box))

    def view(self):
        """
        Shows the current frame
        """
        it = 0
        while True:
            it += 1
            cur_frame = self.get_frame()

            cv2.imshow(
                "processed",
                cv2.resize(
                    cur_frame,
                    (256, 256),
                ),
            )
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    viewer = GameViewer()
    viewer.view()
