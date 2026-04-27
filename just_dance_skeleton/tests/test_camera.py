import unittest
from unittest.mock import MagicMock, patch

from core.camera.base import CameraSource
from core.camera.webcam import WebcamSource


class DummyCamera(CameraSource):
    def __init__(self):
        super().__init__(width=640, height=480, fps=30)
        self.open_called = False
        self.close_called = False

    def open(self) -> bool:
        self.open_called = True
        self.is_opened = True
        return True

    def read_frame(self):
        return False, None

    def close(self) -> None:
        self.close_called = True
        self.is_opened = False

    def is_available(self) -> bool:
        return True


class TestCameraSources(unittest.TestCase):
    def test_camera_source_context_manager_opens_and_closes(self):
        camera = DummyCamera()

        with camera as opened_camera:
            self.assertTrue(opened_camera.is_opened)
            self.assertTrue(camera.open_called)

        self.assertFalse(camera.is_opened)
        self.assertTrue(camera.close_called)

    @patch("core.camera.webcam.cv2.VideoCapture")
    def test_list_available_cameras_returns_empty_when_all_closed(self, mock_capture):
        closed_capture = MagicMock()
        closed_capture.isOpened.return_value = False
        mock_capture.return_value = closed_capture

        available = WebcamSource.list_available_cameras()

        self.assertEqual(available, [])
        self.assertEqual(mock_capture.call_count, 10)

    @patch("core.camera.webcam.cv2.VideoCapture")
    def test_list_available_cameras_returns_open_indices(self, mock_capture):
        open_indices = {0, 2, 5}

        def _capture_factory(index):
            cap = MagicMock()
            cap.isOpened.return_value = index in open_indices
            return cap

        mock_capture.side_effect = _capture_factory

        available = WebcamSource.list_available_cameras()

        self.assertEqual(available, [0, 2, 5])


if __name__ == "__main__":
    unittest.main()
