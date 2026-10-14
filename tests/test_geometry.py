import unittest

import numpy as np

from src.evaluate import calculate_iou
from src.utils.nms import non_max_suppression
from src.utils.windowing import sliding_window


class SlidingWindowTests(unittest.TestCase):
    def test_exact_size_returns_one_window(self):
        image = np.zeros((128, 64), dtype=np.uint8)
        windows = list(sliding_window(image, 16, (64, 128)))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][:2], (0, 0))

    def test_right_and_bottom_edges_are_included(self):
        image = np.zeros((140, 80), dtype=np.uint8)
        coordinates = [(x, y) for x, y, _ in sliding_window(image, 10, (64, 128))]
        self.assertIn((16, 12), coordinates)


class GeometryTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(calculate_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_nms_keeps_highest_score(self):
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]]
        result = non_max_suppression(boxes, [0.5, 0.9, 0.3], overlap_thresh=0.5)
        self.assertEqual(result.tolist(), [[1, 1, 11, 11], [20, 20, 30, 30]])

    def test_empty_nms_shape(self):
        self.assertEqual(non_max_suppression([]).shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
