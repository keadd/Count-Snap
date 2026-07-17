from __future__ import annotations

import unittest

import cv2
import numpy as np

from app.detection import detect_repeated_contours


def _make_repeated_parts_image(part_count: int) -> bytes:
    image = np.full((720, 1000, 3), 238, dtype=np.uint8)
    base_shape = np.array(
        [[-34, -18], [-10, -31], [22, -26], [37, -5], [24, 26], [-4, 32], [-38, 14]],
        dtype=np.float32,
    )
    for index in range(part_count):
        row, column = divmod(index, 5)
        angle = (index % 4 - 1.5) * 7
        scale = 1.0 + (index % 3 - 1) * 0.04
        transform = cv2.getRotationMatrix2D((0, 0), angle, scale)
        points = cv2.transform(base_shape[None, :, :], transform)[0]
        points += np.array([130 + column * 165, 210 + row * 230])
        cv2.fillPoly(image, [np.round(points).astype(np.int32)], (45, 55, 220))

    # This distractor has the same color but a clearly different size and contour.
    cv2.rectangle(image, (30, 30), (280, 105), (45, 55, 220), -1)
    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError("Could not encode synthetic test image.")
    return buffer.tobytes()


class RepeatedContourDetectionTests(unittest.TestCase):
    def test_counts_repeated_irregular_parts(self) -> None:
        result = detect_repeated_contours(
            _make_repeated_parts_image(10),
            min_area=700,
            min_repeat=8,
        )

        self.assertEqual(result["count"], 10)
        self.assertEqual(result["selectedRepeatGroup"]["count"], 10)
        self.assertGreater(result["selectedRepeatGroup"]["meanSimilarity"], 0.8)

    def test_rejects_group_below_minimum_repeat(self) -> None:
        result = detect_repeated_contours(
            _make_repeated_parts_image(7),
            min_area=700,
            min_repeat=8,
        )

        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["selectedRepeatGroup"])
        self.assertEqual(result["repeatGroups"][0]["count"], 7)


if __name__ == "__main__":
    unittest.main()
