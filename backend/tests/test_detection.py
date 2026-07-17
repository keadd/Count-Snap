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


def _make_two_groups_image() -> bytes:
    image = np.full((760, 1200, 3), 238, dtype=np.uint8)
    base_shape = np.array(
        [[-28, -18], [-7, -30], [25, -22], [35, 4], [19, 28], [-15, 27], [-36, 7]],
        dtype=np.float32,
    )
    for index in range(10):
        row, column = divmod(index, 5)
        transform = cv2.getRotationMatrix2D((0, 0), (index % 3 - 1) * 8, 1.0)
        points = cv2.transform(base_shape[None, :, :], transform)[0]
        points += np.array([100 + column * 105, 180 + row * 210])
        cv2.fillPoly(image, [np.round(points).astype(np.int32)], (40, 55, 220))

    for index in range(9):
        row, column = divmod(index, 3)
        cv2.circle(image, (760 + column * 120, 150 + row * 180), 30 + index % 2, (220, 80, 35), -1)

    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError("Could not encode synthetic two-group image.")
    return buffer.tobytes()


def _make_nested_color_parts_image() -> bytes:
    image = np.full((700, 1000, 3), 238, dtype=np.uint8)
    for index in range(8):
        row, column = divmod(index, 4)
        center_x = 100 + column * 210
        center_y = 170 + row * 270
        cv2.rectangle(image, (center_x - 48, center_y - 48), (center_x + 48, center_y + 48), (45, 55, 220), -1)
        cv2.rectangle(image, (center_x - 36, center_y - 36), (center_x + 36, center_y + 36), (55, 66, 195), -1)

    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError("Could not encode synthetic nested-color image.")
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

    def test_returns_multiple_groups_with_detections(self) -> None:
        result = detect_repeated_contours(_make_two_groups_image(), min_area=600, min_repeat=8)

        self.assertEqual([group["count"] for group in result["repeatGroups"][:2]], [10, 9])
        self.assertEqual(len(result["repeatGroups"][0]["detections"]), 10)
        self.assertEqual(len(result["repeatGroups"][1]["detections"]), 9)
        self.assertEqual(result["selectedGroupId"], result["repeatGroups"][0]["id"])

    def test_target_point_selects_matching_group(self) -> None:
        result = detect_repeated_contours(
            _make_two_groups_image(),
            min_area=600,
            min_repeat=8,
            target_point=(760, 150),
        )

        self.assertTrue(result["targetMatched"])
        self.assertEqual(result["count"], 9)
        self.assertEqual(result["selectedGroupId"], result["repeatGroups"][1]["id"])

    def test_roi_excludes_groups_outside_region(self) -> None:
        result = detect_repeated_contours(
            _make_two_groups_image(),
            min_area=600,
            min_repeat=8,
            roi=(20, 80, 570, 430),
        )

        self.assertEqual(result["count"], 10)
        self.assertEqual(len(result["repeatGroups"]), 1)
        self.assertEqual(result["roi"], {"x": 20, "y": 80, "width": 570, "height": 430})

    def test_removes_nested_boxes_from_same_part(self) -> None:
        result = detect_repeated_contours(
            _make_nested_color_parts_image(),
            min_area=800,
            min_repeat=8,
        )

        self.assertEqual(result["candidateCount"], 8)
        self.assertEqual(result["count"], 8)
        self.assertEqual(result["repeatGroups"][0]["count"], 8)


if __name__ == "__main__":
    unittest.main()
