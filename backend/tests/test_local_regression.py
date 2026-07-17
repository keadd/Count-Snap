from __future__ import annotations

import unittest
from pathlib import Path

from app.detection import _bbox_containment, detect_smart_objects


LOCAL_IMAGE = Path(__file__).resolve().parents[1] / "local-test-data" / "red-transparent-parts.jpg"


@unittest.skipUnless(LOCAL_IMAGE.exists(), "Private local regression image is not available.")
class LocalPhotoRegressionTests(unittest.TestCase):
    def test_red_transparent_parts_photo(self) -> None:
        result = detect_smart_objects(
            LOCAL_IMAGE.read_bytes(),
            min_area=900,
            min_repeat=8,
        )

        self.assertEqual(result["params"]["selectedStrategy"], "auto_color_blocks")
        self.assertEqual(result["count"], 42)
        boxes = [tuple(int(value) for value in detection["bbox"]) for detection in result["detections"]]
        nested_pairs = sum(
            1
            for left_index in range(len(boxes))
            for right_index in range(left_index + 1, len(boxes))
            if _bbox_containment(boxes[left_index], boxes[right_index]) >= 0.82
        )
        self.assertEqual(nested_pairs, 0)


if __name__ == "__main__":
    unittest.main()
