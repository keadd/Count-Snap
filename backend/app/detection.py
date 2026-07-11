from __future__ import annotations

import uuid

import cv2
import numpy as np


def _odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _decode_work_image(image_bytes: bytes, max_dimension: int = 1600) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image.")

    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(width, height))
    work = cv2.resize(image, (round(width * scale), round(height * scale))) if scale < 1 else image.copy()
    return image, work, scale, width, height


def _lab_center_to_rgb(center: np.ndarray) -> tuple[list[int], str]:
    lab_pixel = np.uint8([[np.clip(center, 0, 255)]])
    rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)[0, 0]
    rgb = [int(value) for value in rgb_pixel]
    return rgb, "#{:02x}{:02x}{:02x}".format(*rgb)


def _hue_distance(left: float, right: float) -> float:
    delta = abs(float(left) - float(right))
    return min(delta, 180.0 - delta)


def _rgb_to_hsv(rgb: list[int]) -> np.ndarray:
    rgb_pixel = np.uint8([[rgb]])
    return cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0, 0].astype(float)


def _analyze_components(
    cleaned: np.ndarray,
    scaled_min_area: float,
    min_dimension: int,
    work_width: int,
    work_height: int,
    edge_margin: int,
    center_bounds: tuple[float, float, float, float],
    work_area: int,
) -> tuple[list[dict], list[dict]]:
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(cleaned, 8)
    raw_components = []
    valid_components = []
    for component_index in range(1, component_count):
        x, y, w, h, area = [int(value) for value in stats[component_index]]
        if area < max(18, scaled_min_area * 0.22):
            continue

        bbox_area = max(1, w * h)
        density = area / bbox_area
        aspect = max(w / max(1, h), h / max(1, w))
        center_x, center_y = centroids[component_index]
        is_edge = (
            x <= edge_margin
            or y <= edge_margin
            or x + w >= work_width - edge_margin
            or y + h >= work_height - edge_margin
        )
        is_center = (
            center_bounds[0] <= center_x <= center_bounds[2]
            and center_bounds[1] <= center_y <= center_bounds[3]
        )
        is_slender = aspect > 3.2 or density < 0.24
        is_too_large = area > work_area * 0.12 or w > work_width * 0.62 or h > work_height * 0.62
        component = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "density": density,
            "aspect": aspect,
            "isEdge": is_edge,
            "isCenter": is_center,
            "isSlender": is_slender,
            "isTooLarge": is_too_large,
        }
        raw_components.append(component)

        if area >= scaled_min_area and w >= min_dimension and h >= min_dimension and not is_slender and not is_too_large:
            valid_components.append(component)

    return raw_components, valid_components


def _score_color_candidate(
    index: int,
    rgb: list[int],
    hex_color: str,
    cleaned: np.ndarray,
    raw_components: list[dict],
    valid_components: list[dict],
    work_area: int,
    scale: float,
    distance_threshold: float,
    group_indexes: list[int] | None = None,
) -> dict:
    all_areas = [component["area"] for component in raw_components]
    valid_areas = [component["area"] for component in valid_components]
    median_area = float(np.median(valid_areas)) if valid_areas else 0.0
    area_cv = float(np.std(valid_areas) / median_area) if median_area else 2.0
    max_area_ratio = (max(all_areas) / work_area) if all_areas else 0.0
    mask_ratio = cv2.countNonZero(cleaned) / work_area
    edge_ratio = (
        sum(1 for component in raw_components if component["isEdge"]) / len(raw_components)
        if raw_components
        else 1.0
    )
    center_ratio = (
        sum(1 for component in valid_components if component["isCenter"]) / len(valid_components)
        if valid_components
        else 0.0
    )
    slender_ratio = (
        sum(1 for component in raw_components if component["isSlender"]) / len(raw_components)
        if raw_components
        else 1.0
    )
    valid_efficiency = len(valid_components) / len(raw_components) if raw_components else 0.0
    size_consistency = 1.0 / (1.0 + min(area_cv, 3.0))
    coverage_penalty = max(0.0, mask_ratio - 0.12) * 85.0
    giant_penalty = max_area_ratio * 90.0
    fragment_penalty = max(0, len(raw_components) - len(valid_components) * 2) * 0.12

    score = (
        len(valid_components) * 1.75
        + center_ratio * 14.0
        + valid_efficiency * 18.0
        + size_consistency * 7.0
        - min(area_cv, 3.0) * 2.5
        - giant_penalty
        - edge_ratio * 10.0
        - slender_ratio * 8.0
        - coverage_penalty
        - fragment_penalty
    )
    if len(valid_components) < 2:
        score -= 6.0
    if mask_ratio > 0.32:
        score -= 16.0

    candidate = {
        "index": int(index),
        "color": {"rgb": rgb, "hex": hex_color},
        "score": round(float(score), 3),
        "componentCount": int(len(raw_components)),
        "validCount": int(len(valid_components)),
        "medianArea": round(median_area / (scale * scale), 2) if median_area else 0,
        "areaCv": round(area_cv, 3),
        "maxAreaRatio": round(float(max_area_ratio), 4),
        "maskRatio": round(float(mask_ratio), 4),
        "edgeRatio": round(float(edge_ratio), 3),
        "centerRatio": round(float(center_ratio), 3),
        "slenderRatio": round(float(slender_ratio), 3),
        "validEfficiency": round(float(valid_efficiency), 3),
        "distanceThreshold": round(distance_threshold, 2),
    }
    if group_indexes and len(group_indexes) > 1:
        candidate["colorFamilyGroupIndexes"] = group_indexes
        candidate["mergedColorGroupIndexes"] = [item for item in group_indexes if item != index]
    return candidate


def detect_objects(
    image_bytes: bytes,
    min_area: int = 900,
    threshold: int = 0,
    blur: int = 7,
    invert: bool = True,
) -> dict:
    _, work, scale, width, height = _decode_work_image(image_bytes)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur_size = _odd_kernel(blur)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    threshold_mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if threshold <= 0:
        _, binary = cv2.threshold(blurred, 0, 255, threshold_mode | cv2.THRESH_OTSU)
        threshold_used = "otsu"
    else:
        _, binary = cv2.threshold(blurred, threshold, 255, threshold_mode)
        threshold_used = threshold

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    scaled_min_area = max(1, min_area * scale * scale)
    work_area = work.shape[0] * work.shape[1]

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < scaled_min_area or area > work_area * 0.85:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 8:
            continue

        original_bbox = [
            round(x / scale),
            round(y / scale),
            round(w / scale),
            round(h / scale),
        ]
        detections.append(
            {
                "id": f"obj_{uuid.uuid4().hex[:8]}",
                "bbox": original_bbox,
                "area": round(area / (scale * scale), 2),
            }
        )

    detections.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, detection in enumerate(detections, start=1):
        detection["label"] = str(index)

    return {
        "count": len(detections),
        "imageWidth": width,
        "imageHeight": height,
        "detections": detections,
        "params": {
            "minArea": min_area,
            "threshold": threshold_used,
            "blur": blur_size,
            "invert": invert,
        },
    }


def detect_auto_color_blocks(
    image_bytes: bytes,
    min_area: int = 900,
    cluster_count: int = 8,
) -> dict:
    _, work, scale, width, height = _decode_work_image(image_bytes)
    work_height, work_width = work.shape[:2]
    work_area = work_height * work_width
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)

    margin_x = round(work_width * 0.16)
    margin_y = round(work_height * 0.16)
    center_crop = lab[margin_y : work_height - margin_y, margin_x : work_width - margin_x]
    if center_crop.size == 0:
        center_crop = lab

    samples = center_crop.reshape(-1, 3)
    max_samples = 70000
    if len(samples) > max_samples:
        step = max(1, len(samples) // max_samples)
        samples = samples[::step][:max_samples]

    if len(samples) < 20:
        return {
            "count": 0,
            "imageWidth": width,
            "imageHeight": height,
            "detections": [],
            "selectedColor": None,
            "selectedScore": 0,
            "candidateColorGroups": [],
            "params": {"mode": "auto_color_blocks", "minArea": min_area},
        }

    k = max(2, min(cluster_count, len(samples)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.8)
    cv2.setRNGSeed(12345)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)

    scaled_min_area = max(90, min_area * scale * scale * 0.55)
    min_dimension = max(7, round(np.sqrt(scaled_min_area) * 0.35))
    edge_margin = max(3, round(min(work_width, work_height) * 0.012))
    center_bounds = (
        work_width * 0.14,
        work_height * 0.14,
        work_width * 0.86,
        work_height * 0.86,
    )

    candidate_groups = []
    candidate_records = []

    flat_lab = lab.reshape(-1, 3)
    for group_index, center in enumerate(centers):
        assigned = samples[labels == group_index]
        if len(assigned) == 0:
            continue

        assigned_distances = np.linalg.norm(assigned - center, axis=1)
        adaptive_threshold = float(np.percentile(assigned_distances, 82) * 1.45)
        distance_threshold = float(np.clip(max(13.0, adaptive_threshold), 13.0, 38.0))

        distance_map = np.linalg.norm(flat_lab - center, axis=1).reshape(work_height, work_width)
        mask = np.where(distance_map <= distance_threshold, 255, 0).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        raw_components, valid_components = _analyze_components(
            cleaned,
            scaled_min_area,
            min_dimension,
            work_width,
            work_height,
            edge_margin,
            center_bounds,
            work_area,
        )

        rgb, hex_color = _lab_center_to_rgb(center)
        candidate = _score_color_candidate(
            int(group_index),
            rgb,
            hex_color,
            cleaned,
            raw_components,
            valid_components,
            work_area,
            scale,
            distance_threshold,
        )
        candidate_groups.append(candidate)
        candidate_records.append(
            {
                "candidate": candidate,
                "center": center,
                "mask": cleaned,
                "rawComponents": raw_components,
                "validComponents": valid_components,
            }
        )

    family_records = []
    for seed_record in candidate_records:
        seed_candidate = seed_record["candidate"]
        seed_hsv = _rgb_to_hsv(seed_candidate["color"]["rgb"])
        family_mask = seed_record["mask"].copy()
        family_indexes = [seed_candidate["index"]]
        for record in candidate_records:
            candidate = record["candidate"]
            if candidate["index"] == seed_candidate["index"] or candidate["validCount"] < 2:
                continue

            candidate_hsv = _rgb_to_hsv(candidate["color"]["rgb"])
            high_chroma_match = (
                seed_hsv[1] >= 45
                and candidate_hsv[1] >= 45
                and _hue_distance(seed_hsv[0], candidate_hsv[0]) <= 10
            )
            neutral_match = (
                seed_hsv[1] < 45
                and candidate_hsv[1] < 45
                and np.linalg.norm(record["center"] - seed_record["center"]) <= 16
            )
            if not (high_chroma_match or neutral_match):
                continue

            if candidate["maskRatio"] > max(0.1, seed_candidate["maskRatio"] * 2.8):
                continue
            if candidate["edgeRatio"] > 0.22 or candidate["slenderRatio"] > 0.22:
                continue
            if candidate["score"] < max(6.0, seed_candidate["score"] * 0.28):
                continue

            family_mask = cv2.bitwise_or(family_mask, record["mask"])
            family_indexes.append(candidate["index"])

        if len(family_indexes) > 1:
            kernel = np.ones((3, 3), np.uint8)
            family_mask = cv2.morphologyEx(family_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        raw_components, valid_components = _analyze_components(
            family_mask,
            scaled_min_area,
            min_dimension,
            work_width,
            work_height,
            edge_margin,
            center_bounds,
            work_area,
        )
        family_candidate = _score_color_candidate(
            seed_candidate["index"],
            seed_candidate["color"]["rgb"],
            seed_candidate["color"]["hex"],
            family_mask,
            raw_components,
            valid_components,
            work_area,
            scale,
            seed_candidate["distanceThreshold"],
            family_indexes,
        )
        family_records.append(
            {
                "candidate": family_candidate,
                "mask": family_mask,
                "validComponents": valid_components,
            }
        )

    selected_record = max(family_records, key=lambda item: item["candidate"]["score"]) if family_records else None
    selected_group = selected_record["candidate"] if selected_record else None
    selected_components = selected_record["validComponents"] if selected_record else []
    selected_mask = selected_record["mask"] if selected_record else None
    candidate_groups.extend(item["candidate"] for item in family_records if len(item["candidate"].get("colorFamilyGroupIndexes", [])) > 1)
    candidate_groups.sort(key=lambda item: item["score"], reverse=True)

    selected_score = float(selected_group["score"]) if selected_group else 0.0
    if not selected_group or not selected_components or selected_score <= -1.0:
        return {
            "count": 0,
            "imageWidth": width,
            "imageHeight": height,
            "detections": [],
            "selectedColor": selected_group["color"] if selected_group else None,
            "selectedScore": round(selected_score, 3),
            "candidateColorGroups": candidate_groups,
            "params": {
                "mode": "auto_color_blocks",
                "minArea": min_area,
                "clusterCount": cluster_count,
            },
        }

    selected_group["selected"] = True
    median_area = float(np.median([component["area"] for component in selected_components]))
    detections = []
    for component in selected_components:
        x = component["x"]
        y = component["y"]
        w = component["w"]
        h = component["h"]
        area = component["area"]
        estimated_count = 1
        if median_area and area > median_area * 1.75:
            estimated_count = int(np.clip(round(area / median_area), 1, 8))

        split_boxes = []
        if estimated_count > 1 and selected_mask is not None:
            split_boxes = _split_touching_round_parts(selected_mask, (x, y, w, h), scale)

        if split_boxes:
            for bbox in split_boxes:
                detections.append(
                    {
                        "id": f"auto_color_{uuid.uuid4().hex[:8]}",
                        "bbox": bbox,
                        "area": round(bbox[2] * bbox[3], 2),
                        "count": 1,
                    }
                )
        else:
            bbox = [
                round(x / scale),
                round(y / scale),
                round(w / scale),
                round(h / scale),
            ]
            detections.append(
                {
                    "id": f"auto_color_{uuid.uuid4().hex[:8]}",
                    "bbox": bbox,
                    "area": round(area / (scale * scale), 2),
                    "count": estimated_count,
                }
            )

    detections.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, detection in enumerate(detections, start=1):
        detection["label"] = str(index)

    return {
        "count": sum(int(detection.get("count", 1)) for detection in detections),
        "imageWidth": width,
        "imageHeight": height,
        "detections": detections,
        "selectedColor": selected_group["color"],
        "selectedScore": round(selected_score, 3),
        "candidateColorGroups": candidate_groups,
        "params": {
            "mode": "auto_color_blocks",
            "minArea": min_area,
            "clusterCount": cluster_count,
        },
    }


def detect_colored_objects(
    image_bytes: bytes,
    min_area: int = 900,
) -> dict:
    return detect_auto_color_blocks(image_bytes=image_bytes, min_area=min_area)


def _split_touching_round_parts(mask: np.ndarray, bbox: tuple[int, int, int, int], scale: float) -> list[list[int]]:
    x, y, w, h = bbox
    pad = 8
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(mask.shape[1], x + w + pad)
    y2 = min(mask.shape[0], y + h + pad)
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(26, min(w, h) * 0.28),
        param1=80,
        param2=12,
        minRadius=12,
        maxRadius=min(42, max(18, round(min(w, h) * 0.36))),
    )
    if circles is None:
        return []

    boxes = []
    for cx, cy, radius in np.round(circles[0]).astype(int):
        global_x = x1 + cx
        global_y = y1 + cy
        local_x1 = max(0, cx - radius - 5)
        local_y1 = max(0, cy - radius - 5)
        local_x2 = min(roi.shape[1], cx + radius + 5)
        local_y2 = min(roi.shape[0], cy + radius + 5)
        color_area = cv2.countNonZero(roi[local_y1:local_y2, local_x1:local_x2])
        if color_area < 180:
            continue

        left = round((global_x - radius - 5) / scale)
        top = round((global_y - radius - 5) / scale)
        size = round((radius * 2 + 10) / scale)
        boxes.append([left, top, size, size])

    if len(boxes) < 2:
        return []
    boxes.sort(key=lambda item: (item[1], item[0]))
    return boxes
