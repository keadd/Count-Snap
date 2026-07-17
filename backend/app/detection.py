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


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    left_x1, left_y1, left_w, left_h = left
    right_x1, right_y1, right_w, right_h = right
    left_x2 = left_x1 + left_w
    left_y2 = left_y1 + left_h
    right_x2 = right_x1 + right_w
    right_y2 = right_y1 + right_h
    intersection_w = max(0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_h = max(0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_w * intersection_h
    union = left_w * left_h + right_w * right_h - intersection
    return intersection / union if union else 0.0


def _bbox_containment(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    left_x1, left_y1, left_w, left_h = left
    right_x1, right_y1, right_w, right_h = right
    left_x2 = left_x1 + left_w
    left_y2 = left_y1 + left_h
    right_x2 = right_x1 + right_w
    right_y2 = right_y1 + right_h
    intersection_w = max(0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_h = max(0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_w * intersection_h
    smaller_area = min(left_w * left_h, right_w * right_h)
    return intersection / smaller_area if smaller_area else 0.0


def _are_repeat_candidates_duplicate(left: dict, right: dict) -> bool:
    if _bbox_iou(left["bbox"], right["bbox"]) >= 0.58:
        return True
    if _bbox_containment(left["bbox"], right["bbox"]) < 0.82:
        return False

    area_ratio = min(left["area"], right["area"]) / max(left["area"], right["area"])
    color_distance = float(np.linalg.norm(left["labColor"] - right["labColor"]))
    shape_distance = float(cv2.matchShapes(left["contour"], right["contour"], cv2.CONTOURS_MATCH_I1, 0.0))
    return area_ratio >= 0.42 and color_distance <= 50.0 and shape_distance <= 0.72


def _deduplicate_group_detections(detections: list[dict]) -> list[dict]:
    ordered = sorted(
        detections,
        key=lambda item: item["bbox"][2] * item["bbox"][3],
        reverse=True,
    )
    kept = []
    for detection in ordered:
        bbox = tuple(int(value) for value in detection["bbox"])
        if any(
            _bbox_iou(bbox, tuple(int(value) for value in item["bbox"])) >= 0.62
            or _bbox_containment(bbox, tuple(int(value) for value in item["bbox"])) >= 0.82
            for item in kept
        ):
            continue
        kept.append(detection)
    kept.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return kept


def _score_detection_result(result: dict, min_repeat: int) -> dict:
    detections = result.get("detections", [])
    boxes = [tuple(int(value) for value in detection["bbox"]) for detection in detections]
    nested_pairs = 0
    overlap_pairs = 0
    for left_index in range(len(boxes)):
        for right_index in range(left_index + 1, len(boxes)):
            if _bbox_containment(boxes[left_index], boxes[right_index]) >= 0.82:
                nested_pairs += 1
            elif _bbox_iou(boxes[left_index], boxes[right_index]) >= 0.35:
                overlap_pairs += 1

    areas = np.array(
        [max(1.0, float(detection.get("area", detection["bbox"][2] * detection["bbox"][3]))) for detection in detections],
        dtype=float,
    )
    median_area = float(np.median(areas)) if len(areas) else 0.0
    area_cv = float(np.std(areas) / median_area) if median_area else 3.0
    size_consistency = 1.0 / (1.0 + min(area_cv, 3.0) * 2.0)
    count = len(detections)
    score = count * 1.45 + size_consistency * 22.0 - nested_pairs * 18.0 - overlap_pairs * 7.0
    if count < min_repeat:
        score -= 24.0

    repeat_group = result.get("selectedRepeatGroup")
    if repeat_group:
        score += float(repeat_group.get("meanSimilarity", 0.0)) * 12.0
    selected_score = float(result.get("selectedScore", 0.0))
    if selected_score:
        score += float(np.tanh(max(0.0, selected_score) / 35.0)) * 12.0

    return {
        "score": round(float(score), 3),
        "count": count,
        "nestedPairs": nested_pairs,
        "overlapPairs": overlap_pairs,
        "areaCv": round(area_cv, 3),
        "sizeConsistency": round(size_consistency, 3),
    }


def _repeat_similarity(left: dict, right: dict) -> tuple[float, float]:
    area_similarity = min(left["area"], right["area"]) / max(left["area"], right["area"])
    aspect_similarity = float(np.exp(-abs(np.log(left["aspect"] / right["aspect"]))))
    solidity_similarity = max(0.0, 1.0 - abs(left["solidity"] - right["solidity"]) * 2.2)
    compactness_similarity = max(0.0, 1.0 - abs(left["compactness"] - right["compactness"]) * 2.5)
    color_distance = float(np.linalg.norm(left["labColor"] - right["labColor"]))
    color_similarity = float(np.exp(-((color_distance / 30.0) ** 2)))
    shape_distance = float(cv2.matchShapes(left["contour"], right["contour"], cv2.CONTOURS_MATCH_I1, 0.0))
    shape_similarity = float(np.exp(-shape_distance * 3.8))
    score = (
        shape_similarity * 0.4
        + area_similarity * 0.24
        + color_similarity * 0.18
        + aspect_similarity * 0.1
        + solidity_similarity * 0.05
        + compactness_similarity * 0.03
    )
    return score, shape_distance


def _extract_repeat_candidates(
    work: np.ndarray,
    lab: np.ndarray,
    centers: np.ndarray,
    samples: np.ndarray,
    labels: np.ndarray,
    scaled_min_area: float,
) -> list[dict]:
    work_height, work_width = work.shape[:2]
    work_area = work_height * work_width
    flat_lab = lab.reshape(-1, 3)
    edge_margin = max(3, round(min(work_width, work_height) * 0.01))
    candidates: list[dict] = []

    for group_index, center in enumerate(centers):
        assigned = samples[labels == group_index]
        if len(assigned) == 0:
            continue

        assigned_distances = np.linalg.norm(assigned - center, axis=1)
        adaptive_threshold = float(np.percentile(assigned_distances, 80) * 1.35)
        distance_threshold = float(np.clip(max(12.0, adaptive_threshold), 12.0, 32.0))
        distance_map = np.linalg.norm(flat_lab - center, axis=1).reshape(work_height, work_width)
        mask = np.where(distance_map <= distance_threshold, 255, 0).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < max(30.0, scaled_min_area * 0.42) or area > work_area * 0.09:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 7 or h < 7:
                continue

            aspect = max(w / max(1, h), h / max(1, w))
            bbox_area = max(1, w * h)
            extent = area / bbox_area
            hull = cv2.convexHull(contour)
            hull_area = max(1.0, float(cv2.contourArea(hull)))
            solidity = area / hull_area
            perimeter = max(1.0, float(cv2.arcLength(contour, True)))
            compactness = float(4.0 * np.pi * area / (perimeter * perimeter))
            touches_edge = (
                x <= edge_margin
                or y <= edge_margin
                or x + w >= work_width - edge_margin
                or y + h >= work_height - edge_margin
            )
            if aspect > 4.8 or extent < 0.16 or solidity < 0.34:
                continue
            if touches_edge and area > work_area * 0.018:
                continue

            candidates.append(
                {
                    "contour": contour,
                    "bbox": (x, y, w, h),
                    "area": area,
                    "aspect": aspect,
                    "extent": extent,
                    "solidity": solidity,
                    "compactness": compactness,
                    "labColor": center.astype(float),
                    "touchesEdge": touches_edge,
                }
            )

    candidates.sort(key=lambda item: (item["area"], item["solidity"] + item["extent"]), reverse=True)
    deduplicated: list[dict] = []
    for candidate in candidates:
        if any(_are_repeat_candidates_duplicate(candidate, kept) for kept in deduplicated):
            continue
        deduplicated.append(candidate)
        if len(deduplicated) >= 240:
            break

    for index, candidate in enumerate(deduplicated):
        candidate["index"] = index
    return deduplicated


def _build_repeat_groups(candidates: list[dict], scale: float) -> list[dict]:
    if len(candidates) < 2:
        return []

    adjacency = [set([index]) for index in range(len(candidates))]
    similarities: dict[tuple[int, int], float] = {}
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            similarity, shape_distance = _repeat_similarity(candidates[left_index], candidates[right_index])
            similarities[(left_index, right_index)] = similarity
            area_ratio = min(candidates[left_index]["area"], candidates[right_index]["area"]) / max(
                candidates[left_index]["area"], candidates[right_index]["area"]
            )
            color_distance = float(
                np.linalg.norm(candidates[left_index]["labColor"] - candidates[right_index]["labColor"])
            )
            if similarity >= 0.64 and area_ratio >= 0.48 and shape_distance <= 0.58 and color_distance <= 42.0:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    components: list[list[int]] = []
    remaining = set(range(len(candidates)))
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        if len(component) >= 2:
            components.append(component)

    groups = []
    seen_members: set[tuple[int, ...]] = set()
    for component in components:
        def average_similarity(candidate_index: int) -> float:
            values = []
            for other_index in component:
                if candidate_index == other_index:
                    continue
                key = tuple(sorted((candidate_index, other_index)))
                values.append(similarities.get(key, 0.0))
            return float(np.mean(values)) if values else 1.0

        medoid_index = max(component, key=average_similarity)
        members = []
        member_similarities = []
        for candidate_index in component:
            if candidate_index == medoid_index:
                similarity = 1.0
            else:
                key = tuple(sorted((medoid_index, candidate_index)))
                similarity = similarities.get(key, 0.0)
            if similarity >= 0.6:
                members.append(candidate_index)
                member_similarities.append(similarity)

        member_key = tuple(sorted(members))
        if len(members) < 2 or member_key in seen_members:
            continue
        seen_members.add(member_key)

        member_candidates = [candidates[index] for index in members]
        areas = np.array([candidate["area"] for candidate in member_candidates], dtype=float)
        median_area = float(np.median(areas))
        area_cv = float(np.std(areas) / median_area) if median_area else 2.0
        mean_similarity = float(np.mean(member_similarities)) if member_similarities else 0.0
        edge_ratio = sum(1 for candidate in member_candidates if candidate["touchesEdge"]) / len(member_candidates)
        median_lab = np.median(np.array([candidate["labColor"] for candidate in member_candidates]), axis=0)
        rgb, hex_color = _lab_center_to_rgb(median_lab)
        score = len(members) * 5.0 + mean_similarity * 22.0 + 10.0 / (1.0 + area_cv * 4.0) - edge_ratio * 5.0
        groups.append(
            {
                "members": members,
                "count": len(members),
                "score": round(float(score), 3),
                "meanSimilarity": round(mean_similarity, 3),
                "medianArea": round(median_area / (scale * scale), 2),
                "areaCv": round(area_cv, 3),
                "color": {"rgb": rgb, "hex": hex_color},
            }
        )

    groups.sort(key=lambda item: (item["count"], item["score"]), reverse=True)
    return groups


def _build_target_repeat_group(
    candidates: list[dict],
    target_point: tuple[float, float],
    scale: float,
) -> dict | None:
    if not candidates:
        return None

    target_x, target_y = target_point
    containing = []
    nearby = []
    for candidate in candidates:
        x, y, w, h = candidate["bbox"]
        center_x = x + w / 2
        center_y = y + h / 2
        distance = float(np.hypot(target_x - center_x, target_y - center_y))
        if x <= target_x <= x + w and y <= target_y <= y + h:
            containing.append((distance, candidate))
        else:
            nearby.append((distance, candidate))

    if containing:
        reference = min(containing, key=lambda item: item[0])[1]
    else:
        distance, reference = min(nearby, key=lambda item: item[0])
        _, _, reference_w, reference_h = reference["bbox"]
        if distance > max(18.0, max(reference_w, reference_h) * 0.7):
            return None

    members = [reference["index"]]
    member_similarities = [1.0]
    for candidate in candidates:
        if candidate["index"] == reference["index"]:
            continue
        similarity, shape_distance = _repeat_similarity(reference, candidate)
        area_ratio = min(reference["area"], candidate["area"]) / max(reference["area"], candidate["area"])
        color_distance = float(np.linalg.norm(reference["labColor"] - candidate["labColor"]))
        if similarity >= 0.58 and area_ratio >= 0.42 and shape_distance <= 0.72 and color_distance <= 50.0:
            members.append(candidate["index"])
            member_similarities.append(similarity)

    member_candidates = [candidates[index] for index in members]
    areas = np.array([candidate["area"] for candidate in member_candidates], dtype=float)
    median_area = float(np.median(areas))
    area_cv = float(np.std(areas) / median_area) if median_area else 2.0
    mean_similarity = float(np.mean(member_similarities))
    median_lab = np.median(np.array([candidate["labColor"] for candidate in member_candidates]), axis=0)
    rgb, hex_color = _lab_center_to_rgb(median_lab)
    score = len(members) * 5.0 + mean_similarity * 22.0 + 10.0 / (1.0 + area_cv * 4.0)
    return {
        "id": "target_group",
        "members": members,
        "count": len(members),
        "score": round(float(score), 3),
        "meanSimilarity": round(mean_similarity, 3),
        "medianArea": round(median_area / (scale * scale), 2),
        "areaCv": round(area_cv, 3),
        "color": {"rgb": rgb, "hex": hex_color},
        "selectionMethod": "target_point",
        "referenceCandidateIndex": reference["index"],
    }


def _normalize_roi(
    width: int,
    height: int,
    roi: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None

    x, y, roi_width, roi_height = roi
    x = int(np.clip(x, 0, max(0, width - 1)))
    y = int(np.clip(y, 0, max(0, height - 1)))
    roi_width = int(np.clip(roi_width, 1, width - x))
    roi_height = int(np.clip(roi_height, 1, height - y))
    if roi_width < 12 or roi_height < 12:
        return None
    return x, y, roi_width, roi_height


def detect_repeated_contours(
    image_bytes: bytes,
    min_area: int = 900,
    min_repeat: int = 8,
    cluster_count: int = 10,
    target_point: tuple[float, float] | None = None,
    roi: tuple[int, int, int, int] | None = None,
) -> dict:
    _, work, scale, width, height = _decode_work_image(image_bytes)
    normalized_roi = _normalize_roi(width, height, roi)
    offset_x = 0
    offset_y = 0
    if normalized_roi:
        roi_x, roi_y, roi_width, roi_height = normalized_roi
        offset_x = int(round(roi_x * scale))
        offset_y = int(round(roi_y * scale))
        crop_x2 = min(work.shape[1], int(round((roi_x + roi_width) * scale)))
        crop_y2 = min(work.shape[0], int(round((roi_y + roi_height) * scale)))
        work = work[offset_y:crop_y2, offset_x:crop_x2]

    work_height, work_width = work.shape[:2]
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)

    margin_x = round(work_width * 0.08)
    margin_y = round(work_height * 0.08)
    center_crop = lab[margin_y : work_height - margin_y, margin_x : work_width - margin_x]
    if center_crop.size == 0:
        center_crop = lab

    samples = center_crop.reshape(-1, 3)
    max_samples = 80000
    if len(samples) > max_samples:
        step = max(1, len(samples) // max_samples)
        samples = samples[::step][:max_samples]

    min_repeat = int(np.clip(min_repeat, 2, 50))
    if len(samples) < 20:
        return {
            "count": 0,
            "imageWidth": width,
            "imageHeight": height,
            "detections": [],
            "selectedRepeatGroup": None,
            "selectedGroupId": None,
            "repeatGroups": [],
            "candidateCount": 0,
            "targetMatched": target_point is None,
            "roi": (
                {"x": normalized_roi[0], "y": normalized_roi[1], "width": normalized_roi[2], "height": normalized_roi[3]}
                if normalized_roi
                else None
            ),
            "params": {"mode": "repeat_contours", "minArea": min_area, "minRepeat": min_repeat},
        }

    k = max(3, min(cluster_count, len(samples)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 28, 0.75)
    cv2.setRNGSeed(12345)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    scaled_min_area = max(45.0, min_area * scale * scale)
    candidates = _extract_repeat_candidates(work, lab, centers, samples, labels, scaled_min_area)
    groups = _build_repeat_groups(candidates, scale)
    for index, group in enumerate(groups, start=1):
        group["id"] = f"group_{index}"

    target_group = None
    target_matched = target_point is None
    if target_point is not None:
        local_target = (target_point[0] * scale - offset_x, target_point[1] * scale - offset_y)
        if 0 <= local_target[0] <= work_width and 0 <= local_target[1] <= work_height:
            target_group = _build_target_repeat_group(candidates, local_target, scale)
        target_matched = target_group is not None

    selected_group = target_group if target_group else next((group for group in groups if group["count"] >= min_repeat), None)
    response_groups = groups[:8]
    if target_group:
        matching_group = next(
            (group for group in response_groups if set(group["members"]) == set(target_group["members"])),
            None,
        )
        if matching_group:
            matching_group["selectionMethod"] = "target_point"
            matching_group["referenceCandidateIndex"] = target_group["referenceCandidateIndex"]
            selected_group = matching_group
        else:
            response_groups = [target_group, *response_groups[:7]]

    public_groups = []
    for group in response_groups:
        group_detections = []
        for candidate_index in group["members"]:
            candidate = candidates[candidate_index]
            x, y, candidate_width, candidate_height = candidate["bbox"]
            group_detections.append(
                {
                    "id": f"{group['id']}_{candidate_index}",
                    "bbox": [
                        round((x + offset_x) / scale),
                        round((y + offset_y) / scale),
                        round(candidate_width / scale),
                        round(candidate_height / scale),
                    ],
                    "area": round(candidate["area"] / (scale * scale), 2),
                    "count": 1,
                }
            )
        raw_detection_count = len(group_detections)
        group_detections = _deduplicate_group_detections(group_detections)
        for index, detection in enumerate(group_detections, start=1):
            detection["label"] = str(index)

        public_group = {key: value for key, value in group.items() if key != "members"}
        public_group["rawCount"] = raw_detection_count
        public_group["count"] = len(group_detections)
        public_group["meetsMinimum"] = len(group_detections) >= min_repeat
        public_group["detections"] = group_detections
        public_groups.append(public_group)

    selected_group_id = selected_group["id"] if selected_group else None
    public_selected_group = next((group for group in public_groups if group["id"] == selected_group_id), None)
    if target_group is None and (not public_selected_group or not public_selected_group["meetsMinimum"]):
        public_selected_group = next((group for group in public_groups if group["meetsMinimum"]), None)
        selected_group_id = public_selected_group["id"] if public_selected_group else None
    detections = public_selected_group["detections"] if public_selected_group else []
    return {
        "count": len(detections),
        "imageWidth": width,
        "imageHeight": height,
        "detections": detections,
        "selectedRepeatGroup": public_selected_group,
        "selectedGroupId": selected_group_id,
        "repeatGroups": public_groups,
        "candidateCount": len(candidates),
        "targetMatched": target_matched,
        "roi": (
            {"x": normalized_roi[0], "y": normalized_roi[1], "width": normalized_roi[2], "height": normalized_roi[3]}
            if normalized_roi
            else None
        ),
        "params": {
            "mode": "repeat_contours",
            "minArea": min_area,
            "minRepeat": min_repeat,
            "clusterCount": cluster_count,
        },
    }


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


def detect_smart_objects(
    image_bytes: bytes,
    min_area: int = 900,
    min_repeat: int = 8,
    target_point: tuple[float, float] | None = None,
    roi: tuple[int, int, int, int] | None = None,
) -> dict:
    repeated_result = detect_repeated_contours(
        image_bytes=image_bytes,
        min_area=min_area,
        min_repeat=min_repeat,
        target_point=target_point,
        roi=roi,
    )
    repeated_quality = _score_detection_result(repeated_result, min_repeat)

    if target_point is not None or roi is not None:
        selected = dict(repeated_result)
        selected["strategyScores"] = {"repeat_contours": repeated_quality}
        selected["params"] = {
            **repeated_result.get("params", {}),
            "mode": "smart",
            "selectedStrategy": "repeat_contours",
            "selectionReason": "interactive_constraint",
        }
        return selected

    color_result = detect_auto_color_blocks(image_bytes=image_bytes, min_area=min_area)
    color_quality = _score_detection_result(color_result, min_repeat)
    strategy_scores = {
        "repeat_contours": repeated_quality,
        "auto_color_blocks": color_quality,
    }
    if color_quality["score"] >= repeated_quality["score"]:
        selected_strategy = "auto_color_blocks"
        selected_result = color_result
    else:
        selected_strategy = "repeat_contours"
        selected_result = repeated_result

    selected = dict(selected_result)
    selected["strategyScores"] = strategy_scores
    selected["alternativeCounts"] = {
        "repeat_contours": repeated_result.get("count", 0),
        "auto_color_blocks": color_result.get("count", 0),
    }
    selected["params"] = {
        **selected_result.get("params", {}),
        "mode": "smart",
        "selectedStrategy": selected_strategy,
        "selectionReason": "quality_score",
    }
    return selected


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
