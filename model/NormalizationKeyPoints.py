CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed


def normalize_keypoints(keypoints, reference_point, body_parts):
    if reference_point not in keypoints or float(keypoints[reference_point]["x"]) == 0:
        return []  # Skip normalization if reference is missing

    # Get reference keypoint
    ref_x = float(keypoints[reference_point]["x"])
    ref_y = float(keypoints[reference_point]["y"])

    normalized = []
    for part, values in keypoints.items():
        if part in body_parts:
            try:
                x = float(values["x"])
                y = float(values["y"])
                c = float(values["c"])

                # Ignore missing keypoints (x=0, y=0, c=0) or low-confidence keypoints
                if (x == 0 and y == 0) or c < CONFIDENCE_THRESHOLD:
                    continue

                # Normalize relative to the reference point
                norm_x = x - ref_x
                norm_y = y - ref_y
                normalized.extend([norm_x, norm_y])

            except (ValueError, TypeError, KeyError) as e:
                print(f"Error processing keypoint '{part}': {e}")
                continue  # Skip this keypoint and move on

    return normalized
