def normalize_keypoints(keypoints, reference_point):
    if reference_point in keypoints:
        ref_x = float(keypoints[reference_point]["x"])
        ref_y = float(keypoints[reference_point]["y"])
    else:
        return [float(keypoints[part]["x"]) for part in keypoints] + [float(keypoints[part]["y"]) for part in keypoints]

    normalized = []
    for part in keypoints:
        x = float(keypoints[part]["x"]) - ref_x
        y = float(keypoints[part]["y"]) - ref_y
        normalized.extend([x, y])

    return normalized
