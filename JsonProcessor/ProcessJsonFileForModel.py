from JsonProcessor.OpenJsonFile import openJsonFile

UPPER_BODY_PARTS = ["rightEar", "leftEar", "leftElbow", "leftEye", "leftHip",
                    "leftShoulder", "leftWrist", "middleHip", "neck", "nose",
                    "rightElbow", "rightEye", "rightHip", "rightShoulder", "rightWrist"]

LOWER_BODY_PARTS = ["leftBigToe", "leftHeel", "leftHip", "leftKnee", "leftSmallToe",
                    "middleHip", "rightAnkle", "leftAnkle", "rightBigToe", "rightHeel",
                    "rightHip", "rightKnee", "rightSmallToe"]


def processJson(json_file):
    upper_features = []
    lower_features = []
    upper_labels = []
    lower_labels = []

    jsonData = openJsonFile(json_file)

    # Loop through each viewing angle
    for angle in jsonData["subjectsInAngle"]:
        subjects = jsonData["subjectsInAngle"][angle]

        for subject in subjects:
            # Extract body keypoints
            body_keypoints = subject["body"]

            # Extract upper body features
            upper_keypoints = []
            keypointsExtraction(body_keypoints, UPPER_BODY_PARTS, upper_keypoints)

            # Extract lower body features
            lower_keypoints = []
            keypointsExtraction(body_keypoints, LOWER_BODY_PARTS, lower_keypoints)

            # Append the keypoints and labels
            upper_features.append(upper_keypoints)
            lower_features.append(lower_keypoints)
            upper_labels.append(subject["upperBody"])
            lower_labels.append(subject["lowerBody"])

    return upper_features, lower_features, upper_labels, lower_labels


def keypointsExtraction(body_keypoints, body_parts, keypoints):
    for part in body_parts:
        if part in body_keypoints:
            x = float(body_keypoints[part]["x"])
            y = float(body_keypoints[part]["y"])
            keypoints.extend([x, y])
