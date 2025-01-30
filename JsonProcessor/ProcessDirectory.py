import os
from JsonProcessor.ProcessJsonFileForModel import processJson


def processDirectory(directory):
    all_upper_features = []
    all_lower_features = []
    all_upper_labels = []
    all_lower_labels = []

    for filename in sorted(os.listdir(directory)):
        print("Filename: ", filename + " opened.")
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            upper_features, lower_features, upper_labels, lower_labels = processJson(file_path)

            all_upper_features.extend(upper_features)
            all_lower_features.extend(lower_features)
            all_upper_labels.extend(upper_labels)
            all_lower_labels.extend(lower_labels)

    return all_upper_features, all_lower_features, all_upper_labels, all_lower_labels
