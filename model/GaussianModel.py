import numpy as np
from JsonProcessor.ProcessDirectory import processDirectory
from model.UpperModelSection import upperModelExecution
from model.LowerModelSection import lowerModelSection


def execute(json_directory):
    all_upper_features, all_lower_features, all_upper_labels, all_lower_labels = processDirectory(json_directory)

    X_upper = np.array(all_upper_features, dtype=object)
    X_lower = np.array(all_lower_features, dtype=object)
    y_upper = np.array(all_upper_labels)
    y_lower = np.array(all_lower_labels)

    upperModelExecution(X_upper, y_upper)
    lowerModelSection(X_lower, y_lower)
