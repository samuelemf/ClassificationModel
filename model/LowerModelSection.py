import numpy as np
from sklearn.model_selection import train_test_split
from model.TrainingSection import trainingSection


def lowerModelSection(x_lower, y_lower):
    x_train_lower, x_test_lower, y_train_lower, y_test_lower = train_test_split(
        x_lower, y_lower, test_size=0.2, random_state=42
    )

    max_length = max(len(row) for row in x_train_lower)
    x_train_lower_padded = np.array([row + [0] * (max_length - len(row)) for row in x_train_lower], dtype=np.float32)
    x_test_lower_padded = np.array([row + [0] * (max_length - len(row)) for row in x_test_lower], dtype=np.float32)
    y_train_lower = np.array(y_train_lower, dtype=np.int32)

    trainingSection(x_train_lower_padded, y_train_lower, x_test_lower_padded, y_test_lower, 'Lower Body')
