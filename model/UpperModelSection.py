import numpy as np
from sklearn.model_selection import train_test_split
from model.TrainingSection import trainingSection


def upperModelExecution(x_upper, y_upper):
    X_train_upper, X_test_upper, y_train_upper, y_test_upper = train_test_split(
        x_upper, y_upper, test_size=0.2, random_state=42
    )

    max_length = max(len(row) for row in X_train_upper)
    x_train_upper_padded = np.array([row + [0] * (max_length - len(row)) for row in X_train_upper], dtype=np.float32)
    x_test_upper_padded = np.array([row + [0] * (max_length - len(row)) for row in X_test_upper], dtype=np.float32)
    y_train_upper = np.array(y_train_upper, dtype=np.int32)

    trainingSection(x_train_upper_padded, y_train_upper, x_test_upper_padded, y_test_upper, 'Upper Body')
