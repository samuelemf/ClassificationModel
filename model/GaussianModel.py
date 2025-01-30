import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from JsonProcessor.ProcessDirectory import processDirectory


def execute(json_directory):
    all_upper_features, all_lower_features, all_upper_labels, all_lower_labels = processDirectory(json_directory)

    X_upper = np.array(all_upper_features)
    X_lower = np.array(all_lower_features)
    y_upper = np.array(all_upper_labels)
    y_lower = np.array(all_lower_labels)

    X_train_upper, X_test_upper, y_train_upper, y_test_upper = train_test_split(
        X_upper, y_upper, test_size=0.2, random_state=42
    )

    model_upper = GaussianNB()
    model_upper.fit(X_train_upper, y_train_upper)
    y_pred_upper = model_upper.predict(X_test_upper)
    accuracy_upper = accuracy_score(y_test_upper, y_pred_upper)
    print(f'Upper Body Classification Accuracy: {accuracy_upper}')
    print('Upper Body Classification Report:')
    print(classification_report(y_test_upper, y_pred_upper))

    # Train and evaluate for lower body classification
    X_train_lower, X_test_lower, y_train_lower, y_test_lower = train_test_split(
        X_lower, y_lower, test_size=0.2, random_state=42
    )

    model_lower = GaussianNB()
    model_lower.fit(X_train_lower, y_train_lower)
    y_pred_lower = model_lower.predict(X_test_lower)
    accuracy_lower = accuracy_score(y_test_lower, y_pred_lower)
    print(f'Lower Body Classification Accuracy: {accuracy_lower}')
    print('Lower Body Classification Report:')
    print(classification_report(y_test_lower, y_pred_lower))
