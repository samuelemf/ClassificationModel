from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


def trainingSection(X_train_padded, y_train, X_test_padded, y_test, section):
    model_upper = GaussianNB()
    model_upper.fit(X_train_padded, y_train)
    y_pred_upper = model_upper.predict(X_test_padded)
    accuracy_upper = accuracy_score(y_test, y_pred_upper)

    print(f'{section} Classification Accuracy: {accuracy_upper}')
    print(f'{section} Classification Report:')
    print(classification_report(y_test, y_pred_upper))
