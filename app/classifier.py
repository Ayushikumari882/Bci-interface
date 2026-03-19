from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm = SVC(probability=True, random_state=42)
    clf = CalibratedClassifierCV(svm, method='sigmoid', cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    return clf, acc, y_test, y_pred, y_prob, cm, cv_mean