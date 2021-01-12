import os
import subprocess
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


input_file = "student_grades.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('G3')], data['G3']


clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=15, random_state=1)
clf3 = GaussianNB()


eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']):
	scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
	print("Accuracy: %0.4f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
