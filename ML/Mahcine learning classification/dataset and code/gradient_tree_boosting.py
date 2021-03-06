import os
import subprocess
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
from sklearn.utils import shuffle

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

NUMBER_OF_ITERATIONS = 100
NUMBER_OF_EXAMPLES = 650

def evaluate():
	
	accuracy = 0
	precision = 0
	
	
	for i in range(0, NUMBER_OF_ITERATIONS):
		#input_file = "student_grades.csv"  #this line reads the file that contains only the grades set
		input_file = "student_dataset.csv" # this line reads the file that contains all the features
		clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5, random_state=0)
		le = LabelEncoder()

		data = pd.read_csv(input_file, header = 0)
		data = shuffle(data)

		data = data[0:NUMBER_OF_EXAMPLES]
		data = data.apply(le.fit_transform)
		delim = int(len(data) * 0.8)
		# 80% training data
		data_train = data[0:delim]
		# 20% test data
		data_test = data[delim:len(data)]
			
		x_train = data_train[data_train.columns.drop('G3')]
		y_train = data_train['G3']
		x_test = data_test[data_test.columns.drop('G3')]
		y_test = data_test['G3']
		
		clf.fit(x_train, y_train)

		predicted = clf.predict(x_test)

		accuracy += accuracy_score(y_test, predicted)
		precision += precision_score(y_test, predicted)

	
	print "Accuracy:", accuracy / NUMBER_OF_ITERATIONS
	print "Precision:", precision / NUMBER_OF_ITERATIONS

	
evaluate()

