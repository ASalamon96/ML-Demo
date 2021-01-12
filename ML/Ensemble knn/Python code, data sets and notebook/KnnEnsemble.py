#import libraries
import pandas as pd
import numpy as np

from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.utils import shuffle


from sklearn.ensemble import VotingClassifier
import math  

# get a voting ensemble of models
def get_voting(n):
	k=-1; count=0; models = list(); label="-NN"; labelList=[];
	while k<n: 
		k=k+2;
		count=count+1;
		labelList.append(str(k)+label)
		# define the base models
		models.append((str(k)+label, KNeighborsClassifier(n_neighbors=k)))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
 
# get a list of models to evaluate
def get_models(n):
	models = dict()
	k=-1; count=0; label="-NN"; labelList=[];
	while k<n: 
		k=k+2;
		count=count+1;
		labelList.append(str(k)+label)
		# define the base models
		if(k<10):
			models['  '+str(k)+label] = KNeighborsClassifier(n_neighbors=k)
		elif(k>10 and k<100):
			models[' '+str(k)+label] = KNeighborsClassifier(n_neighbors=k)
		else:
			models[str(k)+label] = KNeighborsClassifier(n_neighbors=k)
		
	models['ensemble'] = get_voting(n)
	return models

# evaluate a give model using cross-validation
def evaluate_model(model):
	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

print('Evaluate QSAR dataset')
input_file = "QSAR .csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F43')], data['F43']

n=int(math.sqrt(1055))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate Australian dataset')
input_file = "australian.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F15')], data['F15']

n=int(math.sqrt(690))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate Balance dataset')
input_file = "balance.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F1')], data['F1']

n=int(math.sqrt(625))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate Banknote dataset')
input_file = "banknote.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F5')], data['F5']

n=int(math.sqrt(1372))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate Haberman dataset')
input_file = "haberman.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F4')], data['F4']

n=int(math.sqrt(306))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate Heart dataset')
input_file = "heart.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F14')], data['F14']

n=int(math.sqrt(271))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate Ionosphere dataset')
input_file = "ionosphere.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F35')], data['F35']

n=int(math.sqrt(351))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate Iris dataset')
input_file = "iris.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F5')], data['F5']

n=int(math.sqrt(151))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))



print('Evaluate Liver dataset')
input_file = "liver.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F7')], data['F7']

n=int(math.sqrt(345))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate Parkinson dataset')
input_file = "parkinson.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F1')], data['F1']

n=int(math.sqrt(168))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate Sonar dataset')
input_file = "sonar.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F61')], data['F61']

n=int(math.sqrt(209))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate Wine dataset')
input_file = "wine.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F1')], data['F1']

n=int(math.sqrt(179))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate EEG dataset')
input_file = "EEG.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F15')], data['F15']

n=int(math.sqrt(14980))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))

print('Evaluate Letter-Recognition dataset')
input_file = "letter-recognition.csv"

data = pd.read_csv(input_file, header = 0)

X, y = data[data.columns.drop('F1')], data['F1']

n=int(math.sqrt(20000))


if(n % 2 == 0):
	n=n-1

models = get_models(n)
# evaluate the models and store results
results, names = list(), list()
bestName="1NN"; bestAccuracy=0;
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	zipped= zip(names, results)
names, results = zip(*sorted(zipped))
for x in range (len(names)): 	
	print('%s %.4f ' % (names[x], mean(results[x])))
	if(mean(results[x])> bestAccuracy):
		bestName= names[x]; 
		bestAccuracy= mean(results[x]);
print('Best accuracy :%s with accuracy %.4f '% (bestName, bestAccuracy))
