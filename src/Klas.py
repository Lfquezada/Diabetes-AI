
'''
K L A S

a proyect by Luis Quezada
'''

# Libraries
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Train all models and return the one with best accuracy
def getBestFitModel(X,y,scaled,testSize,returnAllResults):
	# Vars
	allAlgorithmsNames = []
	allKs = [3,5,7,11,13,15]
	allKernelModes = ['linear','rbf','poly','sigmoid']
	classifiers = []
	allResults = []

	# Splitting into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)

	# Feature Scaling
	if scaled:
		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_test = sc.transform(X_test)


	# Decision Tree Classification
	allAlgorithmsNames.append('Decision Tree')
	classifiers.append(DecisionTreeClassifier(criterion = 'entropy', random_state = 0))

	# Random Forest Classification
	allAlgorithmsNames.append('Random Forest')
	classifiers.append(RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0))

	# SVM
	for kernelMode in allKernelModes:
		allAlgorithmsNames.append('SVM ' + kernelMode)
		classifiers.append(SVC(kernel = kernelMode, random_state = 0))

	# K-NN
	for k in allKs:
		allAlgorithmsNames.append('K-NN k=' + str(k))
		classifiers.append(KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2))

	# Naive Bayes
	allAlgorithmsNames.append('Naive Bayes')
	classifiers.append(GaussianNB())

	# Logistic Regression
	allAlgorithmsNames.append('Logistic Regression')
	classifiers.append(LogisticRegression(random_state = 0))

	# Fitting to all
	for classifier in classifiers:
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		allResults.append(accuracy_score(y_test,y_pred))

	bestAccuracy = max(allResults)
	bestAccuracyIndex = allResults.index(bestAccuracy)
	bestFitAlgorithmName = allAlgorithmsNames[bestAccuracyIndex]
	bestFitClassifier = classifiers[bestAccuracyIndex]

	if returnAllResults:
		return bestFitClassifier,bestFitAlgorithmName,bestAccuracy,allAlgorithmsNames,allResults,X_test,y_test
	else:
		return bestFitClassifier,bestFitAlgorithmName,bestAccuracy


def klasPredict(X,y,scaled,testSize,threshold,inputX):

	# Vars
	allKs = [3,5,7,11,13,15]
	allKernelModes = ['linear','rbf','poly','sigmoid']
	classifiers = []
	allResults = []

	# Splitting into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)

	# Feature Scaling
	if scaled:
		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_test = sc.transform(X_test)
		inputX = sc.transform(inputX)


	# Decision Tree Classification
	classifiers.append(DecisionTreeClassifier(criterion = 'entropy', random_state = 0))

	# Random Forest Classification
	classifiers.append(RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0))

	# SVM
	for kernelMode in allKernelModes:
		classifiers.append(SVC(kernel = kernelMode, random_state = 0))

	# K-NN
	for k in allKs:
		classifiers.append(KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2))

	# Naive Bayes
	classifiers.append(GaussianNB())

	# Logistic Regression
	classifiers.append(LogisticRegression(random_state = 0))

	# Fitting to all
	for classifier in classifiers:
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		allResults.append(accuracy_score(y_test,y_pred))

	allPredictions = []

	# only use for predicting those models whose accuracy is >= the threshold set
	for i in range(0,len(classifiers)):
		if allResults[i] >= threshold:
			allPredictions.append(classifiers[i].predict(inputX))

	# the most common prediction from all models is selected as the final one
	if len(allPredictions) > 0:
		finalPred = int(stats.mode(allPredictions)[0][0])
		return finalPred
	else:
		print('! No predictions could be made with threshold ' + str(threshold) + '\n! Returning none.')
		return None

