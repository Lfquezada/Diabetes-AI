# Diabetes AI

<div align="center">
	Diagnosing *diabetes mellitus* based on patient's conditions with *machine learning*.
</div>


<p align="center">
  <img src="https://github.com/Lfquezada/Diabetes-Diagnosis/blob/master/src/assets/DDAI-animations.gif" width="500">
</p>

Diabetes is a disease that occurs when your blood glucose is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin helps glucose, from food, get into your cells to be used for energy. Sometimes your body doesn’t make enough insulin or doesn’t use insulin well. Glucose then stays in your blood and doesn’t reach your cells. 

Thus, a machine learning approach to diagnose *diabetes mellitus* was made. DD AI provides a friendly, yet profesional, user interface designed to aid health centers/clinics.


## Inputs

Input | Description
------------ | -------------
Pregnancies | Number of times pregnant
Glucose | Plasma glucose concentration in an oral glucose tolerance test
Blood Pressure | Diastolic blood pressure (mm Hg)
Skin Thickness | Triceps skin fold thickness (mm)
Insulin | 2-Hour serum insulin (mu U/ml)
BMI | Body mass index
Age | Age (years)

## Algorithm
Based on preliminary data analisis made with [Klas](https://github.com/Lfquezada/Klas-Classifier), KNN (k=13) and SVM (Linear) machine learning algorithms were selected, as they demonstrated to fit best for the dataset. Aditionaly, an artificial neural network was trained. Based on these 3 models, DiabetesClassifier was created as an ensamble learning algorithm to predict whether a patient has diabetes.


## Requirements
* python 3.x
* Numpy
* Pandas
* Matplotlib
* Sklearn
* Keras
* TensorFlow

## Usage
```
python3 diabetesAI.py
```

