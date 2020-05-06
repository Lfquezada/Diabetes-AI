'''
------------------------------------------
Universidad del Valle de Guatemala
Minería de Datos
Proyecto Final
				Diabetes AI

Luis Quezada 18028
Jennifer Sandoval 18962
Esteban del Valle 18221
------------------------------------------
'''


import tkinter as tk
from tkinter import ttk
from statistics import mode
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from keras.models import load_model
import webbrowser


class DiabetesClassifier():
    
    def __init__(self,test_size=0.2):
        dataset = pd.read_csv('diabetes.csv')
        dataset.dropna(inplace=True)
        dataset = dataset.drop(columns=['DiabetesPedigreeFunction'])
        X = dataset.iloc[:, [0,1,2,3,4,5,6]].values
        y = dataset.iloc[:, 7].values
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        self.sc = StandardScaler()
        X_train = self.sc.fit_transform(X_train)
        
        self.setupSVM(X_train, y_train)
        self.setupKNN(X_train, y_train)
        self.setupRNA(X_train, y_train)
    
    def getTestData(self):
        return self.X_test,self.y_test
    
    def setupSVM(self,X_train, y_train):
        self.LinearSVM_Classifier = SVC(kernel='linear', random_state=0)
        self.LinearSVM_Classifier.fit(X_train, y_train)
    
    def setupKNN(self,X_train, y_train):
        self.KNN_Classifier = KNeighborsClassifier(n_neighbors=13, metric='minkowski', p=2)
        self.KNN_Classifier.fit(X_train, y_train)
    
    def setupRNA(self,X_train, y_train):
        self.RNA_Classifier = load_model('rna_model.h5')
    
    def predict(self,inputData):
        inputData = [inputData]
        inputData_scaled = self.sc.transform(inputData)

        pred1 = self.LinearSVM_Classifier.predict(inputData_scaled)[0] > 0.5
        pred2 = self.KNN_Classifier.predict(inputData_scaled)[0] > 0.5
        pred3 = self.RNA_Classifier.predict(inputData_scaled)[0][0] > 0.5
        
        return mode([pred1,pred2,pred3])
    
    def getConfusionMatrix(self):
        Y_pred = []
        for x in self.X_test:
            Y_pred.append(1) if classifier.predict(x) else Y_pred.append(0)
        return confusion_matrix(self.y_test, Y_pred)
    
    def getAccuracy(self):
        Y_pred = []
        for x in self.X_test:
            Y_pred.append(1) if classifier.predict(x) else Y_pred.append(0)
        return accuracy_score(self.y_test, Y_pred)


# Home Screen
def mainApp(reload):
    root.title('Diabetes AI')

    global canvas, frame

    if reload:
        canvas.destroy()
        frame.destroy()

    canvas = tk.Canvas(root,height=500,width=1000,bg=appBgColor)
    canvas.pack()
    frame = tk.Frame(root,bg=appBgColor)
    frame.place(relx=0,rely=0,relwidth=1,relheight=1)

    spacerTop = tk.Label(frame,text='',font='Arial 40',bg=appBgColor)
    spacerTop.pack(side='top')
    logoLabel = tk.Label(frame,image=ddaiLogo,pady=0, padx=0, borderwidth=0, highlightthickness=0)
    logoLabel.pack(side='top')

    button1 = tk.Button(frame,image=infoIcon,pady=0, padx=0, borderwidth=0, highlightthickness=0,command=lambda: redirect('info'))
    button1.place(relx=0.1,rely=0.45)

    button2 = tk.Button(frame,image=statsIcon,pady=0, padx=0, borderwidth=0, highlightthickness=0,command=lambda: redirect('stats'))
    button2.place(relx=0.41,rely=0.45)

    button3 = tk.Button(frame,image=predictIcon,pady=0, padx=0, borderwidth=0, highlightthickness=0,command=lambda: redirect('predict'))
    button3.place(relx=0.7,rely=0.45)


def redirect(page):
    if page == 'info':
        webbrowser.open('https://www.who.int/es/news-room/fact-sheets/detail/diabetes')

    elif page == 'stats':
        statsPage()

    elif page == 'predict':
        predictPage()


def statsPage():
    root.title('Stats')

    global canvas, frame
    canvas.destroy()
    frame.destroy()

    canvas = tk.Canvas(root,height=500,width=1000,bg=appBgColor)
    canvas.pack()
    frame = tk.Frame(root,bg=appBgColor)
    frame.place(relx=0,rely=0,relwidth=1,relheight=1)


def predictPage():
    root.title('Predict')

    global canvas, frame
    canvas.destroy()
    frame.destroy()

    canvas = tk.Canvas(root,height=500,width=1000,bg=appBgColor)
    canvas.pack()
    frame = tk.Frame(root,bg=appBgColor)
    frame.place(relx=0,rely=0,relwidth=1,relheight=1)


def predict(userInput):
    #userInput = [1.0, 199.0, 76.0, 43.0, 0.0, 42.9, 22.0]
    return classifier.predict(userInput)


'''
------------------------------------------
				Run App
------------------------------------------
'''

root = tk.Tk()
root.configure(background='black')

# preload assets
ddaiLogo = tk.PhotoImage(file='assets/DDAI.png')
infoIcon = tk.PhotoImage(file='assets/icon-info.png')
statsIcon = tk.PhotoImage(file='assets/icon-stats.png')
predictIcon = tk.PhotoImage(file='assets/icon-predict.png')

# presets
appBgColor = '#40739e'
lightLetterColor = '#3d3d3d'

# Generate Model
classifier = DiabetesClassifier()

# run home screen
mainApp(reload=False)
root.mainloop()




