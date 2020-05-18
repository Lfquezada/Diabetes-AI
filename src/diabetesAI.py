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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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


    label1 = tk.Label(frame,text = ' INFORMATION ',font='Arial 16 bold',fg='#ffffff',bg=appBgColor)
    label1.place(relx=0.135,rely=0.4)

    label2 = tk.Label(frame,text = ' STATS ',font='Arial 16 bold',fg='#ffffff',bg=appBgColor)
    label2.place(relx=0.465,rely=0.4)

    label3 = tk.Label(frame,text = ' PREDICT ',font='Arial 16 bold',fg='#ffffff',bg=appBgColor)
    label3.place(relx=0.755,rely=0.4)


    button1 = tk.Button(frame,image=infoIcon,pady=0, padx=0, borderwidth=0, highlightthickness=0,command=lambda: redirect('info'))
    button1.place(relx=0.1,rely=0.45)

    button2 = tk.Button(frame,image=statsIcon,pady=0, padx=0, borderwidth=0, highlightthickness=0,command=lambda: redirect('stats'))
    button2.place(relx=0.4,rely=0.45)

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

    canvas = tk.Canvas(root,height=700,width=1300,bg=appBgColor)
    canvas.pack()
    frame = tk.Frame(root,bg=appBgColor)
    frame.place(relx=0.05,rely=0.15,relwidth=0.9,relheight=0.8)

    goBackButton = tk.Button(canvas,image=backIcon,pady=0, padx=0, borderwidth=0, highlightthickness=0,command=lambda: mainApp(reload=True))
    goBackButton.place(relx=0.05,rely=0.07)

    df = pd.read_csv('diabetes.csv')
    figure1 = plt.Figure(figsize=(4,1), dpi=100)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, frame)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    df1 = df[['Age','Outcome']].groupby('Age').sum()
    df1.plot(kind='bar', legend=True, ax=ax1)
    ax1.set_title('Diabetes occurrence by Age')

    figure2 = plt.Figure(figsize=(4,1), dpi=100)
    ax2 = figure2.add_subplot(111)
    ax2.scatter(df['SkinThickness'],df['BMI'], color =appBgColor, s=10, edgecolors='#26445e', linewidth=0.5)
    scatter2 = FigureCanvasTkAgg(figure2, frame) 
    scatter2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    ax2.set_xlabel('SkinThickness (mm)')
    ax2.set_ylabel('BMI')
    ax2.set_title('SkinThickness Vs. BMI')

    figure3 = plt.Figure(figsize=(4,1), dpi=100)
    ax3 = figure3.add_subplot(111)
    ax3.scatter(df['Glucose'],df['BMI'], color =appBgColor, s=10, edgecolors='#26445e', linewidth=0.5)
    scatter3 = FigureCanvasTkAgg(figure3, frame) 
    scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    ax3.set_xlabel('Glucose')
    ax3.set_ylabel('BMI')
    ax3.set_title('Glucose Vs. BMI')


def predictPage():
    root.title('Predict')

    global canvas, frame
    canvas.destroy()
    frame.destroy()

    canvas = tk.Canvas(root,height=500,width=1000,bg=appBgColor)
    canvas.pack()
    frame = tk.Frame(root,bg=appBgColor)
    frame.place(relx=0,rely=0,relwidth=1,relheight=1)

    goBackButton = tk.Button(frame,image=backIcon,pady=0, padx=0, borderwidth=0, highlightthickness=0,command=lambda: mainApp(reload=True))
    goBackButton.place(relx=0.05,rely=0.07)

    col1 = 0.25
    col2 = 0.38

    pregnanciesEntry = tk.Entry(frame,fg=grey,bg='#ffffff',justify='center')
    pregnanciesEntry.place(relx=col1,rely=0.15,width=100)
    label1 = tk.Label(frame,text = 'Pregnancies',font='Arial 14 bold',fg='#ffffff',bg=appBgColor)
    label1.place(relx=col2,rely=0.155)

    glucoseEntry = tk.Entry(frame,fg=grey,bg='#ffffff',justify='center')
    glucoseEntry.place(relx=col1,rely=0.25,width=100)
    label2 = tk.Label(frame,text = 'Glucose',font='Arial 14 bold',fg='#ffffff',bg=appBgColor)
    label2.place(relx=col2,rely=0.255)

    bloodEntry = tk.Entry(frame,fg=grey,bg='#ffffff',justify='center')
    bloodEntry.place(relx=col1,rely=0.35,width=100)
    label3 = tk.Label(frame,text = 'Blood Pressure (mmHg)',font='Arial 14 bold',fg='#ffffff',bg=appBgColor)
    label3.place(relx=col2,rely=0.355)

    skinEntry = tk.Entry(frame,fg=grey,bg='#ffffff',justify='center')
    skinEntry.place(relx=col1,rely=0.45,width=100)
    label4 = tk.Label(frame,text = 'Skin Thickness (mm)',font='Arial 14 bold',fg='#ffffff',bg=appBgColor)
    label4.place(relx=col2,rely=0.455)

    insulinEntry = tk.Entry(frame,fg=grey,bg='#ffffff',justify='center')
    insulinEntry.place(relx=col1,rely=0.55,width=100)
    label5 = tk.Label(frame,text = 'Insulin (mu U/ml)',font='Arial 14 bold',fg='#ffffff',bg=appBgColor)
    label5.place(relx=col2,rely=0.555)

    bmiEntry = tk.Entry(frame,fg=grey,bg='#ffffff',justify='center')
    bmiEntry.place(relx=col1,rely=0.65,width=100)
    label6 = tk.Label(frame,text = 'Body Mass Index',font='Arial 14 bold',fg='#ffffff',bg=appBgColor)
    label6.place(relx=col2,rely=0.655)

    ageEntry = tk.Entry(frame,fg=grey,bg='#ffffff',justify='center')
    ageEntry.place(relx=col1,rely=0.75,width=100)
    label7 = tk.Label(frame,text = 'Age (yrs)',font='Arial 14 bold',fg='#ffffff',bg=appBgColor)
    label7.place(relx=col2,rely=0.755)

    global predLabel
    predLabel = tk.Label(frame,image=squareIcon,fg='#ffffff',bg=appBgColor)
    predLabel.place(relx=0.7,rely=0.4)

    predictButton = tk.Button(frame,text='Predict',width=30,height=1,command=lambda: showPred([pregnanciesEntry.get(),glucoseEntry.get(),bloodEntry.get(),skinEntry.get(),insulinEntry.get(),bmiEntry.get(),ageEntry.get()]))
    predictButton.place(relx=col1,rely=0.85)


def showPred(userInput):

    allInputNotNull = True

    for i in userInput:
        if not isInt(i):
            allInputNotNull = False
            break

    if allInputNotNull:
        if classifier.predict(userInput):
            predLabel['image'] = positiveIcon
        else:
            predLabel['image'] = negativeIcon
    else:
        predLabel['image'] = squareIcon


def isInt(v):
    try:
        x = int(v)
        return True
    except ValueError:
        return False


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
backIcon = tk.PhotoImage(file='assets/icon-arrow.png')

squareIcon = tk.PhotoImage(file='assets/icon-square.png')
positiveIcon = tk.PhotoImage(file='assets/icon-positive.png')
negativeIcon = tk.PhotoImage(file='assets/icon-negative.png')

# presets
appBgColor = '#40739e'
lightLetterColor = '#3d3d3d'
grey = '#363636'

# Generate Model
classifier = DiabetesClassifier(0.1)
#print(classifier.getTestData())

# run home screen
mainApp(reload=False)
root.mainloop()




