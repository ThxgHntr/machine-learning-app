from tkinter import messagebox
from tkinter.ttk import Treeview
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix  
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.ttk import Combobox
import seaborn as sns
import numpy as np
import pandas as pd
import os

window = Tk()
window.title('ML App')
window.maxsize(900, 600)

df = None

def printLinearRegressionResults(accuracy, X_test, y_test, y_pred):
    plt.scatter(X_test, y_test, color ='b')
    plt.plot(X_test, y_pred, color ='k')
    plt.title('Linear Regression')
    plt.xlabel(combo.get())
    plt.ylabel(combo1.get())
    plt.text(0.5, 0.95, f"Accuracy: {accuracy}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
    plt.show()

def printLogisticRegressionResults(accuracy, cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.text(0.5, 0.95, f"Accuracy: {accuracy}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
    plt.show()

def printKNNResults(accuracy, cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.text(0.5, 0.95, f"Accuracy: {accuracy}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
    plt.show()

def printResults():
    selected_model = combo2.get()
    
    if selected_model == 'Linear Regression':
        model = LinearRegression()
    elif selected_model == 'Logistic Regression':
        model = LogisticRegression()
    elif selected_model == 'KNN':
        model = KNeighborsRegressor(n_neighbors=5)
    else:
        messagebox.showerror("Error", "Invalid model selection")
        return
    X = df.iloc[:, [df.columns.get_loc(x) for x in listbox.get(0, END)]].values
    y = df.iloc[:, df.columns.get_loc(combo.get())].values
    
    for i in range(X.shape[1]):
        if isinstance(X[0][i], str):
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i])
    
    print(X)
    print(y)
    
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st_x= StandardScaler()    
    X_train= st_x.fit_transform(X_train)
    X_test= st_x.transform(X_test) 

    if selected_model == 'Linear Regression':
        classifier= LinearRegression()
        classifier.fit(X_train, y_train)
        y_pred= classifier.predict(X_test)
        accuracy = classifier.score(X_test, y_test)
        printLinearRegressionResults(accuracy, X_test, y_test, y_pred)
        
    elif selected_model == 'Logistic Regression':
        classifier= LogisticRegression(random_state=0)  
        classifier.fit(X_train, y_train)
        y_pred= classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)  
        accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
        printLogisticRegressionResults(accuracy, cm)
    elif selected_model == 'KNN':
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = knn.score(X_test, y_test)
        cm = confusion_matrix(y_test, y_pred)
        printKNNResults(accuracy, cm)

def getDataset():
    global df
    file_path = filedialog.askopenfilename()
    file_name = os.path.basename(file_path)
    cvs_name.config(state='normal')
    cvs_name.delete(0, END)
    cvs_name.insert(0, file_name)
    cvs_name.config(state='disabled')

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
        
    all_columns = df.columns.tolist()

    combo['values'] = all_columns
    combo1['values'] = all_columns

def addColumn():
    selected_column = combo1.get()
    listbox.insert(END, selected_column)
    combo1['values'] = [x for x in combo1['values'] if x != selected_column]

def removeColumn():
    selected_index = listbox.curselection()
    if selected_index:
        selected_column = listbox.get(selected_index)
        listbox.delete(selected_index)
        combo1_values = list(combo1['values'])
        combo1_values.append(selected_column)
        combo1['values'] = tuple(combo1_values)
        
left_frame = Frame(window, width=200, height=400, bg='white')
left_frame.grid(row=0, column=0, padx=10, pady=5)

right_frame = Frame(window, width=650, height=400, bg='white')
right_frame.grid(row=0, column=1, padx=10, pady=5)

lbl_dataset = Label(left_frame, text="Dataset:", font=("Arial Bold", 10))
lbl_dataset.grid(column=0, row=0, padx=10, pady=10)

cvs_name = Entry(left_frame, width=40, state='disabled')
cvs_name.grid(column=1, row=0, columnspan=2, padx=10, pady=10)

btn_getCSV = Button(left_frame, text="Browse", bg="orange", fg="white", command=getDataset)
btn_getCSV.grid(column=3, row=0, padx=10, pady=10)    

lbl_iv = Label(left_frame, text="Dependent Variable:", font=("Arial Bold", 10))
lbl_iv.grid(column=0, row=1, padx=10, pady=10)

combo = Combobox(left_frame)
combo.grid(column=1, row=1, padx=10, pady=10)

lbl_dv = Label(left_frame, text="Independent Variable:", font=("Arial Bold", 10))
lbl_dv.grid(column=0, row=2, padx=10, pady=10)

combo1 = Combobox(left_frame)
combo1.grid(column=1, row=2, padx=10, pady=10)

btn_add = Button(left_frame, text="Add", bg="green", fg="white", command=addColumn)
btn_add.grid(column=2, row=2, padx=10, pady=10)
 
lbl_select = Label(left_frame, text="Select DV:", font=("Arial Bold", 10))
lbl_select.grid(column=0, row=3, padx=10, pady=10)

listbox = Listbox(left_frame, height=5, width=40)
listbox.grid(column=1, row=3, padx=10, pady=10)

btn_remove = Button(left_frame, text="Remove", bg="red", fg="white", command=removeColumn)
btn_remove.grid(column=3, row=3, padx=10, pady=10)

combo2 = Combobox(left_frame)
combo2['values'] = ['Linear Regression', 'Logistic Regression', 'KNN']
combo2.grid(column=1, row=4, padx=10, pady=10)

btn_print = Button(left_frame, text="Print", bg="blue", fg="white", command=printResults)
btn_print.grid(column=2, row=4, padx=10, pady=10)

window.mainloop()