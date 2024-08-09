import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import filedialog, ttk
import os
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap  


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler    
from sklearn.ensemble import RandomForestClassifier  

root = tk.Tk()
root.title('App version 1.3.6')

df = None  # To store the dataframe
my_ref = {}  # to store references to checkboxes
i = 1
selected_checkboxes = []  # To store the checkbuttons which are checked
data_types = ["int64", "float64", "object"]
labels = []

# Data functions
# def show_info(df):
#     info_text.delete(1.0, tk.END)  # Xóa nội dung cũ trong widget Text
#     for col in df.columns:
#         info_text.insert(tk.END, f"Column: {col}\n")
#         info_text.insert(tk.END, f"Null values: {df[col].isnull().sum()}\n")
#         info_text.insert(tk.END, f"Unique values: {df[col].nunique()}\n")
#         info_text.insert(tk.END, f"Duplicate values: {df.duplicated().sum()}\n")
#         if df[col].dtype != "object":
#             outline_range = f"{df[col].quantile(0.25)} - {df[col].quantile(0.75)}"
#             info_text.insert(tk.END, f"Outline values: {outline_range}\n")
#         info_text.insert(tk.END, "\n")
        
def upload_file():
    global df, tree_list
    f_types = [('CSV files', "*.csv"), ('All', "*.*")]
    file = filedialog.askopenfilename(filetypes=f_types)
    file_name = os.path.basename(file)
    path_label.config(text=file_name)
    df = pd.read_csv(file)
    tree_list = list(df)  # List of column names as header
    str1 = "Rows:" + str(df.shape[0]) + " , Columns:" + str(df.shape[1])
    count_label.config(text=str1)
    trv_refresh()  # show Treeview
    search_entry.bind('<KeyRelease>', lambda event: my_search())
    target_combobox["values"] = tree_list
    my_columns()

    # Transform tab
    for dtype in data_types:  # Loop through data types
        # Lọc các cột theo định dạng dữ liệu dtype
        columns = [col for col in df.columns if df[col].dtype == dtype]
        # Tạo danh sách tên cột cùng với định dạng dữ liệu của chúng
        columns_with_dtype = [f"{col} {{{dtype}}}" for col in columns]
        # Thêm danh sách cột vào Listbox
        # for col in columns_with_dtype:
        #     transform_list.insert(tk.END, col)

def my_search():
    # Lấy giá trị từ Entry và chuyển về chữ thường
    query = search_entry.get().strip().lower()
    if query:  # Kiểm tra xem giá trị được nhập vào hay không
        keywords = query.split()  # Tách các từ khóa
        # Tạo điều kiện tìm kiếm
        condition = df.apply(lambda row: all(keyword.lower() in str(
            row).lower() for keyword in keywords), axis=1)
        df2 = df[condition]  # Lọc các hàng thỏa mãn điều kiện tìm kiếm
        r_set = df2.to_numpy().tolist()  # Tạo danh sách các hàng kết quả

        # Cập nhật Treeview để hiển thị kết quả tìm kiếm
        trv_refresh(r_set)
    else:
        # Nếu không có giá trị tìm kiếm được nhập vào, hiển thị toàn bộ dữ liệu
        trv_refresh()

def trv_refresh(r_set=None):  # Refresh the Treeview to reflect changes
    global df, trv, tree_list
    if r_set is None:
        r_set = df.to_numpy().tolist()  # create list of list using rows

    if hasattr(root, 'trv'):
        root.trv.destroy()

    trv = ttk.Treeview(dataTab, selectmode='browse', height=10,
                       show='headings', columns=tree_list)
    trv.grid(row=5, column=1, columnspan=3, padx=10, pady=20)

    for i in tree_list:
        trv.column(i, width=90, anchor='c')
        trv.heading(i, text=str(i))

    for dt in r_set:
        v = [r for r in dt]
        # Kiểm tra nếu item chưa tồn tại trong Treeview trước khi chèn
        if not trv.exists(v[0]):
            trv.insert("", 'end', iid=v[0], values=v)

    vs = ttk.Scrollbar(dataTab, orient="vertical", command=trv.yview)
    trv.configure(yscrollcommand=vs.set)  # connect to Treeview
    vs.grid(row=5, column=4, sticky='ns')  # Place on grid

# Model functions
def my_columns():
    global i, my_ref, selected_checkboxes
    i = 1  # to increase the column number
    my_ref = {}  # to store references to checkboxes
    selected_checkboxes = []  # Initialize the list of selected checkboxes
    input_label_cb.config(text=" ")  # Remove the previous checkboxes
    for column in tree_list:
        var = IntVar()
        cb = Checkbutton(modelTab, text=column, variable=var)
        cb.grid(row=i + 3, column=0, padx=5, sticky=tk.W)
        my_ref[column] = var
        i += 1
        # Append checkbox and its variable to the list
        selected_checkboxes.append((column, var))

def execute_model():
    global model_train  # Di chuyển câu lệnh global lên đầu hàm
    target_variable = target_combobox.get()
    input_variables = [column for column, var in selected_checkboxes if var.get() == 1]
    le = LabelEncoder()

    # if input variable is categorical convert to numerical
    for column in input_variables:
        if df[column].dtype == "object":
            df[column] = le.fit_transform(df[column])
    if df[target_variable].dtype == "object":
        df[target_variable] = le.fit_transform(df[target_variable])
    
    # convert to list
    input_variables = list(input_variables)
    # delete item in list empty
    input_variables = [x for x in input_variables if x != ""]

    print(target_variable)
    print(input_variables)
    X = df[input_variables]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    
    #future scaling
    st_x= StandardScaler()    
    X_train= st_x.fit_transform(X_train)    
    X_test= st_x.transform(X_test)  

    model = model_combobox.get()

    print("Executing model...")
    if model == "Logistic Regression":
        model_train = LogisticRegression()
        print("Logistic Regression")
    elif model == "KNN":
        model_train = KNeighborsClassifier()
        print("KNN")
    elif model == "Linear Regression":
        model_train = LinearRegression()
        print("Linear Regression")
    elif model == "Random Forest":
        model_train = RandomForestClassifier(n_estimators= 10, criterion="entropy")  
        print("Random Forest")

    
    model_train.fit(X_train, y_train)
    y_pred = model_train.predict(X_test)
    print("y_pred:", y_pred)

    if isinstance(model_train, LogisticRegression) or isinstance(model_train, KNeighborsClassifier):
        try:
            accuracy = accuracy_score(y_test, y_pred)
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.text(0.5, 0.95, f"Accuracy: {accuracy}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
            plt.show()
        except Exception as e:
            print("Error:", e)
    elif isinstance(model_train, LinearRegression):
        # Calculate R-squared
        r2 = r2_score(y_test, y_pred)
        # Scatter plot
        plt.scatter(y_test, y_pred)
        plt.plot(y_test, y_test, color='red', linewidth=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Scatter plot")
        plt.text(0.5, 0.95, f"R-squared: {r2}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
        plt.show()
    elif isinstance(model_train, RandomForestClassifier):
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        X_set, y_set = X_test, y_test  
        x1, x2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step  =0.01), 
                            np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(x1, x2, model_train.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
                     alpha = 0.75, cmap = ListedColormap(('purple', 'green')))
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],  
                        c = ListedColormap(('purple', 'green'))(i), label = j)
        plt.title('Random Forest Classification')
        plt.xlabel('Independent ')
        plt.ylabel('Dependent')
        plt.legend()
        plt.text(0.5, 0.95, f"Accuracy: {accuracy}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
        plt.show()

# Visualize functions
def selected_vs_type(event):
    quantitative_columns = [column for column in tree_list if df[column].dtype in ['int64', 'float64']]
    categorical_columns = [column for column in tree_list if df[column].dtype == 'object']
    graph_type_1 = ["Histogram", "Box Plot"]
    graph_type_2 = ["Bar Chart", "Pie Chart"]
    graph_type_3 = ["Stacked Bar Chart", "Heatmap"]
    graph_type_4 = ["Bar Chart", "Violin Plot"]
    graph_type_5 = ["Line Plot", "Scatter Plot"]
      
    selected = vs_type_combobox.get()
    if selected == "1 Quantitative":
        column1_label.config(text="Quantitative")
        column1_combobox["values"] = quantitative_columns
        
        column2_combobox["values"] = []
        column2_label.grid_remove()
        column2_combobox.grid_remove()
        
        vs_graph_type_combobox["values"] = graph_type_1
    elif selected == "1 Categorical":
        column1_label.config(text="Categorical")
        column1_combobox["values"] = categorical_columns
        
        column2_combobox["values"] = []
        column2_label.config(text="")
        column2_combobox.grid_remove()
        column2_combobox.grid_remove()
        vs_graph_type_combobox["values"] = graph_type_2
    elif selected == "2 Categorical":
        column1_label.config(text="Categorical 1")
        column2_label.config(text="Categorical 2")
        
        column1_combobox["values"] = categorical_columns
        column2_combobox["values"] = categorical_columns
        
        column2_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        column2_combobox.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        vs_graph_type_combobox["values"] = graph_type_3
    elif selected == "1 Categorical - 1 Quantitative":
        column1_label.config(text="Categorical")
        column2_label.config(text="Quantitative")
        
        column1_combobox["values"] = categorical_columns
        column2_combobox["values"] = quantitative_columns
        
        column2_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        column2_combobox.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        vs_graph_type_combobox["values"] = graph_type_4
    elif selected == "2 Quantitative":
        column1_label.config(text="Quantitative 1")
        column2_label.config(text="Quantitative 2")
        
        column1_combobox["values"] = quantitative_columns
        column2_combobox["values"] = quantitative_columns
        
        column2_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        column2_combobox.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        vs_graph_type_combobox["values"] = graph_type_5
        
def bar_chart_1_column(column):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column)
    plt.title(f"Bar chart for {column}")
    plt.show()
    
def histogram_1_column(column):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f"Histogram for {column}")
    plt.show()
    
def pie_chart_1_column(column):
    plt.figure(figsize=(10, 6))
    df[column].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f"Pie chart for {column}")
    plt.show()
    
def box_plot_1_column(column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y=column)
    plt.title(f"Box plot for {column}")
    plt.show()
    
def stacked_bar_chart_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    df.groupby(column1)[column2].value_counts().unstack().plot(kind='bar', stacked=True)
    plt.title(f"Stacked bar chart for {column1} and {column2}")
    plt.show()
    
def heatmap_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.heatmap(pd.crosstab(df[column1], df[column2]), annot=True, fmt='d')
    plt.title(f"Heatmap for {column1} and {column2}")
    plt.show()

def box_plot_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=column1, y=column2)
    plt.title(f"Box plot for {column1} and {column2}")
    plt.show()
    
def violin_plot_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=column1, y=column2)
    plt.title(f"Violin plot for {column1} and {column2}")
    plt.show()
    
def line_plot_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=column1, y=column2)
    plt.title(f"Line plot for {column1} and {column2}")
    plt.show()    
    
def bar_chart_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column1, hue=column2)
    plt.title(f"Bar chart for {column1} and {column2}")
    plt.show()
    
def scatter_plot_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=column1, y=column2)
    plt.title(f"Scatter plot for {column1} and {column2}")
    plt.show()
    
def box_plot_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=column1, y=column2)
    plt.title(f"Box plot for {column1} and {column2}")
    plt.show()
    
def execute_visualize():
    vs_type = vs_type_combobox.get()
    column1 = column1_combobox.get()
    column2 = column2_combobox.get()
    graph_type = vs_graph_type_combobox.get()
    
    if vs_type == "1 Quantitative":
        if graph_type == "Histogram":
            histogram_1_column(column1)
        elif graph_type == "Box Plot":
            box_plot_1_column(column1)
    elif vs_type == "1 Categorical":
        if graph_type == "Bar Chart":
            bar_chart_1_column(column1)
        elif graph_type == "Pie Chart":
            pie_chart_1_column(column1)
    elif vs_type == "2 Categorical":
        if graph_type == "Stacked Bar Chart":
            stacked_bar_chart_2_columns(column1, column2)
        elif graph_type == "Heatmap":
            heatmap_2_columns(column1, column2)
    elif vs_type == "1 Categorical - 1 Quantitative":
        if graph_type == "Bar Chart":
            bar_chart_2_columns(column1, column2)
        elif graph_type == "Violin Plot":
            violin_plot_2_columns(column1, column2)
    else:
        if graph_type == "Line Plot":
            line_plot_2_columns(column1, column2)
        elif graph_type == "Scatter Plot":
            scatter_plot_2_columns(column1, column2)
            
# Transform functions
# def excute_type():
#     try:
#         selected_index = transform_list.curselection()
#         if not selected_index:
#             messagebox.showerror("Error", "No column selected. Please select a column from the list.")
#             return
        
#         selected_column = transform_list.get(selected_index)
#         column, dtype = selected_column.split(" {")
#         dtype = dtype[:-1]
#         new_dtype = data_types_combobox.get()
        
#         if dtype == new_dtype:
#             messagebox.showinfo("Information", f"Column {column} is already {dtype}")
#             return
        
#         if new_dtype == "int32":
#             df[column] = df[column].astype(int)
#         elif new_dtype == "float64":
#             df[column] = df[column].astype(float)
#         elif new_dtype == "object":
#             df[column] = df[column].astype(str) 
#         df_clean_label.config(text="Data status: modified")
#         print(f"Column {column} has been changed to {new_dtype}")
        
#         # Update Listbox
#         # transform_list.delete(selected_index)
#         # transform_list.insert(tk.END, f"{column} {{{new_dtype}}}")
        
#         # Print dataframe type
#         print(df[column].dtype)
        
#     except Exception as e:
#         messagebox.showerror("Error", str(e))  
    
# GUI
left_frame = tk.LabelFrame(root, text='Choose File')
left_frame.grid(row=0, column=0, padx=10, pady=10)

right_frame = tk.LabelFrame(root)
right_frame.grid(row=0, column=1, padx=10, pady=10)

tabControl = ttk.Notebook(right_frame)
dataTab = ttk.Frame(tabControl)
modelTab = ttk.Frame(tabControl)
visualizeTab = ttk.Frame(tabControl)
cnnTab = ttk.Frame(tabControl)
tabControl.add(dataTab, text='Data')
tabControl.add(modelTab, text='Model')
tabControl.add(visualizeTab, text='Visualize')
#tabControl.add(transformTab, text='Transform')
tabControl.grid(row=0, column=0, columnspan=2)

# Data tab
my_font1 = ('times', 12, 'bold')
path_label = tk.Label(left_frame, text='Read File & create DataFrame',
                      width=30, font=my_font1)
path_label.grid(row=1, column=1)
browse_btn = tk.Button(left_frame, text='Browse File',
                       width=20, command=lambda: upload_file())
browse_btn.grid(row=2, column=1, pady=5)
df_clean_label = tk.Label(left_frame, text='Data status: unknown')
df_clean_label.grid(row=3, column=1, pady=5)

count_label = tk.Label(dataTab, width=40, text='',
                       bg='lightyellow')
count_label.grid(row=3, column=1, padx=5)
search_entry = tk.Entry(dataTab, width=35, font=18)  # added one Entry box
search_entry.grid(row=4, column=1, padx=1)

# Model tab
target_label = tk.Label(modelTab, text="Select Target Variable")
target_label.grid(row=0, column=0, padx=5, sticky=tk.W)

target_combobox = ttk.Combobox(modelTab)
target_combobox.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

input_label = tk.Label(modelTab, text="Select Input Variables")
input_label.grid(row=2, column=0, padx=5, sticky=tk.W)

input_label_cb = tk.Label(modelTab)
input_label_cb.grid(row=3, column=0, padx=5, sticky=tk.W)

model_label = tk.Label(modelTab, text="Choose Model")
model_label.grid(row=0, column=3, padx=50, pady=10, sticky=tk.W)

model_combobox = ttk.Combobox(
    modelTab, values=["Logistic Regression", "KNN", "Linear Regression", "Random Forest"]
)
model_combobox.grid(row=1, column=3, padx=50, sticky=tk.W)

execution_button = tk.Button(modelTab, text="Execution", command=execute_model)
execution_button.grid(row=2, column=3, padx=50, pady=10, sticky=tk.W)

# Visualize tab
visualize_title = tk.Label(visualizeTab, text="Visualize Type")
visualize_title.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
vs_type_combobox = ttk.Combobox(visualizeTab, values= ["1 Quantitative", "1 Categorical", "2 Categorical", 
                                                       "1 Categorical - 1 Quantitative", "2 Quantitative"], width=40)
vs_type_combobox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
vs_type_combobox.bind("<<ComboboxSelected>>", selected_vs_type)

column1_label = tk.Label(visualizeTab, text="Column 1")
column1_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

column2_label = tk.Label(visualizeTab, text="Column 2")
column2_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

column1_combobox = ttk.Combobox(visualizeTab)
column1_combobox.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

column2_combobox = ttk.Combobox(visualizeTab)
column2_combobox.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

vs_graph_type_label = tk.Label(visualizeTab, text="Graph Type")
vs_graph_type_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)

vs_graph_type_combobox = ttk.Combobox(visualizeTab)
vs_graph_type_combobox.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

excute_btn = tk.Button(visualizeTab, text="Execute", command=execute_visualize)
excute_btn.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

# Transform tab
# transform_label = tk.Label(transformTab, text="Select variables(s): ")
# transform_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
# transform_list = tk.Listbox(transformTab, height=5, selectmode=tk.SINGLE)
# transform_list.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W + tk.E + tk.N + tk.S)

# data_types_label = tk.Label(transformTab, text="Data types: ")
# data_types_label.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

# data_types_combobox = ttk.Combobox(transformTab, values=data_types)
# data_types_combobox.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W + tk.E + tk.N + tk.S)

# excute_data_transform = tk.Button(transformTab, text="Execute type", command=excute_type)
# excute_data_transform.grid(row=1, column=3, padx=10, pady=5, sticky=tk.W)

# data_clean_label = tk.Label(transformTab, text="Data status: unknown")
# data_clean_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

# info_text = scrolledtext.ScrolledText(transformTab, width=60, height=15, wrap=tk.WORD)
# info_text.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky=tk.W)
# info_text.config(state=tk.DISABLED)  # Khóa widget Text để người dùng không thể chỉnh sửa

root.mainloop()  # Keep the window open