import csv
import tkinter as tk
from tkinter import Button, Label, filedialog, messagebox, ttk

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

csv_columns = []

def load_csv_file():
    global csv_columns
    global csv_dt
    file_path = filedialog.askopenfilename()
    if file_path.endswith(".csv"):
        csv_columns = []
        csv_dt = pd.read_csv(file_path)
        csv_columns = csv_dt.columns.to_list()
        # clean data
        csv_dt.dropna(inplace=True)
        csv_dt.reset_index(drop=True, inplace=True)
    elif file_path.endswith((".xls", ".xlsx")):
        csv_columns = []
        csv_dt = pd.read_excel(file_path)
        csv_columns = csv_dt.columns.tolist()
        # clean data
        csv_dt.dropna(inplace=True)
        csv_dt.reset_index(drop=True, inplace=True)
    else:
        messagebox.showerror("Error", "Invalid file type")
    target_combobox["values"] = csv_columns
    update_input_listbox()
    
    # Update dataset tree
    dataset_tree.delete(*dataset_tree.get_children())
    dataset_tree["column"] = list(csv_dt.columns)
    dataset_tree["show"] = "headings"
    str1="Rows: " + str(csv_dt.shape[0])+ " , Columns: "+str(csv_dt.shape[1])
    for column in dataset_tree["column"]:
        dataset_tree.heading(column, text=column)
    for index, row in csv_dt.iterrows():
        dataset_tree.insert("", "end", values=list(row))
    dataset_tree.grid(row=0, column=0, sticky="nsew")
    label_count_cr.config(text=str1)
    
def drop_column():
    selected_item = dataset_tree.selection()
    if selected_item:
        column_name = dataset_tree.item(selected_item)["values"][0]
        # Xóa cột từ dataset_tree và dữ liệu
        dataset_tree.delete(selected_item)
        # Lưu thông tin về việc xóa cột để có thể hoàn tác

def undo_drop_column(event=None):
    # Code để hoàn tác việc drop column
    pass

def update_input_listbox():
    input_listbox.delete(0, tk.END)
    for column in csv_columns:
        if column != target_combobox.get():
            input_listbox.insert(tk.END, column)


def add_variable():
    selected_indices = input_listbox.curselection()
    for index in selected_indices:
        selected_variable = input_listbox.get(index)
        selected_listbox.insert(tk.END, selected_variable)
        input_listbox.delete(index)
    input_listbox.selection_clear(0, tk.END)


def remove_variable():
    selected_indices = selected_listbox.curselection()
    for index in selected_indices:
        removed_variable = selected_listbox.get(index)
        input_listbox.insert(tk.END, removed_variable)
        selected_listbox.delete(index)
    selected_listbox.selection_clear(0, tk.END)


def execute_model():
    target_variable = target_combobox.get()
    input_variables = selected_listbox.get(0, tk.END)
    le = LabelEncoder()

    # if input variable is categorical convert to numberical
    for column in input_variables:
        if csv_dt[column].dtype == "object":
            csv_dt[column] = le.fit_transform(csv_dt[column])
    if csv_dt[target_variable].dtype == "object":
        csv_dt[target_variable] = le.fit_transform(csv_dt[target_variable])
    # convert to list
    input_variables = list(input_variables)
    # delete item in list empty
    input_variables = [x for x in input_variables if x != ""]

    print(target_variable)
    print(input_variables)
    X = csv_dt[input_variables]
    y = csv_dt[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = model_combobox.get()

    global model_train

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
    model_train.fit(X_train, y_train)
    y_pred = model_train.predict(X_test)

    if isinstance(model_train, LogisticRegression) or isinstance(
        model_train, KNeighborsClassifier
    ):
        try:
            accuracy = accuracy_score(y_test, y_pred)
            messagebox.showinfo("Model Result", f"Accuracy: {accuracy}")
        except:
            print("error")

    elif isinstance(model_train, LinearRegression):
        r2= str(r2_score(y_test, y_pred))
        messagebox.showinfo("Model Result", f"Mean Squared Error: {r2}")

    # Thực hiện xử lý với các thông tin đã chọn


# Tạo cửa sổ giao diện
window = tk.Tk()
window.title("Giao diện")

# Tạo frame trái phải chứa các phần tử giao diện
left_frame = tk.LabelFrame(window, text="Chức năng")
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Tạo frame phải chứa các phần tử giao diện và nằm ở trên cùng của window
right_frame = tk.LabelFrame(window, text="Dataset")
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Tạo nút "Load CSV File"
load_button = tk.Button(left_frame, text="Load CSV File", command=load_csv_file)
load_button.grid(row=0, column=0, padx=5)

# Tạo nhãn "Select Target Variable"
target_label = tk.Label(left_frame, text="Select Target Variable")
target_label.grid(row=1, column=0, padx=5, sticky=tk.W)

# Tạo combobox để chọn Target Variable
target_combobox = ttk.Combobox(left_frame)
target_combobox.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

# Tạo nhãn "Select Input Variables"
input_label = tk.Label(left_frame, text="Select Input Variables")
input_label.grid(row=3, column=0, padx=5, sticky=tk.W)

# Tạo scrollbar cho danh sách các Input Variables
input_scrollbar = tk.Scrollbar(left_frame)
input_scrollbar.grid(row=4, column=1, sticky=tk.N + tk.S)

# Tạo danh sách các Input Variables
input_listbox = tk.Listbox(
    left_frame, height=5, yscrollcommand=input_scrollbar.set, selectmode=tk.MULTIPLE
)
input_listbox.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W + tk.E + tk.N + tk.S)

# Kết nối scrollbar với danh sách các Input Variables
input_scrollbar.config(command=input_listbox.yview)

# Tạo nút "Add"
add_button = tk.Button(left_frame, text="Add", command=add_variable)
add_button.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)

# Tạo nhãn "Selected Input Variables"
selected_label = tk.Label(left_frame, text="Selected Input Variables")
selected_label.grid(row=6, column=0, padx=5, sticky=tk.W)

# Tạo scrollbar cho danh sách các Selected Input Variables
selected_scrollbar = tk.Scrollbar(left_frame)
selected_scrollbar.grid(row=7, column=1, sticky=tk.N + tk.S)

# Tạo danh sách các Selected Input Variables
selected_listbox = tk.Listbox(left_frame, height=5, yscrollcommand=selected_scrollbar.set)
selected_listbox.grid(row=7, column=0, padx=5, pady=5, sticky=tk.W + tk.E + tk.N + tk.S)

# Kết nối scrollbar với danh sách các Selected Input Variables
selected_scrollbar.config(command=selected_listbox.yview)

# Tạo nút "Remove"
remove_button = tk.Button(left_frame, text="Remove", command=remove_variable)
remove_button.grid(row=8, column=0, padx=5, pady=5, sticky=tk.W)

# Tạo nhãn "Chọn Model"
model_label = tk.Label(left_frame, text="Chọn Model")
model_label.grid(row=9, column=0, padx=5, pady=10, sticky=tk.W)

# Tạo combobox để chọn Model
model_combobox = ttk.Combobox(
    left_frame, values=["Logistic Regression", "KNN", "Linear Regression"]
)
model_combobox.grid(row=10, column=0, padx=5, sticky=tk.W)

# Tạo nút "Execution"
execution_button = tk.Button(left_frame, text="Execution", command=execute_model)
execution_button.grid(row=11, column=0, padx=5, pady=10, sticky=tk.W)

dataset_tree = ttk.Treeview(right_frame)
dataset_tree.grid(row=0, column=0, sticky="nsew")

label_count_cr = tk.Label(right_frame, width=40, text='', bg='lightyellow')
label_count_cr.grid(row=1, column=0, padx=5, pady=10)

# Bắt sự kiện chuột phải để hiển thị menu
def popup(event):
    column_id = dataset_tree.identify_column(event.x)
    column_name = dataset_tree.heading(column_id)["text"]
    menu = tk.Menu(window, tearoff=0)
    menu.add_command(label=f"Drop {column_name}", command=drop_column)
    menu.post(event.x_root, event.y_root)

dataset_tree.bind("<Button-3>", popup)

# Bắt sự kiện "Ctrl + Z" để hoàn tác drop column
window.bind("<Control-z>", undo_drop_column)

# Chạy giao diện
window.mainloop()
