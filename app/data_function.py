import os
import tkinter as tk1
import re
import numpy as np
import pandas as pd
import seaborn as sns
from tkinter import Checkbutton, IntVar, filedialog, messagebox, simpledialog
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import tree
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

checkbox_widgets = []


def edit_column(column_name):
    from gui import target_combobox
    global df, trv, tree_list
    try:
        # Hiển thị cửa sổ để đổi tên cột được chọn
        new_name = simpledialog.askstring(
            "Edit Column", f"Enter new name for {column_name}")
        if new_name:
            df = df.rename(columns={column_name: new_name})
            tree_list = list(df)
            target_combobox["values"] = tree_list
            trv_refresh()
    except Exception as e:
        print("Error:", e)


def delete_column(column_name):
    from gui import target_combobox
    global df, trv, tree_list
    try:
        # Hiển thị cửa sổ xác nhận xóa cột được chọn
        confirm = messagebox.askokcancel(
            "Delete Column", f"Are you sure you want to delete {column_name}?")
        if confirm:
            df = df.drop(column_name, axis=1)
            tree_list = list(df)
            target_combobox["values"] = tree_list
            trv_refresh()
    except Exception as e:
        print("Error:", e)


def show_popup(event):
    global trv, df
    try:
        # Lấy index của cột được chọn
        column = trv.identify_column(event.x)
        # Lấy tên cột được chọn
        column_name = trv.heading(column)['text']
        # Hiển thị cửa sổ với 2 nút edit và delete với tên cột được chọn
        popup_menu = tk1.Menu(trv, tearoff=0)
        popup_menu.add_command(
            label="Edit", command=lambda: edit_column(column_name))
        popup_menu.add_command(
            label="Delete", command=lambda: delete_column(column_name))
        popup_menu.tk_popup(event.x_root, event.y_root)
    except Exception as e:
        print("Error:", e)


def upload_file():
    # Di chuyển import vào hàm upload_to_GUI
    from gui import count_label, path_label, target_combobox
    global file_name, str1, df, tree_list
    f_types = [('All', "*.*")]
    file = filedialog.askopenfilename(filetypes=f_types)
    # Nếu không chọn file thì thoát hàm
    if not file:
        return
    else:
        file_name = os.path.basename(file)
        # nếu file là csv thì đọc pd.read csv còn excel thì pd.read_excel và các loại file khác
        if file_name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
        tree_list = list(df)  # List of column names as header
        str1 = "Rows:" + str(df.shape[0]) + " , Columns:" + str(df.shape[1])

        path_label.config(text=file_name)
        count_label.config(text=str1)
        trv_refresh()
        target_combobox["values"] = tree_list

        check_dataset(df)


def trv_refresh(r_set=None):  # Refresh the Treeview to reflect changes
    from gui import dataTab, ttk
    global df, tree_list, trv
    if r_set is None:
        r_set = df.to_numpy().tolist()  # create list of list using rows

    # Kiểm tra xem treeview đã tồn tại chưa, nếu có thì xóa đi
    for widget in dataTab.winfo_children():
        if isinstance(widget, ttk.Treeview):
            widget.destroy()
        if isinstance(widget, ttk.Scrollbar):
            widget.destroy()

    # Tạo Treeview mới
    trv = ttk.Treeview(dataTab, selectmode='browse',
                       height=10, show='headings', columns=tree_list)
    trv.grid(row=5, column=1, columnspan=3, padx=10, pady=20)
    trv.bind("<Button-3>", show_popup)

    # Định nghĩa các cột
    for i in trv['columns']:
        trv.column(i, width=90, anchor='c', stretch=False)
        trv.heading(i, text=str(i))

    # Thêm dữ liệu vào Treeview
    for dt in r_set:
        v = [r for r in dt]
        trv.insert("", 'end', values=v)

    # Thêm thanh scrollbar dọc
    vs = ttk.Scrollbar(dataTab, orient="vertical", command=trv.yview)
    trv.configure(yscrollcommand=vs.set)
    vs.grid(row=5, column=4, sticky='ns')

    # Thêm thanh scrollbar ngang chứa columns
    hs = ttk.Scrollbar(dataTab, orient="horizontal", command=trv.xview)
    trv.configure(xscrollcommand=hs.set)
    hs.grid(row=6, column=0, columnspan=3, sticky='ew')

    trv.columnconfigure(0, weight=1)
    trv.rowconfigure(0, weight=1)

    # Gọi lại hàm my_columns để cập nhật các checkbox input
    my_columns()

# Clean data functions


def check_dataset(df):
    from gui import duplicate_lbl, null_lbl, outlier_lbl, other_lbl
    global duplicate_rows, null_values, outlier_count, other_issues, special_char_issues
    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        duplicate_lbl.config(text=f"Duplicate Rows: {duplicate_rows}")
    else:
        duplicate_lbl.config(text="Duplicate Rows: 0")

    # Check for null values
    null_values = df.isnull().sum().sum()
    if null_values > 0:
        null_lbl.config(text=f"Null Values: {null_values}")
    else:
        null_lbl.config(text="Null Values: 0")

    # Check for outliers
    outlier_count = 0
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count += len(df[(df[column] < lower_bound)
                                 | (df[column] > upper_bound)])
    if outlier_count > 0:
        outlier_lbl.config(text=f"Outliers: {outlier_count}")
    else:
        outlier_lbl.config(text="Outliers: 0")

    other_issues = 0
    special_char_issues = 0
    # Duyệt có các cột có kiểu dữ liệu chuỗi và kiểm tra xem có kí tự đặc biệt không
    for column in df.columns:
        if df[column].dtype == 'object':
            special_char_issues += len(
                df[df[column].str.contains(r'[^a-zA-Z0-9]', na=False)])

    other_issues = special_char_issues  # + other common issues + other issues

    if other_issues > 0:
        other_lbl.config(text=f"Other Issues: {other_issues}")
    else:
        other_lbl.config(text="Other Issues: 0")


def remove_duplicate():
    global df, duplicate_rows
    # Nếu duplicate_rows <= 0 thì hiện thông báo và thoát hàm
    if duplicate_rows <= 0:
        messagebox.showinfo("Info", "No duplicate rows found")
        return
    else:
        df = df.drop_duplicates()
        trv_refresh()
        check_dataset(df)


def remove_null():
    global df, null_values
    # Nếu null_values <= 0 thì hiện thông báo và thoát hàm
    if null_values <= 0:
        messagebox.showinfo("Info", "No null values found")
        return
    else:
        df = df.dropna()
        trv_refresh()
        check_dataset(df)


def replace_null():
    from gui import replace_null_combobox
    global df
    method = replace_null_combobox.get()
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Nếu null_values <= 0 thì hiện thông báo và thoát hàm
    if null_values <= 0:
        messagebox.showinfo("Info", "No null values found")
        return
    else:
        if method == "Mean":
            df[numeric_columns] = df[numeric_columns].fillna(
                df[numeric_columns].mean())
        elif method == "Median":
            df[numeric_columns] = df[numeric_columns].fillna(
                df[numeric_columns].median())
        elif method == "Mode":
            mode_values = df[numeric_columns].mode().iloc[0]
            df[numeric_columns] = df[numeric_columns].fillna(mode_values)
        trv_refresh()
        check_dataset(df)


def remove_outliers():
    global df, outlier_count
    # Nếu outlier_count <= 0 thì hiện thông báo và thoát hàm
    if outlier_count <= 0:
        messagebox.showinfo("Info", "No outliers found")
        return
    else:
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[column] >= lower_bound) &
                        (df[column] <= upper_bound)]
        trv_refresh()
        check_dataset(df)


def other_cleaning():
    global df, other_issues, special_char_issues
    # Nếu other_issues <= 0 thì hiện thông báo và thoát hàm
    if other_issues <= 0:
        messagebox.showinfo("Info", "No other issues found")
        return
    else:
        # Nếu special_char_issues > 0 thì duyệt qua các cột và xóa các kí tự đặc biệt
        if special_char_issues > 0:
            for column in df.columns:
                if df[column].dtype == 'object':  # Chỉ áp dụng cho các cột có kiểu dữ liệu chuỗi
                    df[column] = df[column].apply(lambda x: re.sub(
                        r'[^a-zA-Z0-9]', '', str(x)) if pd.notnull(x) else x)
        trv_refresh()
        check_dataset(df)


def save_dataset():
    global df
    # Chọn nơi lưu file
    save_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[
                                             ("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    if save_file:
        # Lưu file dưới dạng csv hoặc excel
        if save_file.endswith('.csv'):
            df.to_csv(save_file, index=False)
        elif save_file.endswith('.xlsx'):
            df.to_excel(save_file, index=False)
        else:
            df.to_csv(save_file, index=False)
        messagebox.showinfo("Info", "File saved successfully")

# Model functions


def my_columns():
    from gui import input_label_cb, tk
    global i, my_ref, selected_checkboxes, checkbox_widgets
    i = 1  # Initialize a counter for row placement of checkboxes
    my_ref = {}  # Dictionary to store references to checkboxes
    selected_checkboxes = []  # Initialize the list of selected checkboxes
    input_label_cb.config(text=" ")  # Remove the previous checkboxes

    # Remove previous checkboxes
    for cb in checkbox_widgets:
        cb.destroy()
    checkbox_widgets = []

    # Loop through each column in the dataset
    col_count = 0
    row_count = 0
    for column in tree_list:
        var = IntVar()  # Create a Tkinter variable to store checkbox state (0 or 1)
        cb = Checkbutton(input_label_cb, text=column, variable=var)
        cb.grid(row=row_count, column=col_count, padx=5, sticky='w')

        # Store the checkbox variable reference in a dictionary
        my_ref[column] = var
        col_count += 1  # Increment column counter

        # Move to next row after 2 columns
        if col_count == 2:
            col_count = 0
            row_count += 1

        # Append the checkbox widget and its variable to the list
        selected_checkboxes.append((column, var))
        checkbox_widgets.append(cb)

    input_label_cb.grid(row=3, column=0, padx=5, sticky=tk.W)


def execute_model():
    from gui import model_combobox, target_combobox
    global model_train  # Di chuyển câu lệnh global lên đầu hàm
    target_variable = target_combobox.get()
    input_variables = [column for column,
                       var in selected_checkboxes if var.get() == 1]

    # nếu target_variable hoặc input_variables rỗng thì hiện cửa sổ thông báo và thoát hàm
    if not target_variable or not input_variables:
        messagebox.showinfo("Info", "Please select target and input variables")
        return
    else:
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

        # future scaling
        st_x = StandardScaler()
        X_train = st_x.fit_transform(X_train)
        X_test = st_x.transform(X_test)

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
        elif model == "Decision Tree":
            model_train = DecisionTreeClassifier(criterion="entropy")
            print("Decision Tree")
        elif model == "Random Forest":
            model_train = RandomForestClassifier(
                n_estimators=10, criterion="entropy")
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
                plt.text(0.5, 0.95, f"Accuracy: {
                         accuracy}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
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
            plt.text(0.5, 0.95, f"R-squared: {r2}", ha='center',
                     va='top', transform=plt.gca().transAxes, fontsize=10)
            plt.show()
        elif isinstance(model_train, DecisionTreeClassifier) or isinstance(model_train, RandomForestClassifier):
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            X_set, y_set = X_test, y_test
            x1, x2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                                 np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
            plt.contourf(x1, x2, model_train.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                         alpha=0.75, cmap=ListedColormap(('purple', 'green')))
            plt.xlim(x1.min(), x1.max())
            plt.ylim(x2.min(), x2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c=ListedColormap(('purple', 'green'))(i), label=j)
            plt.title('Random Forest Classification')
            plt.xlabel('Independent ')
            plt.ylabel('Dependent')
            plt.legend()
            plt.text(0.5, 0.95, f"Accuracy: {
                     accuracy}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
            plt.show()

# Visualize functions


def selected_vs_type(event):
    from gui import vs_type_combobox, column1_label, column1_combobox, column2_label, column2_combobox, vs_graph_type_combobox
    quantitative_columns = [
        column for column in tree_list if df[column].dtype in ['int64', 'float64']]
    categorical_columns = [
        column for column in tree_list if df[column].dtype == 'object']
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

        column2_label.grid(row=2, column=0, padx=5, pady=5)
        column2_combobox.grid(row=2, column=1, padx=5, pady=5)

        vs_graph_type_combobox["values"] = graph_type_3
    elif selected == "1 Categorical - 1 Quantitative":
        column1_label.config(text="Categorical")
        column2_label.config(text="Quantitative")

        column1_combobox["values"] = categorical_columns
        column2_combobox["values"] = quantitative_columns

        column2_label.grid(row=2, column=0, padx=5, pady=5)
        column2_combobox.grid(row=2, column=1, padx=5, pady=5)

        vs_graph_type_combobox["values"] = graph_type_4
    elif selected == "2 Quantitative":
        column1_label.config(text="Quantitative 1")
        column2_label.config(text="Quantitative 2")

        column1_combobox["values"] = quantitative_columns
        column2_combobox["values"] = quantitative_columns

        column2_label.grid(row=2, column=0, padx=5, pady=5)
        column2_combobox.grid(row=2, column=1, padx=5, pady=5)

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
    df.groupby(column1)[column2].value_counts(
    ).unstack().plot(kind='bar', stacked=True)
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
    from gui import vs_type_combobox, column1_combobox, column2_combobox, vs_graph_type_combobox
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


def on_target_combobox_select(event):
    from gui import target_combobox
    selected = target_combobox.get()
    tree_list_copy = [x for i, x in enumerate(tree_list) if x != selected]
    print(tree_list_copy)
    my_columns2(tree_list_copy)


def my_columns2(valid_input_column):
    from gui import input_label_cb, tk
    global i, my_ref, selected_checkboxes, checkbox_widgets
    i = 1  # Initialize a counter for row placement of checkboxes
    my_ref = {}  # Dictionary to store references to checkboxes
    selected_checkboxes = []  # Initialize the list of selected checkboxes
    input_label_cb.config(text=" ")  # Remove the previous checkboxes

    # Remove previous checkboxes
    for cb in checkbox_widgets:
        cb.destroy()
    checkbox_widgets = []

    # Loop through each column in the dataset
    col_count = 0
    row_count = 0
    for column in valid_input_column:
        var = IntVar()  # Create a Tkinter variable to store checkbox state (0 or 1)
        cb = Checkbutton(input_label_cb, text=column, variable=var)
        cb.grid(row=row_count, column=col_count, padx=5, sticky='w')

        # Store the checkbox variable reference in a dictionary
        my_ref[column] = var
        col_count += 1  # Increment column counter

        # Move to next row after 2 columns
        if col_count == 2:
            col_count = 0
            row_count += 1

        # Append the checkbox widget and its variable to the list
        selected_checkboxes.append((column, var))
        checkbox_widgets.append(cb)

    input_label_cb.grid(row=3, column=0, padx=5, sticky=tk.W)
