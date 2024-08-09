import tkinter as tk
from tkinter import ttk

from data_function import upload_file, execute_model, execute_visualize, selected_vs_type, remove_duplicate, remove_null, replace_null, remove_outliers, other_cleaning, save_dataset, on_target_combobox_select

root = tk.Tk()
root.title('App version 1.3.6')

left_frame = tk.LabelFrame(root, text='Choose File')
left_frame.grid(row=0, column=0, padx=10, pady=10)

right_frame = tk.LabelFrame(root)
right_frame.grid(row=0, column=1, padx=10, pady=10)

tabControl = ttk.Notebook(right_frame)
cleanTab = ttk.Frame(tabControl)
dataTab = ttk.Frame(right_frame)
modelTab = ttk.Frame(tabControl)
visualizeTab = ttk.Frame(tabControl)

# tabControl.add(dataTab, text='Data')
tabControl.add(cleanTab, text='Clean Data')
tabControl.add(modelTab, text='Model')
tabControl.add(visualizeTab, text='Visualize')
dataTab.grid(row=0, column=0, padx=10, pady=10)
tabControl.grid(row=1, column=0, padx=10, pady=10)

# Clean tab
clean_title = tk.Label(cleanTab, text="Choose Cleaning Method:")
clean_title.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

duplicate_lbl = tk.Label(cleanTab, text="Duplicate Rows: ")
duplicate_lbl.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
duplicate_btn = tk.Button(
    cleanTab, text="Remove Duplicate", command=remove_duplicate)
duplicate_btn.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

null_lbl = tk.Label(cleanTab, text="Null Values: ")
null_lbl.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
null_btn = tk.Button(cleanTab, text="Remove Null", command=remove_null)
null_btn.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

replace_null_combobox = ttk.Combobox(
    cleanTab, text="Select Method", values=["Mean", "Median", "Mode"], width=10)
replace_null_combobox.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
replace_null_btn = tk.Button(
    cleanTab, text="Replace Null", command=replace_null)
replace_null_btn.grid(row=3, column=2, padx=5, pady=5, sticky=tk.W)

outlier_lbl = tk.Label(cleanTab, text="Outliers: ")
outlier_lbl.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
outlier_btn = tk.Button(cleanTab, text="Remove Outliers",
                        command=remove_outliers)
outlier_btn.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

other_lbl = tk.Label(cleanTab, text="Other: ")
other_lbl.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
other_btn = tk.Button(cleanTab, text="Other Cleaning", command=other_cleaning)
other_btn.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)

save_btn = tk.Button(cleanTab, text="Save Dataset", command=save_dataset)
save_btn.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)

# Data tab
my_font1 = ('times', 12, 'bold')
path_label = tk.Label(left_frame, text='Choose File:',
                      width=23, font=my_font1)
path_label.grid(row=1, column=1)
browse_btn = tk.Button(left_frame, text='Browse File',
                       width=20, command=lambda: upload_file())
browse_btn.grid(row=2, column=1, pady=5)

count_label = tk.Label(left_frame, width=20, text='',
                       bg='lightyellow')
count_label.grid(row=3, column=1, padx=5)

# Model tab
target_label = tk.Label(modelTab, text="Select Target Variable")
target_label.grid(row=0, column=0, padx=5, sticky=tk.W)

target_combobox = ttk.Combobox(modelTab)
target_combobox.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
target_combobox.bind("<<ComboboxSelected>>", on_target_combobox_select)

input_label = tk.Label(modelTab, text="Select Input Variables")
input_label.grid(row=2, column=0, padx=5, sticky=tk.W)

input_label_cb = tk.Label(modelTab)

model_label = tk.Label(modelTab, text="Choose Model")
model_label.grid(row=0, column=3, padx=50, pady=10, sticky=tk.W)

model_combobox = ttk.Combobox(
    modelTab, values=["Logistic Regression", "KNN",
                      "Linear Regression", "Decision Tree", "Random Forest"]
)
model_combobox.grid(row=1, column=3, padx=50, sticky=tk.W)

execution_button = tk.Button(modelTab, text="Execution", command=execute_model)
execution_button.grid(row=2, column=3, padx=50, pady=10)

# Visualize tab
visualize_title = tk.Label(visualizeTab, text="Visualize Type")
visualize_title.grid(row=0, column=0, padx=5, pady=5)

vs_type_combobox = ttk.Combobox(visualizeTab, values=["1 Quantitative", "1 Categorical", "2 Categorical",
                                                      "1 Categorical - 1 Quantitative", "2 Quantitative"], width=40)
vs_type_combobox.grid(row=0, column=1, padx=5, pady=5)
vs_type_combobox.bind("<<ComboboxSelected>>", selected_vs_type)

column1_label = tk.Label(visualizeTab, text="Column 1")
column1_label.grid(row=1, column=0, padx=5, pady=5)

column2_label = tk.Label(visualizeTab, text="Column 2")
column2_label.grid(row=2, column=0, padx=5, pady=5)

column1_combobox = ttk.Combobox(visualizeTab)
column1_combobox.grid(row=1, column=1, padx=5, pady=5)

column2_combobox = ttk.Combobox(visualizeTab)
column2_combobox.grid(row=2, column=1, padx=5, pady=5)

vs_graph_type_label = tk.Label(visualizeTab, text="Graph Type")
vs_graph_type_label.grid(row=3, column=0, padx=5, pady=5)

vs_graph_type_combobox = ttk.Combobox(visualizeTab)
vs_graph_type_combobox.grid(row=3, column=1, padx=5, pady=5)

excute_btn = tk.Button(visualizeTab, text="Execute", command=execute_visualize)
excute_btn.grid(row=4, column=1, padx=5, pady=5)

root.mainloop()  # Keep the window open
