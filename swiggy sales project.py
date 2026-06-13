###import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#load data set
data=pd.read_csv("C:\imagecon\Dataset\swiggy_sales_dataset_nulls_yes_no.csv")
data
#data inspection
data.info()
data.shape
data.head()
data.tail()
data.describe()

#data cleaning
data.isnull().sum()

#fill na() for city
data["City"]=data["City"].fillna(data["City"].mode()[0])
#fill na() for age
data["Order_Value"]=data["Order_Value"].fillna(data["Order_Value"].median())
#fill na for dis percent
data["Discount_Percent"]=data["Discount_Percent"].fillna(data["Discount_Percent"].mean())
#fill na for rest rating
data["Restaurant_Rating"]=data["Restaurant_Rating"].fillna(data["Restaurant_Rating"].mean())



#covert string to numerical (label encoder)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["High_Sales"] = le.fit_transform(data["High_Sales"])
#splitting dataset
x=data.iloc[:,[4,6,8,7]].values
y=data.iloc[:,-1].values

#train test split
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.70,random_state=20)

#pre processing (standard scalar)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
#finding shape()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

#predection
y_pred=classifier.predict(x_test)
y_pred
#score
score=classifier.score(x_train,y_train)
score=classifier.score(x_test,y_test)
score

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm 

plt.figure(dpi=500)
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


import tkinter as tk
from tkinter import messagebox

def predict_sales():
    try:
        order_value = float(order_entry.get())
        rating = float(rating_entry.get())
        delivery_time = float(delivery_entry.get())
        distance = float(distance_entry.get())

        new_data = [[order_value, rating, delivery_time, distance]]

        new_data = sc.transform(new_data)

        prediction = classifier.predict(new_data)

        if prediction[0] == 1:
            result_label.config(
                text="🔥 HIGH SALES",
                fg="green"
            )
        else:
            result_label.config(
                text="📉 LOW SALES",
                fg="red"
            )

    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Swiggy Sales Prediction")
root.geometry("700x600")
root.configure(bg="#FC8019")

header = tk.Label(
    root,
    text="🍔 SWIGGY HIGH SALES PREDICTION",
    font=("Arial",22,"bold"),
    bg="#FC8019",
    fg="white"
)
header.pack(pady=20)

frame = tk.Frame(root,bg="white",bd=3,relief="ridge")
frame.pack(padx=30,pady=20,fill="both",expand=True)

tk.Label(frame,text="Order Value",
         font=("Arial",12,"bold"),
         bg="white").pack(pady=10)

order_entry = tk.Entry(frame,font=("Arial",12),width=30)
order_entry.pack()

tk.Label(frame,text="Restaurant Rating",
         font=("Arial",12,"bold"),
         bg="white").pack(pady=10)

rating_entry = tk.Entry(frame,font=("Arial",12),width=30)
rating_entry.pack()

tk.Label(frame,text="Delivery Time (Minutes)",
         font=("Arial",12,"bold"),
         bg="white").pack(pady=10)

delivery_entry = tk.Entry(frame,font=("Arial",12),width=30)
delivery_entry.pack()

tk.Label(frame,text="Distance (KM)",
         font=("Arial",12,"bold"),
         bg="white").pack(pady=10)

distance_entry = tk.Entry(frame,font=("Arial",12),width=30)
distance_entry.pack()

predict_btn = tk.Button(
    frame,
    text="PREDICT SALES",
    font=("Arial",14,"bold"),
    bg="#FC8019",
    fg="white",
    width=20,
    height=2,
    command=predict_sales
)
predict_btn.pack(pady=25)

result_label = tk.Label(
    frame,
    text="Prediction Result",
    font=("Arial",20,"bold"),
    bg="white"
)
result_label.pack(pady=20)

root.mainloop()
