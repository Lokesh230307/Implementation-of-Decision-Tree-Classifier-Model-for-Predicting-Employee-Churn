# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data: Read CSV data, handle nulls, encode categorical features like "salary".
2. Feature-target split: Select relevant features for x and set y as the "left" column.
3. Train-test split & modeling: Split the data and train a DecisionTreeClassifier using the "entropy" criterion..
4. Evaluate & predict: Measure accuracy on the test set and make predictions on new data

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: LOKESH S
RegisterNumber:  212224230143
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data

data.head()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![decision tree classifier model](sam.png)
![Screenshot 2025-05-14 112126](https://github.com/user-attachments/assets/56f6d8e1-ab6c-41ce-b645-db992babefb2)
![Screenshot 2025-05-14 112146](https://github.com/user-attachments/assets/01845365-48cf-4761-bd73-5b29c3b2dc22)
![Screenshot 2025-05-14 112207](https://github.com/user-attachments/assets/d28e9897-6d6c-4030-bd0e-a5e6d3e9575b)
![Screenshot 2025-05-14 112225](https://github.com/user-attachments/assets/b2ed64d4-47cb-4eef-ae4e-098b74502541)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
