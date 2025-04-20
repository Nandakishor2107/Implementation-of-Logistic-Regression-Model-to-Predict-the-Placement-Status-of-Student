# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load dataset, drop unnecessary columns, and encode categorical variables using LabelEncoder.
2. Separate features (x) and target (y), then split into training and testing sets.
3. Train a logistic regression model on the training data.
4. Predict on test data, evaluate accuracy and classification report, and make a sample prediction.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Nanda Kishor S P
RegisterNumber:  212224040210
*/
```

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('Placement_Data.csv')
df.head()
dfc = df.copy()
dfc = dfc.drop(["sl_no", "salary"], axis = 1)
dfc.head()
dfc.isnull().sum()
dfc.duplicated().sum()
le=LabelEncoder()
dfc["gender"] = le.fit_transform(dfc["gender"])
dfc["ssc_b"] = le.fit_transform(dfc["ssc_b"])
dfc["hsc_b"] = le.fit_transform(dfc["hsc_b"])
dfc["hsc_s"] = le.fit_transform(dfc["hsc_s"])
dfc["degree_t"] = le.fit_transform(dfc["degree_t"])
dfc["workex"] = le.fit_transform(dfc["workex"])
dfc["specialisation"] = le.fit_transform(dfc["specialisation"])
dfc["status"] = le.fit_transform(dfc["status"])
dfc
x = dfc.iloc[:, :-1]
x
y = dfc["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
from sklearn.metrics import classification_report
classification_report_ = classification_report(y_test, y_pred)
# print(classification_report_)
print(lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]))
```


## Output:

### Head Values - DataFrame
![alt text](<Screenshot 2025-04-20 183809.png>)  

### Head Values - DataFrame Copy(Dropped)
![alt text](<Screenshot 2025-04-20 201248.png>)  

### Sum - Null Values
![alt text](<Screenshot 2025-04-20 201341.png>)  

### Sum - Duplicated
![alt text](<Screenshot 2025-04-20 201441.png>)  

### DataFrame - LabelEncoded
![alt text](<Screenshot 2025-04-20 201613.png>)  

### Predicted Y values
![alt text](<Screenshot 2025-04-20 201713.png>)  

### Accuracy Score
![alt text](<Screenshot 2025-04-20 201817.png>)  

### Confusion Matrix
![alt text](<Screenshot 2025-04-20 202038.png>)
![alt text](<Screenshot 2025-04-20 202056.png>)  

### Classification Report
![alt text](<Screenshot 2025-04-20 202205.png>)  

### Model Prediction for input
![alt text](<Screenshot 2025-04-20 202903.png>)  


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
