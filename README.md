# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Nanda Kishor S P 
RegisterNumber: 212224040210
*/
```

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Placement_Data.csv')
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:, :-1]
x
y = data1["status"]
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
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```

## Output:
### Head Values - DataFrame
![image](https://github.com/user-attachments/assets/b7084282-3ceb-4257-9c6c-990a1f9a6553)

### Head Values - DataFrame Copy
![image](https://github.com/user-attachments/assets/09e8ade9-9733-4f4c-8d20-d3069ab6341d)

### Sum - Null Values
![image](https://github.com/user-attachments/assets/c227b71b-98f3-480c-8609-4132ad142032)

### Sum - Duplicated
![image](https://github.com/user-attachments/assets/46566be5-5dac-4e0f-89f8-aa2cf348b191)

### Label Encoded - DataFrame
![image](https://github.com/user-attachments/assets/a942afd9-943d-4766-bf57-733e8bcaeac7)

### Predicted Y values for Test Data
![image](https://github.com/user-attachments/assets/e787d500-6ca2-46fc-91dc-3e245056e9fc)

### Accuracy Score
![image](https://github.com/user-attachments/assets/1cd41a3e-264a-4fc4-87f5-d0bc726ce64b)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/927fd5ae-8854-4839-bacc-758991460348)
![image](https://github.com/user-attachments/assets/0be5d6bb-9446-408d-b165-644f3f47c59a)

### Classification Report
![image](https://github.com/user-attachments/assets/1171f487-5a3e-4193-a739-f2fa8b827b52)

### Final Prediction Result
![image](https://github.com/user-attachments/assets/dfe1d392-e88d-422d-a8d4-218f8645aef1)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
