# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.

2.Import the dataset to operate on.

3.Split the dataset.

4.Predict the required output.

5.End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: M.Vidya Neela
RegisterNumber:  212221230120
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![204126417-3c757f24-0327-41e0-a3d4-90dd1a4528e7](https://user-images.githubusercontent.com/94169318/204435921-e02e478c-3353-4fb3-bfd6-cb392aa21b0a.png)

![204126433-66650b34-6924-4038-87ea-5481eedd4917](https://user-images.githubusercontent.com/94169318/204435935-f708a174-19c2-45a9-9737-5ef1e1b85e47.png)

![204126437-5f159cb8-684e-416c-adc6-c648f17a0c3d](https://user-images.githubusercontent.com/94169318/204435960-ad0637e2-0dea-4293-8c1e-19218e5e1662.png)

![204126446-9e129516-60d8-4c08-b307-d59f247e318e](https://user-images.githubusercontent.com/94169318/204435982-223cb1ee-5cb9-49e6-ab58-496b593e4294.png)

![204126457-a9afae17-4936-4a88-af9d-ed0fcb8a8612](https://user-images.githubusercontent.com/94169318/204436007-39af4f6d-3419-4bfd-90e9-c9ac6549e922.png)

![204126460-d35603b6-2f6f-445c-8c31-1055bc3378d8](https://user-images.githubusercontent.com/94169318/204436021-b941b198-12e3-44f8-8ae3-db13a61ab2ff.png)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
