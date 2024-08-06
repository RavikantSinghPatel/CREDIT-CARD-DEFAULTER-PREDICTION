


     

Importing the Dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
     

# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv("creditcard.csv")
     

# first 5 rows of the dataset
credit_card_data.head()
     
	Time 	V1 	V2 	V3 	V4 	V5 	V6 	V7 	V8 	V9 	... 	V21 	V22 	V23 	V24 	V25 	V26 	V27 	V28 	Amount 	Class
0 	0 	-1.359807 	-0.072781 	2.536347 	1.378155 	-0.338321 	0.462388 	0.239599 	0.098698 	0.363787 	... 	-0.018307 	0.277838 	-0.110474 	0.066928 	0.128539 	-0.189115 	0.133558 	-0.021053 	149.62 	0.0
1 	0 	1.191857 	0.266151 	0.166480 	0.448154 	0.060018 	-0.082361 	-0.078803 	0.085102 	-0.255425 	... 	-0.225775 	-0.638672 	0.101288 	-0.339846 	0.167170 	0.125895 	-0.008983 	0.014724 	2.69 	0.0
2 	1 	-1.358354 	-1.340163 	1.773209 	0.379780 	-0.503198 	1.800499 	0.791461 	0.247676 	-1.514654 	... 	0.247998 	0.771679 	0.909412 	-0.689281 	-0.327642 	-0.139097 	-0.055353 	-0.059752 	378.66 	0.0
3 	1 	-0.966272 	-0.185226 	1.792993 	-0.863291 	-0.010309 	1.247203 	0.237609 	0.377436 	-1.387024 	... 	-0.108300 	0.005274 	-0.190321 	-1.175575 	0.647376 	-0.221929 	0.062723 	0.061458 	123.50 	0.0
4 	2 	-1.158233 	0.877737 	1.548718 	0.403034 	-0.407193 	0.095921 	0.592941 	-0.270533 	0.817739 	... 	-0.009431 	0.798278 	-0.137458 	0.141267 	-0.206010 	0.502292 	0.219422 	0.215153 	69.99 	0.0

5 rows × 31 columns

# Last 5 rows of the dataset
credit_card_data.tail()
     
	Time 	V1 	V2 	V3 	V4 	V5 	V6 	V7 	V8 	V9 	... 	V21 	V22 	V23 	V24 	V25 	V26 	V27 	V28 	Amount 	Class
5969 	6634 	-1.611463 	0.190648 	0.901715 	1.531254 	-1.535865 	0.799245 	1.513786 	0.495829 	0.200390 	... 	0.211223 	0.007477 	1.026272 	0.057628 	-0.024955 	-0.368263 	0.081684 	0.140669 	458.92 	0.0
5970 	6635 	-1.420272 	1.449354 	1.320110 	-1.894320 	0.913695 	0.454601 	0.894179 	-0.385450 	2.433841 	... 	-0.529027 	-0.368394 	-0.247773 	-1.189156 	-0.126040 	0.701487 	0.277333 	-0.222694 	0.77 	0.0
5971 	6637 	-1.206696 	0.284728 	2.152053 	-2.850437 	-0.437285 	-0.238376 	-0.333341 	0.334679 	2.870542 	... 	0.039460 	0.464476 	-0.457193 	-0.556105 	0.517579 	0.008006 	0.366054 	0.185008 	14.00 	0.0
5972 	6644 	1.067611 	0.091006 	-0.153917 	0.704233 	0.113894 	-0.826866 	0.567690 	-0.464181 	0.957295 	... 	-0.476723 	-1.410090 	-0.037550 	-0.177773 	0.321810 	0.114930 	-0.109640 	0.023205 	139.90 	0.0
5973 	6645 	-0.535272 	-0.132299 	2.180041 	1.018303 	-1.498819 	0.529570 	0.420147 	0.045445 	1.543919 	... 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN 	NaN

5 rows × 31 columns

credit_card_data.info()
     

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5974 entries, 0 to 5973
Data columns (total 31 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Time    5974 non-null   int64  
 1   V1      5974 non-null   float64
 2   V2      5974 non-null   float64
 3   V3      5974 non-null   float64
 4   V4      5974 non-null   float64
 5   V5      5974 non-null   float64
 6   V6      5974 non-null   float64
 7   V7      5974 non-null   float64
 8   V8      5974 non-null   float64
 9   V9      5974 non-null   float64
 10  V10     5974 non-null   float64
 11  V11     5974 non-null   float64
 12  V12     5974 non-null   float64
 13  V13     5974 non-null   float64
 14  V14     5974 non-null   float64
 15  V15     5974 non-null   float64
 16  V16     5974 non-null   float64
 17  V17     5974 non-null   float64
 18  V18     5973 non-null   float64
 19  V19     5973 non-null   float64
 20  V20     5973 non-null   float64
 21  V21     5973 non-null   float64
 22  V22     5973 non-null   float64
 23  V23     5973 non-null   float64
 24  V24     5973 non-null   float64
 25  V25     5973 non-null   float64
 26  V26     5973 non-null   float64
 27  V27     5973 non-null   float64
 28  V28     5973 non-null   float64
 29  Amount  5973 non-null   float64
 30  Class   5973 non-null   float64
dtypes: float64(30), int64(1)
memory usage: 1.4 MB


# Checking the no. of missing values in each column
credit_card_data.isnull().sum()
     

Time      0
V1        0
V2        0
V3        0
V4        0
V5        0
V6        0
V7        0
V8        0
V9        0
V10       0
V11       0
V12       0
V13       0
V14       0
V15       0
V16       0
V17       0
V18       1
V19       1
V20       1
V21       1
V22       1
V23       1
V24       1
V25       1
V26       1
V27       1
V28       1
Amount    1
Class     1
dtype: int64


# Distribution of legit transaction and fraudulent transaction
credit_card_data['Class'].value_counts()
     

Class
0.0    5970
1.0       3
Name: count, dtype: int64

This DataSet Highly Unbalanced

0 --> Normal Transaction

1 --> Fraudulant Transaction

# Saparating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
     

print(legit.shape)
print(fraud.shape)
     

(5970, 31)
(3, 31)


# Stastical measures of the data
legit.Amount.describe()
     

count    5970.000000
mean       64.965707
std       192.429839
min         0.000000
25%         4.450000
50%        15.620000
75%        56.485000
max      7712.430000
Name: Amount, dtype: float64


fraud.Amount.describe()
     

count      3.000000
mean     256.310000
std      264.880121
min        0.000000
25%      119.965000
50%      239.930000
75%      384.465000
max      529.000000
Name: Amount, dtype: float64


# Compaire the values for both transaction
credit_card_data.groupby('Class').mean()
     
	Time 	V1 	V2 	V3 	V4 	V5 	V6 	V7 	V8 	V9 	... 	V20 	V21 	V22 	V23 	V24 	V25 	V26 	V27 	V28 	Amount
Class 																					
0.0 	2677.40201 	-0.264965 	0.285625 	0.844580 	0.102656 	0.000958 	0.195420 	0.018542 	-0.039195 	0.397472 	... 	0.055426 	-0.043268 	-0.161540 	-0.036683 	0.028985 	0.089890 	-0.040132 	0.025238 	0.006163 	64.965707
1.0 	1780.00000 	-2.553039 	0.184644 	-0.293711 	2.872264 	0.005330 	-0.855718 	-0.549831 	0.308239 	-1.093098 	... 	0.599742 	0.294921 	-0.177321 	0.361160 	-0.020311 	0.056068 	-0.170050 	0.015979 	-0.086847 	256.310000

2 rows × 30 columns

Under-Sampling

Build a sample dataset containing similar distribution of normal transaction Fraudulent Transaction

Number of Fraudulant Transaction --> 3

legit_sample = legit.sample(n = 3)
     

Concatenating two DataFrames

new_dataset = pd.concat([legit_sample, fraud], axis = 0)
     

new_dataset.head()
     
	Time 	V1 	V2 	V3 	V4 	V5 	V6 	V7 	V8 	V9 	... 	V21 	V22 	V23 	V24 	V25 	V26 	V27 	V28 	Amount 	Class
454 	332 	1.084303 	0.127678 	1.389853 	2.532559 	-0.636871 	0.651109 	-0.685289 	0.356924 	-0.052520 	... 	-0.055487 	-0.088642 	-0.012251 	-0.026491 	0.290882 	-0.039353 	0.033400 	0.022966 	11.34 	0.0
1444 	1121 	1.410012 	-1.354061 	1.053231 	-1.234618 	-1.746970 	0.330006 	-1.606554 	0.159226 	-1.075106 	... 	-0.225977 	-0.051159 	0.035677 	-0.273577 	0.232300 	-0.146318 	0.095326 	0.028719 	26.00 	0.0
3291 	2834 	-0.518657 	0.891789 	1.460511 	1.067438 	0.632649 	0.552317 	0.638102 	0.020176 	-0.321206 	... 	-0.040376 	0.353851 	-0.222866 	-0.281085 	-0.105248 	-0.261398 	0.123383 	-0.018735 	3.82 	0.0
541 	406 	-2.312227 	1.951992 	-1.609851 	3.997906 	-0.522188 	-1.426545 	-2.537387 	1.391657 	-2.770089 	... 	0.517232 	-0.035049 	-0.465211 	0.320198 	0.044519 	0.177840 	0.261145 	-0.143276 	0.00 	1.0
623 	472 	-3.043541 	-3.157307 	1.088463 	2.288644 	1.359805 	-1.064823 	0.325574 	-0.067794 	-0.270953 	... 	0.661696 	0.435477 	1.375966 	-0.293803 	0.279798 	-0.145362 	-0.252773 	0.035764 	529.00 	1.0

5 rows × 31 columns

new_dataset.tail()
     
	Time 	V1 	V2 	V3 	V4 	V5 	V6 	V7 	V8 	V9 	... 	V21 	V22 	V23 	V24 	V25 	V26 	V27 	V28 	Amount 	Class
1444 	1121 	1.410012 	-1.354061 	1.053231 	-1.234618 	-1.746970 	0.330006 	-1.606554 	0.159226 	-1.075106 	... 	-0.225977 	-0.051159 	0.035677 	-0.273577 	0.232300 	-0.146318 	0.095326 	0.028719 	26.00 	0.0
3291 	2834 	-0.518657 	0.891789 	1.460511 	1.067438 	0.632649 	0.552317 	0.638102 	0.020176 	-0.321206 	... 	-0.040376 	0.353851 	-0.222866 	-0.281085 	-0.105248 	-0.261398 	0.123383 	-0.018735 	3.82 	0.0
541 	406 	-2.312227 	1.951992 	-1.609851 	3.997906 	-0.522188 	-1.426545 	-2.537387 	1.391657 	-2.770089 	... 	0.517232 	-0.035049 	-0.465211 	0.320198 	0.044519 	0.177840 	0.261145 	-0.143276 	0.00 	1.0
623 	472 	-3.043541 	-3.157307 	1.088463 	2.288644 	1.359805 	-1.064823 	0.325574 	-0.067794 	-0.270953 	... 	0.661696 	0.435477 	1.375966 	-0.293803 	0.279798 	-0.145362 	-0.252773 	0.035764 	529.00 	1.0
4920 	4462 	-2.303350 	1.759247 	-0.359745 	2.330243 	-0.821628 	-0.075788 	0.562320 	-0.399147 	-0.238253 	... 	-0.294166 	-0.932391 	0.172726 	-0.087330 	-0.156114 	-0.542628 	0.039566 	-0.153029 	239.93 	1.0

5 rows × 31 columns

new_dataset['Class'].value_counts()
     

Class
0.0    3
1.0    3
Name: count, dtype: int64


new_dataset.groupby('Class').mean()
     
	Time 	V1 	V2 	V3 	V4 	V5 	V6 	V7 	V8 	V9 	... 	V20 	V21 	V22 	V23 	V24 	V25 	V26 	V27 	V28 	Amount
Class 																					
0.0 	1429.0 	0.658552 	-0.111531 	1.301198 	0.788459 	-0.583731 	0.511144 	-0.551247 	0.178775 	-0.482944 	... 	-0.081668 	-0.107280 	0.071350 	-0.06648 	-0.193718 	0.139311 	-0.149023 	0.084037 	0.010983 	13.72
1.0 	1780.0 	-2.553039 	0.184644 	-0.293711 	2.872264 	0.005330 	-0.855718 	-0.549831 	0.308239 	-1.093098 	... 	0.599742 	0.294921 	-0.177321 	0.36116 	-0.020311 	0.056068 	-0.170050 	0.015979 	-0.086847 	256.31

2 rows × 30 columns

Splitting the data into Features & Targets

X = new_dataset.drop(columns='Class', axis = 1)
Y = new_dataset['Class']
     

print(X)
     

      Time        V1        V2        V3        V4        V5        V6  \
454    332  1.084303  0.127678  1.389853  2.532559 -0.636871  0.651109   
1444  1121  1.410012 -1.354061  1.053231 -1.234618 -1.746970  0.330006   
3291  2834 -0.518657  0.891789  1.460511  1.067438  0.632649  0.552317   
541    406 -2.312227  1.951992 -1.609851  3.997906 -0.522188 -1.426545   
623    472 -3.043541 -3.157307  1.088463  2.288644  1.359805 -1.064823   
4920  4462 -2.303350  1.759247 -0.359745  2.330243 -0.821628 -0.075788   

            V7        V8        V9  ...       V20       V21       V22  \
454  -0.685289  0.356924 -0.052520  ... -0.151190 -0.055487 -0.088642   
1444 -1.606554  0.159226 -1.075106  ... -0.305190 -0.225977 -0.051159   
3291  0.638102  0.020176 -0.321206  ...  0.211376 -0.040376  0.353851   
541  -2.537387  1.391657 -2.770089  ...  0.126911  0.517232 -0.035049   
623   0.325574 -0.067794 -0.270953  ...  2.102339  0.661696  0.435477   
4920  0.562320 -0.399147 -0.238253  ... -0.430022 -0.294166 -0.932391   

           V23       V24       V25       V26       V27       V28  Amount  
454  -0.012251 -0.026491  0.290882 -0.039353  0.033400  0.022966   11.34  
1444  0.035677 -0.273577  0.232300 -0.146318  0.095326  0.028719   26.00  
3291 -0.222866 -0.281085 -0.105248 -0.261398  0.123383 -0.018735    3.82  
541  -0.465211  0.320198  0.044519  0.177840  0.261145 -0.143276    0.00  
623   1.375966 -0.293803  0.279798 -0.145362 -0.252773  0.035764  529.00  
4920  0.172726 -0.087330 -0.156114 -0.542628  0.039566 -0.153029  239.93  

[6 rows x 30 columns]


print(Y)
     

454     0.0
1444    0.0
3291    0.0
541     1.0
623     1.0
4920    1.0
Name: Class, dtype: float64

Split the data into Training Data & Testing Data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)
     

print(X.shape, X_train.shape, X_test.shape)
     

(6, 30) (4, 30) (2, 30)

Model Training

Logistic Regression

model = LogisticRegression()
     

# training the Logistic Model with Training Data
model.fit(X_train, Y_train)
     

LogisticRegression()

In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

Model Evaluation

Accuracy Score

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
     

print('Accuracy on Training Data : ',training_data_accuracy)
     

Accuracy on Training Data :  1.0


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
     

print('Accuracy on Test Data : ',test_data_accuracy)
     

Accuracy on Test Data :  0.5



     
