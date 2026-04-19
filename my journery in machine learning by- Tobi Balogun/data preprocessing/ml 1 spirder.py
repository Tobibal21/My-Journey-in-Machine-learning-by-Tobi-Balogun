# Ran in spirder environment
# importing libaries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# data preprocessing 
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy='median')
imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])


# Encoding categorical data 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Transform first column (index 0) of X
ct = ColumnTransformer(
    [('one_hot', OneHotEncoder(), [0])],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# Reshape Y to 2D if it's 1D
if Y.ndim == 1:
    Y = Y.reshape(-1, 1)

# Transform last column of Y
ct = ColumnTransformer(
    [('one_hot', OneHotEncoder(), [-1])],
    remainder='passthrough'
)
Y = ct.fit_transform(Y)


# splitting the dataset 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2 , random_state = 0 )
# feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print("X (features):")
print(X)
print("\nY (target):")
print(Y)