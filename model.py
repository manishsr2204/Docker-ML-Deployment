
# Import the libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Import the dataset
data = pd.read_csv('Dataset/50_Startups.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values


data.head()



# One Hot Encoding categorical data
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Fitting Multiple Linear Regression to the Training set
regression = LinearRegression()
regression.fit(X_train, y_train)



# Predicting the Test set results
y_pred = regression.predict(X_test)


df = pd.DataFrame(data=y_test, columns=['y_test'])
df['y_pred'] = y_pred



# Predicting the sigle observation results. Here 1,0,0 represents that the state is Calfornia
a = [1,0,0,160349,134321,401400]
b = np.array(a)
b = b.reshape(1, -1)
y_pred_single_obs = regression.predict(b)
round(float(y_pred_single_obs), 2)

r2_score(y_test, y_pred)


from sklearn.externals import joblib
joblib.dump(regression, "model.pkl")


print("model saved")

#from sklearn.externals import joblib
#NewYork = 1
#California = 0
#Florida = 0
#RnD_Spend = 160349
#Administration_Spend = 134321
#Marketing_Spend = 401400
#pred_args = [NewYork,California,Florida,RnD_Spend,Administration_Spend,Marketing_Spend]
#pred_args_arr = np.array(pred_args)
#pred_args_arr = pred_args_arr.reshape(1, -1)
#mul_reg = open("multiple_regression_model.pkl","rb")
#ml_model = joblib.load(mul_reg)
#model_prediction = ml_model.predict(pred_args_arr)

#round(float(model_prediction), 2)






