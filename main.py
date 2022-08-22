import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

diabetes_X = diabetes.data

# print(diabetes_X)
# print(diabetes.DESCR)

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]


model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_predicted = model.predict(diabetes_X_test)

print("Mean square error is: ", mean_squared_error(diabetes_y_test , diabetes_y_predicted))
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# plt.scatter(diabetes_X_test, diabetes_y_test)
# plt.plot(diabetes_X_test, diabetes_y_predicted)
# plt.show()

# Mean square error is:  3035.060115291269
# Weights:  [941.43097333]
# Intercept:  153.39713623331644