import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv")

# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] #attributes

# print(data.head())

predict = "G3"      #label

X = np.array(data.drop(["G3"],1)) #training data
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    # saving the model -> so you dont need to rerun everytime

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

# compare results of test vs actual results

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()