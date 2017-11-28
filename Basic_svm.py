
from sklearn import svm

## Regression analysis

# data
X = [[0, 0], [2, 2]]

#labels
y = [0.5, 2.5]

#machine learn SVM
clf = svm.SVR()

# Train our SVM
clf.fit(X, y)

#test Data
Test_data=[[1.5, 1]]

# # test trained algorithm
# print clf.predict(Test_data)

# Basic classification

#Data
X = [[0, 0], [1, 1], [5,5]]
#labels
y = [0, 1, 3]

#classsifier SVM model
clf = svm.SVC()

#Training model
clf.fit(X, y)

#Prediction
print clf.predict([[4, 4]])
# # get support vectors
# print clf.support_vectors_
# # get indices of support vectors
# print clf.support_
# # get number of support vectors for each class
# print clf.n_support_

