############################### using a decision tree #############################

from sklearn import tree

features = [[140,1],[130,1],[150,0],[170,0],[145,1]]
labels=[0,0,1,1,0]

clf = tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)
print clf.predict_proba([[2., 2.]])
print clf.predict([[155,0]])
print clf.predict([[146,0]])

########################### visualizing a decision tree ############################
print "-----------------"

import numpy as np
from sklearn.datasets import load_iris

iris= load_iris()
test_idx = [0,50,100]

#training data
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf.fit(train_data,train_target)

print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[60]
print test_target
print clf.predict(test_data)

