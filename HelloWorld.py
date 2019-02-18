from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0],[145,1]]
labels=[0,0,1,1,0]

clf = tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)
print clf.predict([[155,0]])
print clf.predict([[146,0]])


