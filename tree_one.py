from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import graphviz

wine = load_wine()
len(wine.data)

x = wine.data
y = wine.target

np.unique(y)

pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis=1)
wine.feature_names
wine.target_names

# 划分数据集
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)
xtrain.shape,ytrain.shape
# 建模训练
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(xtrain,ytrain)
score = clf.score(xtest,ytest)
print(score)
# 可视化
feature_names = ["酒精","苹果酸","灰","灰的碱性","镁","总酚","类黄酮","非黄烷类酚类","花青素","颜色强度","色调","od280/od315稀释葡萄酒","脯氨酸"]

dot_data = tree.export_graphviz(clf,
                                feature_names=feature_names,
                                class_names=["琴酒","雪莉","贝尔摩德"],
                                filled=True,
                                rounded=True
)
graph = graphviz.Source(dot_data)
graph.view()
