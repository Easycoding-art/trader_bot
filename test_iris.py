from sklearn import datasets
import numpy as np
from sklearn.linear_model import SGDClassifier
'''
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X, y)
clf.predict([[2., 2.]])
SGDClassifier(max_iter=5)
'''
# Загружаем набор данных Ирисы:
iris = datasets.load_iris()
# Смотрим на названия переменных
#print(iris.feature_names)
# Смотрим на данные, выводим 10 первых строк: 
#print(iris.data[:10])
# Смотрим на целевую переменную:
#print(iris.target_names)
#print(iris.target)
h, s = iris.data.shape
#print(h)
dataset = np.column_stack((iris.data, iris.target))
np.random.shuffle(dataset)
print(dataset)
train = dataset[0:146, :]
test = dataset[147:, :]
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
#clf.fit(iris.data, iris.target)
clf.fit(train[:, 0:4], train[:, 4:])
print(f'x: {train[:, 0:4].shape}  y: {train[:, 4:].shape}')
print(test[:, 0:4])
print(clf.predict(test[:, 0:4]))
print(test[:, 4:])