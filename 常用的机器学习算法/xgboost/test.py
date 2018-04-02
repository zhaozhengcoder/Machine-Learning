from numpy import loadtxt 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

from xgboost import plot_importance
from matplotlib import pyplot

print ("ok")

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#---------------------------二选一执行-----------------------------#
#choice one
#model = XGBClassifier()
#model.fit(X_train, y_train)

#choice two
model = XGBClassifier() 
eval_set = [(X_test, y_test)] 
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True) 

#---------------------------   END   -----------------------------#

#可视化
plot_importance(model)
pyplot.show()


y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

