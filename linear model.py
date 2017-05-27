import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn import datasets,linear_model


x_parameter=[[150],[200],[250],[300],[350],[400],[600]]
y_parameter=[6450,7450,8450,9450,11450,15450,18450]

#print (x_parameter)
#print (y_parameter)

def linear_model_main(x_parameter,y_parameter,predict_value):
    regr=linear_model.LinearRegression()
    regr.fit(x_parameter,y_parameter)
    predict_outcome=regr.predict(predict_value)

    predictions={}
    predictions['intercept']=regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions



def show_linear_line(X_parameters,Y_parameters):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters,Y_parameters,color='blue')
    plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.show()



predict_squre=700
result=linear_model_main(x_parameter,y_parameter,predict_squre)
#print ("Intercept value " , result['intercept'])
#print ("coefficient" , result['coefficient'])
print ("Predicted value: ",result['predicted_value'])

show_linear_line(x_parameter,y_parameter)


