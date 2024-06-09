from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import r2_score

df1 = pd.read_csv("tennis_stats.csv")
df2 = pd.read_csv("diabetes.csv")

#Simple linear regression based off of 1 input feature (Nx1 data set) with one theta and one bias 


class SimpleLinearRegression:
    def __init__(self,df):
        self.alpha = .001
        self.epochs = 1
        self.x = df["Wins"]
        self.y = df["Winnings"]
        self.w0 = 0
        self.w1 = 0

    def calculate_gradient(self):
        c = -2 * (1/len(self.x))
        dJdw0 = c * sum([(self.y[i] - (self.x[i] * self.w1 + self.w0)) for i in range(len(self.x))])
        dJdw1 = c * sum([self.x[j]*(self.y[j] - (self.x[j] * self.w1 + self.w0)) for j in range(len(self.x))])
        gradientJ = [dJdw0,dJdw1]
        return gradientJ

    def gradient_descent(self):
        for _ in range(self.epochs):
          gradientJ = self.calculate_gradient()
          self.w0 = self.w0 - self.alpha * gradientJ[0]
          self.w1 = self.w1 - self.alpha * gradientJ[1]

    def predict(self,x):
        return x * self.w1 + self.w0


    
class MultipleLinearRegression:
    def __init__(self,df):
        self.alpha = .01 #1e-4 for GD :: .01 for SGD are optimized alphas
        self.epochs = 1000
        self.x_train, self.x_test, self.y_train, self.y_test = self.preprocessing(df)
        self.weights = np.asmatrix([0 for _ in range(self.x_train.shape[1])]).transpose()
        
    def preprocessing(self,df):
        df["Bias"] = [1 for _ in range(len(df))]
        X = df[["Bias", "Losses", "DoubleFaults", "Wins"]]
        y = df[["Winnings"]]
        X = scale(X) #if not going to scale input data, need to tune hyperperameters for non-scaled data to optimize accuracy
        for i in range(X.shape[0]):
            X[i][0] = 1
        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state=5)
        x_train, x_test, y_train, y_test = np.asmatrix(x_train), np.asmatrix(x_test), np.asmatrix(y_train), np.asmatrix(y_test)
        return x_train, x_test, y_train, y_test
    
    def shapes(self):
        print("X_train shape: " + str(self.x_train.shape))
        print("X_test shape: " + str(self.x_test.shape))
        print("Y_train shape: " + str(self.y_train.shape))
        print("Y_test shape: " + str(self.y_test.shape))
        print("Weights shape: " + str(self.weights.shape))
        
    def stochastic_gradient_descent(self):
        for _ in range(self.epochs):
            rand_i = randint(0,self.x_train.shape[0]-1)
            gradient = -2 * np.dot(self.x_train[rand_i].transpose(),self.y_train[rand_i]) + 2 * np.dot(np.dot(self.x_train[rand_i].transpose(),self.x_train[rand_i]),self.weights)
            self.weights = self.weights - self.alpha * gradient 
            
    def gradient_descent(self):
        for _ in range(self.epochs):
            gradient = -2 * np.dot(self.x_train.transpose(),self.y_train) + 2 * np.dot(np.dot(self.x_train.transpose(),self.x_train),self.weights)
            self.weights = self.weights - self.alpha * gradient
              
    def predict(self,inpt):
        yHat = np.dot(inpt,self.weights)
        return yHat
    
    def metrics(self):
        self.alpha = .01
        self.stochastic_gradient_descent()
        yHat_train = self.predict(self.x_train)
        yHat_test = self.predict(self.x_test)
        print("R^2 score for testing data using STD: " + str(r2_score(self.y_test,yHat_test)))
        print("R^2 score for training data using STD: " + str(r2_score(self.y_train,yHat_train)))
        #plt.scatter(yHat_train,self.y_train)
        #plt.show()
        
        self.alpha = 1e-4
        self.gradient_descent()
        yHat_train = self.predict(self.x_train)
        yHat_test = self.predict(self.x_test)
        print("R^2 score for testing data using GD: " + str(r2_score(self.y_test,yHat_test)))
        print("R^2 score for training data using GD: " + str(r2_score(self.y_train,yHat_train))) 
        #plt.scatter(yHat_train,self.y_train)
        #plt.show()
        
    
#LR = SimpleLinearRegression(df1)
#LR.gradient_descent()
#plt.scatter(df1["Wins"],df1["Winnings"])
#predictions = [df1["Wins"][i] * LR.w1 + LR.w0 for i in range(len(df1["Wins"]))]
#plt.plot(df1["Wins"],predictions,color="red")
#plt.show()
#print(LR.w0)

LR = MultipleLinearRegression(df1)
LR.metrics()




df = pd.read_csv("train.csv")

df["Bias"] = [1 for _ in range(len(df))]
X = df[["Bias","OverallQual","YearBuilt","YearRemodAdd","FullBath","GarageCars","GarageArea","GrLivArea","TotalBsmtSF"]]
y = df[["SalePrice"]]

X = np.asmatrix(X)
weights = np.asmatrix([0 for _ in range(X.shape[1])]).transpose()

LR = SimpleLinearRegression(df1)
LR.gradient_descent()
plt.scatter(df1["Wins"],df1["Winnings"])
predictions = [df1["Wins"][i] * LR.w1 + LR.w0 for i in range(len(df1["Wins"]))]
print("R squared: " + str(r2_score(LR.y,predictions)))
plt.plot(df1["Wins"],predictions,color="red")
plt.show()

LR = MultipleLinearRegression(df1)
LR.metrics()


#print(df1.corr()["Winnings"])
#study correlation matrix to find best features to use that have biggest impact on output
#dataframe.corr()["output"] usually > .7 good to use as features in dataset

#Shown that gradient descent results in better R^2 score than stochastic gradient descent for this dataset with no noticeable change to time complexity

#study correlation matrix to find best features to use that have biggest impact on output
#dataframe.corr()["output"] usually > .5 good to use as features in dataset
