#Import Libraries and Define Auxiliary Function
# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier

#This function is to plot the confusion matrix.

def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 

#Load the dataframe
URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
data = pd.read_csv(URL1)

print(data.head())

URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
X = pd.read_csv(URL2)
print(X.head(100))

#TASK 1
#Create a NumPy array from the column Class in data, by applying the method to_numpy() 
# then assign it to the variable Y,make sure the output is a Pandas series (only one bracket 
# df['name of column']).
Y = data['Class'].to_numpy()
print(Y)

#TASK 2
#Standardize the data in X then reassign it to the variable X using the transform provided below.
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

print(X[:5])   

#TASK 3
#Use the function train_test_split to split the data X and Y into training and test data. 
# Set the parameter test_size to 0.2 and random_state to 2. 
# The training data and test data should be assigned to the following labels.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}, {Y_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}, {Y_test.shape}")
print(Y_test.shape)

#task 4
#Create a logistic regression object then create a GridSearchCV object logreg_cv with cv = 10. 
# Fit the object to find the best parameters from the dictionary parameters.

parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()
logreg_cv = GridSearchCV(lr, parameters, cv=10)
logreg_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

#TASK 5
#Calculate the accuracy on the test data using the method score
accuracy = logreg_cv.score(X_test, Y_test)

print(f"Precisión en los datos de prueba: {accuracy:.4f}")

#Lets look at the confusion matrix:
yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

#TASK 6
#Create a support vector machine object then create a GridSearchCV object svm_cv with cv = 10. 
# Fit the object to find the best parameters from the dictionary parameters.

parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()

svm_cv = GridSearchCV(svm, parameters, cv=10, scoring='accuracy')

svm_cv.fit(X, Y)

print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)

#TASK 7
#Calculate the accuracy on the test data using the method score:

accuracy = svm_cv.best_estimator_.score(X_test, Y_test)

print("Precisión en los datos de prueba:", accuracy)

#We can plot the confusion matrix
yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

#TASK 8
#Create a decision tree classifier object then create a GridSearchCV object tree_cv with cv = 10. 
# Fit the object to find the best parameters from the dictionary parameters.
parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, parameters, cv=10)

tree_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)

#TASK 9
#Calculate the accuracy of tree_cv on the test data using the method score
accuracy = tree_cv.score(X_test, Y_test)

print("Precisión en datos de prueba:", accuracy)

#We can plot the confusion matrix
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

#TASK 10
#Create a k nearest neighbors object then create a GridSearchCV object knn_cv with cv = 10. 
# Fit the object to find the best parameters from the dictionary parameters.

parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()

knn_cv = GridSearchCV(KNN, parameters, cv=10)
knn_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)

#TASK 11
#Calculate the accuracy of knn_cv on the test data using the method score:

accuracy = knn_cv.score(X_test, Y_test)

print("Precisión en datos de prueba:", accuracy)

#We can plot the confusion matrix
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

#task 12

#Find the method performs best:
# Accuracy scores for each model
logreg_accuracy = logreg_cv.score(X_test, Y_test)
svm_accuracy = svm_cv.best_estimator_.score(X_test, Y_test)
tree_accuracy = tree_cv.score(X_test, Y_test)
knn_accuracy = knn_cv.score(X_test, Y_test)

print(f"Logistic Regression Accuracy: {logreg_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"Decision Tree Accuracy: {tree_accuracy:.4f}")
print(f"KNN Accuracy: {knn_accuracy:.4f}")

# Determine the best model
accuracies = {
    "Logistic Regression": logreg_accuracy,
    "SVM": svm_accuracy,
    "Decision Tree": tree_accuracy,
    "KNN": knn_accuracy
}

best_model = max(accuracies, key=accuracies.get)
print(f"The best model is: {best_model} with an accuracy of {accuracies[best_model]:.4f}")















