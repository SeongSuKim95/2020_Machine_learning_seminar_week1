import sklearn as sk
from sklearn import datasets
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def main():

    iris_test = pd.read_csv('iris.csv',header=None)
    iris_test = np.array(iris_test)
    X = iris_test[:,2:4]
    y = iris_test[:,4] - 1
    y = y.astype(int)

    # iris = datasets.load_iris()
    # X = iris.data[:,[2,3]]
    # y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=0)

    sc = StandardScaler(with_mean=False,with_std=False)
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ppn = Perceptron(n_iter_no_change=40, eta0 =0.1, random_state = 0)
    ppn.fit(X_train_std,y_train)

    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples : %d'%(y_test != y_pred).sum())

    X_combined = np.vstack((X_train,X_test))
    X_combined_std = np.vstack((X_train_std,X_test_std))
    y_combined = np.hstack((y_train,y_test))
    plot_decision_regions(X=X_combined_std,y= y_combined,classifier =ppn, test_idx = range(105,150))

    plt.xlabel('petal length[standardized]')
    plt.ylabel('petal width[standardized]')
    plt.legend(loc='upper left')
    plt.show()

    ## SVM
    svm = SVC(kernel='rbf', C=1.0, random_state=0, gamma = 0.2)
    svm.fit(X_train, y_train)

    plot_decision_regions(X_combined,y_combined, classifier = svm, test_idx= range(105,150))
    plt.xlabel('petal length[standardized]')
    plt.ylabel('petal width[standardized]')
    plt.legend(loc='upper left')
    plt.show()

    ## KNN
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=knn, test_idx=range(105, 150))
    plt.xlabel('petal length[0 mean, 1 std]')
    plt.ylabel('petal width[0 mean, 1 std]')

    plt.legend(loc='upper left')
    plt.show()



    ## Tree
    tree = DecisionTreeClassifier(criterion = 'entropy',max_depth =3 , random_state = 0)
    tree.fit(X_train,y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined,y_combined,classifier=tree,test_idx=range(105,150))
    plt.xlabel('petal length[standardized]')
    plt.ylabel('petal width[standardized]')
    plt.legend(loc='upper left')
    plt.show()

    ##Random Forest
    forest = RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=1,n_jobs=2)
    forest.fit(X_train,y_train)
    plot_decision_regions(X_combined,y_combined,classifier=forest,test_idx=range(105,150))
    plt.xlabel('petal length[standardized]')
    plt.ylabel('petal width[standardized]')
    plt.legend(loc='upper left')
    plt.show()

def plot_decision_regions(X,y,classifier,test_idx =None, resolution = 0.02):

    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() +1
    x2_min, x2_max = X[:,1].min() - 1 ,X[:,1].max() +1

    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1,xx2,Z,alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #plot all samples

    X_test, y_test = X[test_idx,:],y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y==cl,0],y= X[y==cl,1],alpha = 0.8,c=cmap(idx),marker=markers[idx],label=cl)

    # if test_idx:
    #     X_test,y_test = X[test_idx,:],y[test_idx]
    #     plt.scatter(X_test[:,0],X_test[:,1],c='orange',alpha=1.0,linewidth=1,marker='p',s=55,label='test set')

main()