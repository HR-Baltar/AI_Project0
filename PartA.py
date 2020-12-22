# Rey Baltar, hrb217

## PART A PROGRAM

##  SOURCES
##https://medium.com/@belen.sanchez27/predicting-iris-flower-species-with-k-means-clustering-in-python-f6e46806aaee
##https://www.datacamp.com/community/tutorials/machine-learning-python
##https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/
##https://medium.com/codebagng/basic-analysis-of-the-iris-data-set-using-python-2995618a6342

from sklearn import datasets
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.preprocessing import scale

print("1: Iris\n2: Digits")

var = input("input: ")

#   Irist Dataset   #
if var == '1':
    #load dataset

    data = datasets.load_iris()

    #display data
    print(data.data)
    print(data.target)
    print(data.target_names)
    # display box and whiskers
    data_box = data.data #data array
    target_box = data.data #lables array
    sns.boxplot(data = data_box, width = 0.5, fliersize=5)
    sns.set(rc={'figure.figsize':(2,15)})
    plt.show()

    # target and predictors for K means clustering specifically

    x_k = data.data[:, :2]
    y_k = data.target

    #scatter plot
    plt.scatter(x_k[:,0], x_k[:,1], c=y_k, cmap='gist_rainbow')
    plt.xlabel('Spea1 Length', fontsize=18)
    plt.ylabel('Sepal Width', fontsize=18)
    plt.show()

    # kmeans instance
    kmc = KMeans(n_clusters=3, n_jobs=4, random_state=24)
    kmc.fit(x_k)

    # identify center points of data
    centers = kmc.cluster_centers_

    #  which cluster data belongs
    new_labels=kmc.labels_

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(16,8))
    axes[0].scatter(x_k[:, 0], x_k[:, 1], c=y_k, cmap='gist_rainbow',
    edgecolor='k', s=150)
    axes[1].scatter(x_k[:, 0], x_k[:, 1], c=new_labels, cmap='gist_rainbow',
    edgecolor='k', s=150)
    axes[0].set_xlabel('Sepal length', fontsize=18)
    axes[0].set_ylabel('Sepal width', fontsize=18)
    axes[1].set_xlabel('Sepal length', fontsize=18)
    axes[1].set_ylabel('Sepal width', fontsize=18)
    axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
    axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
    axes[0].set_title('Actual', fontsize=18)
    axes[1].set_title('Predicted', fontsize=18)
    plt.show()


    # moving on to other algorithms
    # define x and y
    
    x = data.data
    y = data.target


    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=11)

    print('\n\n')
    print ("Decision Tree :")
    d_tree = DecisionTreeClassifier()
    d_tree.fit(x_train, y_train)
    predicts = d_tree.predict(x_test)
    print(accuracy_score(y_test, predicts))
    print(confusion_matrix(y_test, predicts))
    print(classification_report(y_test, predicts))

    print('\n\n')
    print ("K Nearest Neighbors :")
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    predicts = knn.predict(x_test)
    print(accuracy_score(y_test, predicts))
    print(confusion_matrix(y_test, predicts))
    print(classification_report(y_test, predicts))

    print('\n\n')
    print ("Backprobagation :")
    bp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
    bp.fit(x_train, y_train)
    predicts = bp.predict(x_test)
    print(accuracy_score(y_test, predicts))
    print(confusion_matrix(y_test, predicts))
    print(classification_report(y_test, predicts))

if var == '2':
    data = datasets.load_digits()

    print(data.data)
    print(data.target)
    print(data.target_names)

    #reshape to get 64 images
    data.images.reshape((1797, 64))


    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(data.images[i], cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(data.target[i]))
    plt.show()

    
    #scale data so that it can be plotted.
    scale_data = scale(data.data)
    x = scale_data
    y = data.target
    img = data.images
    
    #split data into training and test sets and instantiate KMeans model
    x_train, x_test, y_train, y_test, img_train, img_test = model_selection.train_test_split(x, y, img, test_size=.25, random_state=42)
    kmc = KMeans(init='k-means++', n_clusters=10, random_state=42)
    kmc.fit(x_train)
    predicts=kmc.predict(x_test)
    kmc.cluster_centers_.shape


    from sklearn.manifold import Isomap

    # fit the digits data
    x_iso = Isomap(n_neighbors=10).fit_transform(x_train)
    clusters = kmc.fit_predict(x_train)

    # actual vs predicted scatter plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle('Actual vs. Predicted', fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.85) 
    ax[0].scatter(x_iso[:, 0], x_iso[:, 1], c=y_train)
    ax[0].set_title('Actual Training Labels')
    ax[1].scatter(x_iso[:, 0], x_iso[:, 1], c=clusters)
    ax[1].set_title('Predicted Training Labels')
    plt.show()

    ## conufusion matrixes and accuracy scores
    print ("Decision Tree :")
    d_tree = DecisionTreeClassifier()
    d_tree.fit(x_train, y_train)
    predicts = d_tree.predict(x_test)
    print(accuracy_score(y_test, predicts))
    print(confusion_matrix(y_test, predicts))
    print(classification_report(y_test, predicts))
    print('\n\n')
    
    print ("K Nearest Neighbors :")
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    predicts = knn.predict(x_test)
    print(accuracy_score(y_test, predicts))
    print(confusion_matrix(y_test, predicts))
    print(classification_report(y_test, predicts))
    print('\n\n')
    
    print ("Backprobagation :")
    bp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
    bp.fit(x_train, y_train)
    predicts = bp.predict(x_test)
    print(accuracy_score(y_test, predicts))
    print(confusion_matrix(y_test, predicts))
    print(classification_report(y_test, predicts))
    
