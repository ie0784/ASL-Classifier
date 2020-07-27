###############################################
# LETS DO THIS!
# We will import the pictures, convert them to grey scale with the hand being black
# then we will use methods like in Lab 3 and Lab 4 to train and evaluate models based
# on the grey scale images at identifying the ASL alphabet
# 
# Group Project
# Brad Eddowes, Isaiah English, Shane Allison
# 4/10/2019
# 
# Loop through all pictures once loaded and then give converted pictures and label array to models described in the second to last lab
#
# Measure accuracy, ROC, and what not and create pretty graphs
###############################################

#openCV method
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

#SGD Stuff
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
#End of SGD

#Decision tree stuff
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn import metrics
#End of Decision trees stuff

def openCVPConvertMethod(filePathName):
    image = cv2.imread(filePathName)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    light_skin = (100, 50, 0)
    dark_skin = (250, 250, 250)

    #create mask
    mask = cv2.inRange(hsv_image, light_skin, dark_skin)

    #result = cv2.bitwise_and(image, image, mask=mask)

    """ print(mask)
    print(mask.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()  """

    #reduce the image, 200x200 is too large
    final = np.ravel(mask.reshape(50,mask.shape[0]//50,50,mask.shape[1]//50).sum(axis=1).sum(axis=2))
    for j, z in enumerate(final):
        final[j] = (final[j] / 4)

    return final

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

 
#openCVPConvertMethod('./K_TRAIN.jpg')

#openCVPConvertMethod('./Y_TRAIN.jpg')

X = list()
y = list()
#This method interates through all files in the training directory, building the lists with the picture data
def getPictures(): 
    directory = './asl-alphabet-TRAIN/asl_alphabet_train/asl_alphabet_train'

    for filename in sorted(os.listdir(directory)):
        print(filename)
        for picture in sorted(os.listdir(directory + '/' + filename)):
            print(picture)
            y.append(filename)

            temp = openCVPConvertMethod(directory + '/' + filename + '/' + picture)

            X.append(list(temp))
#This was just a wrapper for the above method to be able to check its output
def getAndProcessPictureData():
    getPictures()

    #make lists into numpy arrays to make it easier to work with
    y = np.array(y)
    X = np.array(X)

    print(y)
    #print(X[2])

    print(y.shape)
    print(X.shape)

    #show one
    """ print(y[2]) 
    some_digit = X[2]
    some_digit_image = some_digit.reshape(50, 50)
    plt.imshow(some_digit_image, cmap = 'Greys', interpolation="nearest")
    plt.axis("off")
    plt.show() """

    np.save('labels.npy', y)
    np.save('pictureData.npy', X)
#Everying below here is working with the saved numpy arrays from the OG picture data.
#The main program, working with the loaded picture data and then training on it. 
def mainProgram():
    #getAndProcessPictureData()

    labels = np.load('labels.npy', y)
    data = np.load('pictureData.npy', X)

    print(data.shape)
    print(labels[72000]) 
    some_digit = data[72000]
    some_digit_image = some_digit.reshape(50, 50)
    plt.imshow(some_digit_image, cmap = 'Greys', interpolation="nearest")
    plt.axis("off")
    plt.show()

    data_new = data

    #crop picture data more? 
    """ data_new = np.zeros((84000, 1600))
    for i, a in enumerate(data):
        print('%.2f Complete.' % ((i/84000) * 100))
        some_digit = data[i]
        some_digit_image = some_digit.reshape(50, 50)[5:45,5:45] 
        data_new[i]= some_digit_image.ravel() """
    print(labels[72000]) 
    some_digit = data_new[72000]
    #some_digit_image = some_digit.reshape(40, 40)
    some_digit_image = some_digit.reshape(50, 50)
    plt.imshow(some_digit_image, cmap = 'Greys', interpolation="nearest")
    plt.axis("off")
    plt.show()

    #reduce to just black and white?
    #reduce values...
    """ for i, a in enumerate(data_new):
        print('%.2f Complete.' % ((i/84000) * 100))
        for j, b in enumerate(a):
            if b <= 127:
                data_new[i][j] = 0.
            else:
                data_new[i][j] = 255.  """  
    print(labels[72000]) 
    some_digit = data_new[72000]
    #some_digit_image = some_digit.reshape(40, 40)
    some_digit_image = some_digit.reshape(50, 50)
    plt.imshow(some_digit_image, cmap = 'Greys', interpolation="nearest")
    plt.axis("off")
    plt.show() 

    #Shuffle it.....
    shuffle_index = np.random.permutation(84000)
    data_new_shuff, labels_shuff = data_new[shuffle_index], labels[shuffle_index]

    #Split into training and test data....
    X_train, X_test, y_train, y_test = data_new_shuff[:80000], data_new_shuff[80000:], labels_shuff[:80000], labels_shuff[80000:]

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    #SGD Classifier
    y_train_Y = (y_train == 'Y') 
    y_test_Y = (y_test == 'Y') 

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_Y)

    print("Prediction Score")
    print(sgd_clf.score(X_test,y_test_Y))

    print("Cross val scores, SGD")
    print(cross_val_score(sgd_clf, X_train, y_train_Y, cv=3, scoring="accuracy"))

    y_scores = cross_val_predict(sgd_clf, X_train, y_train_Y, cv=3, method="decision_function")

    precisions, recalls, thresholds = precision_recall_curve(y_train_Y, y_scores)

    average_precision = average_precision_score(y_train_Y, y_scores)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    plt.step(recalls, precisions, color='b', alpha=0.2,where='post')
    plt.fill_between(recalls, precisions, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])    
    plt.title('One V All-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()

    #print(precisions)
    #print(recalls)
    #print(thresholds)

    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_train_Y, y_scores)

    plot_roc_curve(fpr, tpr)
    plt.show() 

    #Decision Trees....
    f = list()

    for x in range(2500):
        f.append('pixel' + str(x))

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train_Y)
    y_pred = tree_clf.predict(X_test)
    print("Binary accuracy")
    print(metrics.accuracy_score(y_test_Y,y_pred))

    export_graphviz(
        tree_clf,
        out_file=("./BinaryY.dot"),
        feature_names=f,
        class_names=str(tree_clf.classes_),
        rounded=True,
        filled=True
    )

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    print("Multiclass accuracy")
    print(metrics.accuracy_score(y_test,y_pred))

    export_graphviz(
        tree_clf,
        out_file=("./Multiclass.dot"),
        feature_names=f,
        class_names=str(tree_clf.classes_),
        rounded=True,
        filled=True
    ) 

#RUN IT
mainProgram()
